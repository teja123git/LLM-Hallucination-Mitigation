"""Knowledge-base construction (leak-free by design).

Stage 1 indexed TruthfulQA's own *correct and incorrect answer strings* and the gold HaluEval passages as
"knowledge".  That lets RAG read the answer key and also mixes misconceptions into the evidence.  Here the
corpus only contains *source documents*:

  * every HaluEval-QA ``knowledge`` paragraph (Wikipedia text, ~10k) pooled together - the gold paragraph of
    an evaluation question is hidden among thousands of distractors (the usual "pooled corpus" setting);
  * the Wikipedia pages cited as sources by TruthfulQA questions (fetched once, cached on disk).

No reference answer, correct-answer list or incorrect-answer list is ever indexed.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import unquote, urlparse

from .config import Cfg, data_dir
from .retrieval import Embedder, HybridRetriever, Passage, chunk_text
from .utils import get_logger, sha1

log = get_logger()


class KBBuilder:
    """Collects passages with de-duplication and remembers which passages belong to which source document."""

    def __init__(self, chunk_size: int = 256, overlap: int = 50):
        self.chunk_size, self.overlap = chunk_size, overlap
        self.passages: List[Passage] = []
        self._seen: Dict[str, int] = {}
        self.doc_pids: Dict[str, List[int]] = {}

    def add_document(self, doc_key: str, text: str, title: str = "", source: str = "", meta: Optional[dict] = None) -> List[int]:
        pids: List[int] = []
        for ch in chunk_text(text, self.chunk_size, self.overlap):
            key = f"{title}||{ch.strip()}"
            if key in self._seen:
                pid = self._seen[key]
            else:
                pid = len(self.passages)
                self.passages.append(Passage(pid, ch.strip(), title, source, dict(meta or {}, doc=doc_key)))
                self._seen[key] = pid
            pids.append(pid)
        self.doc_pids.setdefault(doc_key, [])
        self.doc_pids[doc_key].extend(p for p in pids if p not in self.doc_pids[doc_key])
        return pids


# --------------------------------------------------------------------------- Wikipedia (TruthfulQA sources)
_URL_RE = re.compile(r"https?://[^\s;,]+", re.I)


def extract_urls(source_field: str) -> List[str]:
    return _URL_RE.findall(source_field or "")


def wikipedia_title_from_url(url: str) -> Optional[str]:
    p = urlparse(url)
    if not p.netloc.endswith("wikipedia.org") or "/wiki/" not in p.path:
        return None
    title = unquote(p.path.split("/wiki/", 1)[1]).replace("_", " ").strip()
    return title or None


def fetch_wikipedia_pages(titles: Sequence[str], cache_dir: Path, max_words: int = 1500,
                          user_agent: str = "halluguard-course-project/1.0", sleep: float = 0.15) -> Dict[str, Dict[str, str]]:
    """Fetch plain-text extracts through the MediaWiki API (one request per page; cached on disk).

    Returns {requested_title: {"title": canonical_title, "text": ...}}; titles that cannot be fetched are omitted.
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent})
    out: Dict[str, Dict[str, str]] = {}
    failures = 0
    for i, title in enumerate(titles):
        f = cache_dir / f"{sha1(title)[:20]}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("text"):
                out[title] = rec
            continue
        try:
            r = sess.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json",
                "titles": title, "formatversion": 2}, timeout=30)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", [])
            page = pages[0] if pages else {}
            text = (page.get("extract") or "").strip()
            # drop trailing reference-like sections
            text = re.split(r"\n==+\s*(See also|References|External links|Further reading|Notes|Sources)\s*==+", text)[0]
            text = re.sub(r"\n==+[^=\n]+==+\n", "\n", text)
            words = text.split()
            text = " ".join(words[:max_words])
            rec = {"title": page.get("title", title), "text": text}
            f.write_text(json.dumps(rec), encoding="utf-8")
            if text:
                out[title] = rec
        except Exception as e:  # network problems must not abort a long run
            failures += 1
            if failures <= 3:
                log.warning("Wikipedia fetch failed for %r: %s", title, e)
        time.sleep(sleep)
    if failures:
        log.warning("%d/%d Wikipedia pages could not be fetched (offline? blocked?). RAG coverage for TruthfulQA will be lower.", failures, len(titles))
    return out


# --------------------------------------------------------------------------- QA knowledge base
def build_qa_kb(cfg: Cfg, halueval: Sequence[dict], tqa: Sequence[dict], embedder: Embedder, tqa_pages: Optional[Dict[str, Dict[str, str]]] = None
                ) -> tuple[HybridRetriever, Dict[str, List[int]]]:
    """Build the retriever used by Baseline B and by all claim-verification stages.

    Returns (retriever, doc_pids) where ``doc_pids['halu_12']`` lists the passage ids holding the gold paragraph
    of HaluEval item 12 and ``doc_pids['tqa:<Wikipedia title>']`` those of a TruthfulQA source page.
    """
    kb = KBBuilder(cfg.retrieval.chunk_size, cfg.retrieval.chunk_overlap)
    if cfg.kb.halueval_pool:
        for it in halueval:
            if it.get("knowledge"):
                kb.add_document(it["id"], it["knowledge"], source="halueval")
    if tqa_pages:
        for title, rec in tqa_pages.items():
            kb.add_document(f"tqa:{title}", rec["text"], title=rec["title"], source="wikipedia")
    if not kb.passages:
        raise RuntimeError("Knowledge base is empty - enable kb.halueval_pool or kb.tqa_wikipedia")
    retr = HybridRetriever(embedder, rrf_k=cfg.retrieval.rrf_k, candidates=cfg.retrieval.candidates, mode=cfg.retrieval.mode)
    cache = data_dir(cfg, "kb", "mock" if cfg.project.get("mock") else "qa")
    retr.build(kb.passages, cache_dir=cache)
    log.info("QA knowledge base: %d passages (%d HaluEval docs, %d Wikipedia pages)", retr.size,
             sum(1 for k in kb.doc_pids if k.startswith("halu")), sum(1 for k in kb.doc_pids if k.startswith("tqa:")))
    return retr, kb.doc_pids


def build_pool_kb(cfg: Cfg, docs: Sequence[tuple], embedder: Embedder, name: str) -> tuple[HybridRetriever, Dict[str, List[int]]]:
    """Generic pooled KB from (doc_key, title, text) triples (used for the FEVER evidence pool)."""
    kb = KBBuilder(cfg.retrieval.chunk_size, cfg.retrieval.chunk_overlap)
    for key, title, text in docs:
        kb.add_document(key, text, title=title, source=name)
    retr = HybridRetriever(embedder, rrf_k=cfg.retrieval.rrf_k, candidates=cfg.retrieval.candidates, mode=cfg.retrieval.mode)
    retr.build(kb.passages, cache_dir=data_dir(cfg, "kb", "mock" if cfg.project.get("mock") else name))
    return retr, kb.doc_pids
