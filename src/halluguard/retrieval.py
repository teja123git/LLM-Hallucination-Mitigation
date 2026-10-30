"""Hybrid retrieval: dense (FAISS inner product on L2-normalised embeddings) + sparse (Okapi BM25),
fused with Reciprocal Rank Fusion.  Supports ``mode`` = hybrid | dense | bm25 for retrieval ablations.

Fixes relative to the Stage 1 notebook:
  * BM25 uses a proper tokenizer (punctuation no longer glued to words) and stop-word removal;
  * documents with BM25 score 0 are *not* fed into the fusion (they used to enter with arbitrary ranks);
  * the last chunk of a document is not emitted twice as a fully-overlapped tail;
  * embeddings are cached on disk (re-building a 10k+ passage index is the slowest local step);
  * every hit records the rank in each stream, which makes retrieval errors analysable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .costs import local_timer
from .utils import get_logger, sha1, tokenize

log = get_logger()


# --------------------------------------------------------------------------- data classes
@dataclass
class Passage:
    pid: int
    text: str
    title: str = ""
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.title}: {self.text}" if self.title else self.text


@dataclass
class Hit:
    pid: int
    text: str
    title: str
    source: str
    score: float
    rank: int
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None

    @property
    def full_text(self) -> str:
        return f"{self.title}: {self.text}" if self.title else self.text

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- chunking
def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> List[str]:
    """Split ``text`` into overlapping word windows. Short texts give exactly one chunk."""
    if not text or not isinstance(text, str):
        return []
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if i + chunk_size >= len(words):  # reached the end: do not emit a redundant, fully-overlapped tail chunk
            break
    return chunks


# --------------------------------------------------------------------------- embedders
class Embedder:
    name = "base"

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size=64, show_progress=False):
        emb = self.model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress,
                                convert_to_numpy=True, normalize_embeddings=True)
        return np.ascontiguousarray(emb, dtype="float32")


class HashingEmbedder(Embedder):
    """Deterministic bag-of-ngrams embedder (no download).  Used for offline tests / mock mode only."""

    def __init__(self, dim: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing-{dim}"
        self.vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2", ngram_range=(1, 2), lowercase=True)

    def encode(self, texts, batch_size=64, show_progress=False):
        X = self.vec.transform(list(texts)).toarray().astype("float32")
        return np.ascontiguousarray(X)


def make_embedder(cfg) -> Embedder:
    if cfg.project.get("mock"):
        return HashingEmbedder()
    return SentenceTransformerEmbedder(cfg.retrieval.embed_model)


# --------------------------------------------------------------------------- retriever
class HybridRetriever:
    def __init__(self, embedder: Embedder, rrf_k: int = 60, candidates: int = 50, mode: str = "hybrid"):
        assert mode in ("hybrid", "dense", "bm25")
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.mode = mode
        self.passages: List[Passage] = []
        self.index = None
        self.bm25 = None
        self.embeddings: Optional[np.ndarray] = None

    # ---- build ---------------------------------------------------------------
    def build(self, passages: Sequence[Passage], cache_dir: Optional[str | Path] = None, show_progress: bool = True) -> "HybridRetriever":
        import faiss
        from rank_bm25 import BM25Okapi

        self.passages = list(passages)
        if not self.passages:
            raise ValueError("Cannot build a retriever over an empty corpus")
        texts = [p.full_text for p in self.passages]
        emb = None
        cache_file = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            fp = sha1(self.embedder.name, len(texts), *texts[:: max(1, len(texts) // 200)], texts[-1])[:16]
            cache_file = cache_dir / f"emb_{fp}.npy"
            if cache_file.exists():
                emb = np.load(cache_file)
                if emb.shape[0] != len(texts):
                    emb = None
                else:
                    log.info("Loaded cached embeddings: %s", cache_file.name)
        if emb is None:
            log.info("Embedding %d passages with %s ...", len(texts), self.embedder.name)
            emb = self.embedder.encode(texts, batch_size=64, show_progress=show_progress)
            if cache_file is not None:
                np.save(cache_file, emb)
        self.embeddings = emb
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.ascontiguousarray(emb, dtype="float32"))
        self.bm25 = BM25Okapi([tokenize(t, remove_stopwords=True) or ["_empty_"] for t in texts])
        return self

    @property
    def size(self) -> int:
        return len(self.passages)

    # ---- streams -------------------------------------------------------------
    def _dense(self, query: str, n: int) -> List[tuple]:
        qv = self.embedder.encode([query])
        scores, ids = self.index.search(qv, min(n, self.size))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def _sparse(self, query: str, n: int) -> List[tuple]:
        toks = tokenize(query, remove_stopwords=True)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n] if n < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]  # score 0 => no lexical evidence at all

    # ---- public --------------------------------------------------------------
    def search(self, query: str, k: int = 5, mode: Optional[str] = None) -> List[Hit]:
        mode = mode or self.mode
        with local_timer():
            n = max(self.candidates, k)
            dense = self._dense(query, n) if mode in ("hybrid", "dense") else []
            sparse = self._sparse(query, n) if mode in ("hybrid", "bm25") else []
            d_rank = {pid: r for r, (pid, _) in enumerate(dense, start=1)}
            s_rank = {pid: r for r, (pid, _) in enumerate(sparse, start=1)}
            if mode == "dense":
                ordered = [(pid, sc) for pid, sc in dense[:k]]
            elif mode == "bm25":
                ordered = [(pid, sc) for pid, sc in sparse[:k]]
            else:
                fused: Dict[int, float] = {}
                for r, (pid, _) in enumerate(dense, start=1):
                    fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.rrf_k + r)
                for r, (pid, _) in enumerate(sparse, start=1):
                    fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.rrf_k + r)
                ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
            hits = []
            for rank, (pid, score) in enumerate(ordered, start=1):
                p = self.passages[pid]
                hits.append(Hit(pid, p.text, p.title, p.source, float(score), rank, d_rank.get(pid), s_rank.get(pid)))
            return hits

    # ---- persistence of the corpus (embeddings are cached separately) -------------
    def save_passages(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def local_rank(query: str, texts: Sequence[str], k: int = 3) -> List[int]:
    """Tiny BM25 ranking over an *ad-hoc* list of passages (e.g. windows of a RAGTruth context)."""
    from rank_bm25 import BM25Okapi

    if len(texts) <= k:
        return list(range(len(texts)))
    bm = BM25Okapi([tokenize(t, True) or ["_empty_"] for t in texts])
    scores = bm.get_scores(tokenize(query, True) or ["_"])
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]
