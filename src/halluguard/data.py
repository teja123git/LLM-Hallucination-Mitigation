"""Dataset loaders with a unified schema, plus leak-free train/dev/test splitting.

Primary sources (schemas verified against the real files):
  * HaluEval-QA  : RUCAIBox/HaluEval  data/qa_data.json   -> knowledge, question, right_answer, hallucinated_answer (10,000)
  * TruthfulQA   : sylinrl/TruthfulQA TruthfulQA.csv      -> 790 questions (the HF copy has 817; both are supported)
  * RAGTruth     : ParticleMedia/RAGTruth dataset/{response,source_info}.jsonl (response-level + span-level labels)
  * FEVER        : HuggingFace ``copenlu/fever_gold_evidence`` (claims with gold evidence sentences) - optional

All raw files are downloaded once into ``data/raw`` (see docs/DATA.md for manual download instructions).
In ``mock`` mode tiny bundled excerpts from ``data/toy`` are used instead (offline smoke tests only).
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import ROOT, Cfg, data_dir
from .kb import extract_urls, wikipedia_title_from_url
from .utils import get_logger, read_jsonl

log = get_logger()

URLS = {
    "halueval_qa": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
    "truthfulqa": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
    "ragtruth_response": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
    "ragtruth_source": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
}
TOY = ROOT / "data" / "toy"


def download(url: str, dest: Path, min_bytes: int = 1000, timeout: int = 120) -> Path:
    import requests

    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size < min_bytes:
        raise IOError(f"Downloaded file from {url} is suspiciously small ({tmp.stat().st_size} bytes)")
    tmp.replace(dest)
    return dest


def _raw(cfg: Cfg, name: str, url_key: str) -> Path:
    return download(URLS[url_key], data_dir(cfg, "raw") / name)


# --------------------------------------------------------------------------- HaluEval
def load_halueval_qa(cfg: Cfg) -> List[dict]:
    """Return [{id, question, knowledge, right_answer, hallucinated_answer, source}]."""
    if cfg.project.get("mock"):
        return read_jsonl(TOY / "halueval_qa.jsonl")
    path = _raw(cfg, "halueval_qa.json", "halueval_qa")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    need = {"knowledge", "question", "right_answer", "hallucinated_answer"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"Unexpected HaluEval schema: {list(rows[0]) if rows else 'empty file'}")
    return [{"id": f"halu_{i}", "question": r["question"].strip(), "knowledge": r["knowledge"].strip(),
             "right_answer": r["right_answer"].strip(), "hallucinated_answer": r["hallucinated_answer"].strip(),
             "source": "halueval"} for i, r in enumerate(rows)]


# --------------------------------------------------------------------------- TruthfulQA
def _split_answers(s: str) -> List[str]:
    return [a.strip() for a in (s or "").split(";") if a.strip()]


def load_truthfulqa(cfg: Cfg) -> List[dict]:
    """Return [{id, question, category, type, best_answer, correct_answers, incorrect_answers, source_urls, wiki_titles}]."""
    if cfg.project.get("mock"):
        return read_jsonl(TOY / "truthfulqa.jsonl")
    items: List[dict] = []
    try:
        path = _raw(cfg, "TruthfulQA.csv", "truthfulqa")
        with open(path, encoding="utf-8", newline="") as f:
            for i, r in enumerate(csv.DictReader(f)):
                best = r["Best Answer"].strip()
                correct = _split_answers(r["Correct Answers"])
                if best and best not in correct:
                    correct.insert(0, best)
                urls = extract_urls(r.get("Source", ""))
                items.append({"id": f"tqa_{i}", "question": r["Question"].strip(), "category": r["Category"], "type": r["Type"],
                              "best_answer": best, "best_incorrect": r["Best Incorrect Answer"].strip(),
                              "correct_answers": correct, "incorrect_answers": _split_answers(r["Incorrect Answers"]),
                              "source_urls": urls, "wiki_titles": [t for t in map(wikipedia_title_from_url, urls) if t],
                              "source": "truthfulqa"})
    except Exception as e:  # fall back to the HuggingFace copy
        log.warning("GitHub download of TruthfulQA failed (%s); trying HuggingFace datasets", e)
        from datasets import load_dataset

        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        for i, r in enumerate(ds):
            urls = extract_urls(r.get("source", ""))
            correct = list(r["correct_answers"])
            if r["best_answer"] and r["best_answer"] not in correct:
                correct.insert(0, r["best_answer"])
            items.append({"id": f"tqa_{i}", "question": r["question"], "category": r["category"], "type": r["type"],
                          "best_answer": r["best_answer"], "best_incorrect": "", "correct_answers": correct,
                          "incorrect_answers": list(r["incorrect_answers"]), "source_urls": urls,
                          "wiki_titles": [t for t in map(wikipedia_title_from_url, urls) if t], "source": "truthfulqa"})
    return items


# --------------------------------------------------------------------------- RAGTruth
def _parse_passages(passages: str) -> List[str]:
    parts = re.split(r"passage\s+\d+\s*:", passages or "", flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def load_ragtruth(cfg: Cfg, split: str = "test", tasks: Sequence[str] = ("QA",)) -> List[dict]:
    """Return response-level records with context passages and hallucination labels (quality == 'good' only)."""
    if cfg.project.get("mock"):
        recs = read_jsonl(TOY / "ragtruth_qa.jsonl")
        return [r for r in recs if r["split"] == split and r["task"] in tasks]
    resp = [json.loads(l) for l in open(_raw(cfg, "ragtruth_response.jsonl", "ragtruth_response"), encoding="utf-8")]
    src = {s["source_id"]: s for s in (json.loads(l) for l in open(_raw(cfg, "ragtruth_source_info.jsonl", "ragtruth_source"), encoding="utf-8"))}
    out: List[dict] = []
    for r in resp:
        s = src.get(r["source_id"])
        if s is None or r["split"] != split or s["task_type"] not in tasks or r.get("quality", "good") != "good":
            continue
        si = s["source_info"]
        if s["task_type"] == "QA":
            question = si.get("question", "")
            ctx = _parse_passages(si.get("passages", ""))
        elif isinstance(si, str):
            question, ctx = "", [si]
        else:  # Data2txt: structured JSON
            question, ctx = "", [json.dumps(si, ensure_ascii=False)]
        out.append({"id": f"rt_{r['id']}", "source_id": r["source_id"], "task": s["task_type"], "model": r["model"],
                    "split": r["split"], "question": question, "context": ctx, "prompt": s["prompt"], "response": r["response"].strip(),
                    "hallucinated": bool(r["labels"]), "n_spans": len(r["labels"]), "label_types": sorted({l["label_type"] for l in r["labels"]}),
                    "source": "ragtruth"})
    return out


# --------------------------------------------------------------------------- FEVER
_FEVER_LABELS = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES", "NOT ENOUGH INFO": "NOT ENOUGH INFO", "NEI": "NOT ENOUGH INFO",
                 "SUPPORTED": "SUPPORTS", "REFUTED": "REFUTES", "NOTENOUGHINFO": "NOT ENOUGH INFO"}


def _clean_wiki_title(t: str) -> str:
    return (t or "").replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]").strip()


def _parse_fever_evidence(evidence: Any) -> List[dict]:
    """Defensively extract [{title, text}] from the various FEVER evidence encodings."""
    out: List[dict] = []
    for ev in evidence or []:
        title = text = ""
        if isinstance(ev, dict):
            text = ev.get("text") or ev.get("sentence") or ev.get("evidence_text") or ""
            title = ev.get("title") or ev.get("page") or ev.get("wiki_url") or ""
        elif isinstance(ev, (list, tuple)):
            strs = [x for x in ev if isinstance(x, str) and x.strip()]
            if strs:
                text = strs[-1]
                if len(strs) >= 3:
                    title = strs[-3] if len(strs) >= 4 else strs[0]
        elif isinstance(ev, str):
            text = ev
        text = text.strip()
        if text:
            out.append({"title": _clean_wiki_title(title), "text": text})
    # de-duplicate
    seen, uniq = set(), []
    for e in out:
        k = (e["title"], e["text"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq


def load_fever(cfg: Cfg, split: str = "validation") -> List[dict]:
    """Return [{id, claim, label, evidence:[{title,text}]}] - FEVER claims with gold evidence sentences."""
    if cfg.project.get("mock"):
        return read_jsonl(TOY / "fever.jsonl")
    from datasets import load_dataset

    ds = load_dataset("copenlu/fever_gold_evidence", split=split)
    names = getattr(ds.features.get("label"), "names", None)
    items = []
    for i, r in enumerate(ds):
        lab = r["label"]
        if isinstance(lab, int) and names:
            lab = names[lab]
        lab = _FEVER_LABELS.get(str(lab).upper().strip())
        if lab is None:
            continue
        items.append({"id": f"fever_{r.get('id', i)}", "claim": r["claim"].strip(), "label": lab,
                      "evidence": _parse_fever_evidence(r.get("evidence")), "source": "fever"})
    with_ev = [x for x in items if x["label"] != "NOT ENOUGH INFO"]
    if with_ev and sum(bool(x["evidence"]) for x in with_ev) < 0.5 * len(with_ev):
        raise ValueError("Could not parse FEVER evidence sentences - run `python scripts/prepare_data.py --check` and inspect the schema "
                         "(see docs/DATA.md). Stage 2 can run without FEVER: set stage2.fever.n=0.")
    return items


# --------------------------------------------------------------------------- sampling / splitting
def sample_fever(items: Sequence[dict], n: int, seed: int) -> List[dict]:
    """Balanced over the three FEVER labels (the NEI claims have no gold evidence, so retrieval must find some)."""
    rng = random.Random(seed)
    by: Dict[str, List[dict]] = {}
    for x in items:
        if x["label"] == "NOT ENOUGH INFO" or x["evidence"]:
            by.setdefault(x["label"], []).append(x)
    per = max(1, n // 3)
    out: List[dict] = []
    for lab in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
        pool = by.get(lab, [])
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out


def sample_ragtruth(items: Sequence[dict], n: int, pos_frac: float, seed: int) -> List[dict]:
    rng = random.Random(seed)
    pos = [x for x in items if x["hallucinated"]]
    neg = [x for x in items if not x["hallucinated"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = min(len(pos), int(round(n * pos_frac)))
    out = pos[:n_pos] + neg[: n - n_pos]
    rng.shuffle(out)
    return out


def make_splits(cfg: Cfg, halueval: Sequence[dict], tqa: Sequence[dict]) -> Dict[str, List[dict]]:
    """Disjoint HaluEval dev / test / e2e pools (fixed offsets => changing ``n`` never makes them overlap).

    * ``halueval_dev``  - calibrate detector thresholds, fit the hybrid detector (Stage 2)
    * ``halueval_test`` - report detection metrics (Stage 2)
    * ``halueval_e2e``  - end-to-end generation baselines & ablation (Stages 1 and 3)
    * ``tqa_e2e``       - TruthfulQA questions for the same
    """
    seed = cfg.project.seed
    rng = random.Random(seed)
    hv = list(halueval)
    rng.shuffle(hv)
    L = len(hv)
    a, b, c = (500, 1500, 2500) if L >= 2500 else (int(0.4 * L), int(0.7 * L), L)
    n_e2e = max(cfg.stage1.n_per_dataset, cfg.stage3.n_per_dataset)
    tq = list(tqa)
    random.Random(seed + 1).shuffle(tq)
    return {
        "halueval_dev": hv[:a][: cfg.stage2.halueval.n_dev],
        "halueval_test": hv[a:b][: cfg.stage2.halueval.n_test],
        "halueval_e2e": hv[b:c][:n_e2e],
        "tqa_e2e": tq[:n_e2e],
    }
