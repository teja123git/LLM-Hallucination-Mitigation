"""Small shared helpers: JSON parsing, text utilities, IO, parallelism."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")


def get_logger(name: str = "halluguard") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha1(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


# ------------------------------------------------------------------ IO
def read_jsonl(path: str | Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


# ------------------------------------------------------------------ LLM JSON parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def parse_json(text: str) -> Optional[Any]:
    """Robustly parse the first JSON object/array in an LLM response (handles code fences / chatter)."""
    if not text:
        return None
    candidates = [text.strip()]
    m = _FENCE.search(text)
    if m:
        candidates.insert(0, m.group(1).strip())
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = cand.find(opener), cand.rfind(closer)
            if i != -1 and j > i:
                try:
                    return json.loads(cand[i : j + 1])
                except Exception:
                    continue
    return None


# ------------------------------------------------------------------ text
STOPWORDS = frozenset(
    """a an the and or but if then else of in on at to for from by with without about as into over under
    is are was were be been being am do does did done have has had having it its this that these those
    i you he she we they them his her their our your my me him us who whom whose which what when where why how
    not no nor so than too very can could should would may might will shall just also there here s t""".split()
)

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def truncate_words(text: str, n: int) -> str:
    words = (text or "").split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


_ABSTAIN_PATTERNS = [
    r"does not contain enough information",
    r"doesn'?t contain enough information",
    r"not enough (reliable )?(information|evidence)",
    r"insufficient (information|evidence)",
    r"could not find enough",
    r"i (do not|don'?t) know",
    r"i'?m not sure",
    r"i (cannot|can'?t|am unable to|'m unable to) (answer|determine|find|provide|verify)",
    r"no (relevant )?information (is )?(available|provided|found)",
    r"the (provided )?context does not",
    r"unable to (answer|determine|find|verify)",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.I)


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text or "")) or not (text or "").strip()


def parallel_map(fn: Callable[[T], R], items: Sequence[T], workers: int = 1, desc: str = "", progress: bool = True) -> List[R]:
    """Order-preserving map with optional threads and a tqdm bar."""
    from tqdm.auto import tqdm

    if workers <= 1:
        return [fn(x) for x in tqdm(items, desc=desc, disable=not progress)]
    out: List[Optional[R]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        from concurrent.futures import as_completed

        for fut in tqdm(as_completed(futs), total=len(items), desc=desc, disable=not progress):
            out[futs[fut]] = fut.result()
    return out  # type: ignore[return-value]


def batched(seq: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
