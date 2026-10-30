"""Shared experiment plumbing: resource loading, quota-aware question loop, result bookkeeping."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .. import data as D
from .. import plotting
from ..config import Cfg, data_dir, results_dir
from ..factory import is_mock, make_llm
from ..kb import build_qa_kb, fetch_wikipedia_pages
from ..llm import BaseLLM, DailyQuotaExceeded, LLMError
from ..nli import NLIModel, make_nli
from ..retrieval import Embedder, HybridRetriever, make_embedder
from ..utils import get_logger, parallel_map, set_seed

log = get_logger()


@dataclass
class Resources:
    cfg: Cfg
    halueval: List[dict]
    tqa: List[dict]
    splits: Dict[str, List[dict]]
    embedder: Embedder
    retriever: HybridRetriever
    doc_pids: Dict[str, List[int]]
    llm: BaseLLM
    wiki_pages: Dict[str, Dict[str, str]] = field(default_factory=dict)
    _nli: Optional[NLIModel] = None

    @property
    def nli(self) -> NLIModel:
        if self._nli is None:
            self._nli = make_nli(self.cfg)
            if not is_mock(self.cfg):
                self._nli.sanity_check()
        return self._nli


def load_resources(cfg: Cfg) -> Resources:
    """Datasets, leak-free knowledge base (+ Wikipedia pages cited by TruthfulQA) and the LLM client."""
    set_seed(cfg.project.seed)
    plotting.set_mock(is_mock(cfg))
    hv, tq = D.load_halueval_qa(cfg), D.load_truthfulqa(cfg)
    splits = D.make_splits(cfg, hv, tq)
    wiki: Dict[str, Dict[str, str]] = {}
    if cfg.kb.tqa_wikipedia and not is_mock(cfg):
        titles = sorted({t for x in tq for t in x["wiki_titles"]})
        wiki = fetch_wikipedia_pages(titles, data_dir(cfg, "cache", "wikipedia"), cfg.kb.wikipedia_max_words, cfg.kb.wikipedia_user_agent)
        log.info("Wikipedia pages available for TruthfulQA: %d / %d", len(wiki), len(titles))
    embedder = make_embedder(cfg)
    retriever, doc_pids = build_qa_kb(cfg, hv, tq, embedder, wiki)
    return Resources(cfg, hv, tq, splits, embedder, retriever, doc_pids, make_llm(cfg), wiki)


def map_questions(fn: Callable[[Any], Any], items: Sequence[Any], workers: int, desc: str) -> Tuple[List[Any], Optional[DailyQuotaExceeded], int]:
    """Apply ``fn`` to every item.  A daily-quota error stops the loop gracefully (finished items are kept - and cached);
    other LLM failures only drop the affected item.  Returns (results with None for missing items, quota error, #failed)."""
    results: List[Any] = [None] * len(items)
    stop = threading.Event()
    state: Dict[str, Any] = {"err": None, "failed": 0}

    def wrapped(pair):
        i, it = pair
        if stop.is_set():
            return None
        try:
            results[i] = fn(it)
        except DailyQuotaExceeded as e:
            state["err"] = e
            stop.set()
        except LLMError as e:
            state["failed"] += 1
            log.error("LLM failure on item %s - skipped: %s", i, str(e)[:200])
        return None

    parallel_map(wrapped, list(enumerate(items)), workers=workers, desc=desc)
    return results, state["err"], state["failed"]


def run_meta(cfg: Cfg, **extra) -> dict:
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "mock": is_mock(cfg), "model": cfg.llm.model,
            "judge_model": cfg.llm.judge_model, "embed_model": cfg.retrieval.embed_model, "nli_model": cfg.nli.model, "seed": cfg.project.seed, **extra}


def banner(cfg: Cfg, name: str) -> None:
    if is_mock(cfg):
        log.warning("=== %s in MOCK mode: fake LLM / NLI / embedder, toy data. Numbers are NOT real results. ===", name)
