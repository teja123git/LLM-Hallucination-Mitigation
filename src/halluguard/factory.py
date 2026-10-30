"""Builds the LLM / retriever / NLI / pipeline objects from a config."""
from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ROOT, Cfg, data_dir, results_dir
from .detection import Calibration
from .llm import BaseLLM, GroqLLM, ResponseCache
from .nli import NLIModel, make_nli
from .pipeline import FactualityPipeline
from .retrieval import Embedder, HybridRetriever, make_embedder
from .utils import get_logger

log = get_logger()


def is_mock(cfg: Cfg) -> bool:
    return bool(cfg.project.get("mock")) or os.environ.get("HALLUGUARD_MOCK") == "1"


def make_llm(cfg: Cfg) -> BaseLLM:
    kw = dict(model=cfg.llm.model, judge_model=cfg.llm.judge_model, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens,
              sample_temperature=cfg.llm.sample_temperature, seed=cfg.project.seed)
    if is_mock(cfg):
        from .mock import MockLLM

        return MockLLM(**kw)
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # pragma: no cover
        pass
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        if sys.stdin and sys.stdin.isatty():
            key = getpass.getpass("Enter your Groq API key (free at https://console.groq.com): ")
        else:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                               "(free at https://console.groq.com), or export GROQ_API_KEY.")
    cache = ResponseCache(data_dir(cfg, "cache") / "llm_cache.sqlite") if cfg.llm.cache else None
    return GroqLLM(api_key=key, rpm=cfg.llm.rpm, tpm=cfg.llm.tpm, max_retries=cfg.llm.max_retries, timeout=cfg.llm.timeout, cache=cache,
                    limits={k: dict(v) for k, v in cfg.llm.get("limits", {}).items()},
                    reasoning_effort=cfg.llm.get("reasoning_effort", "low"), reasoning_headroom=cfg.llm.get("reasoning_headroom", 600), **kw)


def calibration_path(cfg: Cfg) -> Path:
    return results_dir(cfg, "stage2") / "calibration.json"


def load_calibration(cfg: Cfg) -> Calibration:
    path = calibration_path(cfg)
    cal = Calibration.load(path)
    if not cal.calibrated:
        log.warning("No calibration found at %s - using default thresholds. Run Stage 2 first for calibrated detectors.", path)
    else:
        for name, h in cal.heads.items():
            if h.threshold <= 0.0:
                log.warning("Detector %r has threshold %.3f: it flags EVERY claim, so systems using it will abstain. The detector probably has "
                            "no signal on the dev split - inspect results/stage2/detection_metrics.csv.", name, h.threshold)
    return cal


def build_pipeline(cfg: Cfg, llm: BaseLLM, retriever: HybridRetriever, nli: Optional[NLIModel] = None,
                   calibration: Optional[Calibration] = None) -> FactualityPipeline:
    return FactualityPipeline(cfg, llm, retriever, nli or make_nli(cfg), calibration or load_calibration(cfg))
