"""Markdown report assembly (results/REPORT.md) from the CSV/JSON artefacts of the three stages."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Cfg, results_dir
from .utils import read_json


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3f", cols: Optional[Sequence[str]] = None) -> str:
    if df is None or not len(df):
        return "_(no data)_\n"
    d = df[[c for c in cols if c in df]] if cols else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join(["---"] * len(d.columns)) + "|\n"
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def _read(p: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(p) if p.exists() else None


def build_report(cfg: Cfg) -> Path:
    root = results_dir(cfg)
    mock = bool(cfg.project.get("mock"))
    lines = ["# Results report (auto-generated)\n"]
    if mock:
        lines.append("> **MOCK RUN - fake LLM, fake NLI, toy data. None of these numbers are real results.**\n")
    for stage, title in (("stage1", "Stage 1 - baselines & retrieval"), ("stage2", "Stage 2 - hallucination detection"), ("stage3", "Stage 3 - ablation")):
        meta = read_json(root / stage / "meta.json")
        lines.append(f"\n## {title}\n")
        if meta:
            lines.append(f"_run {meta.get('timestamp')} | model `{meta.get('model')}` | judge `{meta.get('judge_model')}` | embedder `{meta.get('embed_model')}` | NLI `{meta.get('nli_model')}`_\n")
        else:
            lines.append("_not run yet_\n")
            continue
        if stage == "stage1":
            lines += ["### Retrieval quality (gold passage in top-k)\n", df_to_md(_read(root / stage / "retrieval_metrics.csv")),
                      "### Baselines A (LLM only) / B (hybrid RAG)\n",
                      df_to_md(_read(root / stage / "summary_by_dataset.csv"), cols=["dataset", "system", "n", "abstention_rate", "bertscore_f1", "rougeL_f1", "contains", "tqa_truthful_sim", "cost_calls", "cost_prompt_tokens", "cost_completion_tokens", "cost_latency"]),
                      "![quality](stage1/quality.png)\n\n![retrieval](stage1/retrieval.png)\n"]
        elif stage == "stage2":
            lines += ["### Detection metrics (thresholds calibrated on HaluEval dev with Youden's J)\n",
                      df_to_md(_read(root / stage / "detection_metrics.csv"), cols=["dataset", "detector", "n", "pos_rate", "threshold", "accuracy", "precision", "recall", "f1", "auroc", "auprc"]),
                      "![comparison](stage2/detector_comparison.png)\n\n![roc](stage2/roc_pr_halueval.png)\n\n![selfcheck](stage2/selfcheck_n_curve.png)\n"]
        else:
            lines += ["### Ablation (all questions)\n",
                      df_to_md(_read(root / stage / "summary_overall.csv"), cols=["system", "n", "correct", "incorrect", "not_attempted", "correct_given_attempted", "f_score", "claim_support_rate", "claim_contradiction_rate", "factscore", "free_correct", "free_incorrect", "cost_calls", "cost_total_tokens", "cost_latency"]),
                      "### Paired contrasts\n", df_to_md(_read(root / stage / "contrasts.csv")),
                      "![grades](stage3/grades.png)\n\n![marginal](stage3/marginal_contributions.png)\n\n![pareto](stage3/pareto.png)\n\n![cost](stage3/cost.png)\n"]
    out = root / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
