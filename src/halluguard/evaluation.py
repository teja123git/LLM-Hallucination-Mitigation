"""Answer-level evaluation shared by Stage 1 (baselines) and Stage 3 (ablation)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics as M
from .config import Cfg
from .retrieval import Embedder
from .utils import get_logger, is_abstention

log = get_logger()


def kind_of(item: dict) -> str:
    return "halueval" if item.get("source") == "halueval" else "truthfulqa"


def reference_answer(item: dict) -> str:
    return item["right_answer"] if kind_of(item) == "halueval" else item["best_answer"]


def gold_pids_for(item: dict, doc_pids: Dict[str, List[int]]) -> set:
    if kind_of(item) == "halueval":
        return set(doc_pids.get(item["id"], []))
    out: set = set()
    for t in item.get("wiki_titles", []):
        out |= set(doc_pids.get(f"tqa:{t}", []))
    return out


def tqa_similarity_truthful(embedder: Embedder, answer: str, item: dict) -> Dict[str, float]:
    """Cheap, judge-free TruthfulQA score: the answer is 'truthful' if it is closer (embedding cosine) to the set of
    reference-correct answers than to the set of reference-incorrect answers (cf. Lin et al. 2022, BLEU/ROUGE variant)."""
    refs = item["correct_answers"] + item["incorrect_answers"]
    E = embedder.encode([answer] + refs)
    sims = E[1:] @ E[0]
    nc = len(item["correct_answers"])
    c, i = float(np.max(sims[:nc])), float(np.max(sims[nc:])) if len(sims) > nc else -1.0
    return {"sim_correct": c, "sim_incorrect": i, "margin": c - i, "truthful": float(c > i)}


def _bertscore_unique(preds: List[str], refs: List[str], cfg: Cfg) -> np.ndarray:
    pairs = sorted(set(zip(preds, refs)))
    scores = M.bertscore_f1([p for p, _ in pairs], [r for _, r in pairs], cfg.eval.bertscore_model, cfg.eval.bertscore_rescale)
    lut = dict(zip(pairs, scores))
    return np.array([lut[(p, r)] for p, r in zip(preds, refs)])


def free_metrics(outputs: Sequence, items: Dict[str, dict], embedder: Embedder, cfg: Cfg, doc_pids: Optional[Dict[str, List[int]]] = None,
                 use_bertscore: bool = True) -> pd.DataFrame:
    """One row per (system, question): overlap metrics, QA accuracy, similarity-truthfulness, abstention, cost, citation metrics."""
    rows = []
    for o in outputs:
        it = items[o.qid]
        kind = kind_of(it)
        ref = reference_answer(it)
        r = {"system": o.system, "qid": o.qid, "dataset": kind, "abstained": bool(o.abstained), "answer": o.final_answer,
             "n_final_claims": len(o.final_claims), "n_flagged": o.n_flagged, "n_removed": o.n_removed, "n_rewritten": o.n_rewritten,
             "answer_words": len(o.final_answer.split()), "reference": ref}
        rl = M.rouge_l(o.final_answer, ref)
        r.update({"rougeL_f1": rl["f1"], "rougeL_precision": rl["precision"], "rougeL_recall": rl["recall"]})
        if kind == "halueval":
            r.update({"em": M.exact_match(o.final_answer, ref), "token_f1": M.token_f1(o.final_answer, ref), "contains": M.contains_match(o.final_answer, ref)})
        else:
            if not o.abstained:
                s = tqa_similarity_truthful(embedder, o.final_answer, it)
                r.update({"tqa_truthful_sim": s["truthful"], "tqa_margin": s["margin"]})
            r["tqa_has_wiki"] = bool(it.get("wiki_titles"))
        if doc_pids is not None and o.sources:
            gold = gold_pids_for(it, doc_pids)
            cited = [s["pid"] for s in o.sources if s["pid"] >= 0]
            if gold and cited:
                r["cite_precision"] = sum(p in gold for p in cited) / len(cited)
                r["cite_hit"] = float(any(p in gold for p in cited))
        for k, v in o.cost.items():
            r[f"cost_{k}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        # judge-free correctness: HaluEval -> gold answer contained; TruthfulQA -> closer to correct than to incorrect answers
        fc = np.where(df["dataset"] == "halueval", df.get("contains", np.nan) == 1, df.get("tqa_truthful_sim", np.nan) == 1)
        df["free_correct"] = fc & ~df["abstained"]
        df["free_incorrect"] = ~df["abstained"] & ~df["free_correct"]
    if use_bertscore and len(df):
        try:
            df["bertscore_f1"] = _bertscore_unique(df["answer"].tolist(), df["reference"].tolist(), cfg)
        except Exception as e:  # bert-score / model download problems must not kill the run
            log.warning("BERTScore unavailable (%s) - continuing without it", e)
    return df


def add_judgments(df: pd.DataFrame, judgments: Dict[tuple, dict]) -> pd.DataFrame:
    """judgments[(system, qid)] -> judge_output dict."""
    df = df.copy()
    df["grade"] = [judgments.get((s, q), {}).get("grade") for s, q in zip(df["system"], df["qid"])]
    def cv(s, q, what):
        v = judgments.get((s, q), {}).get("claim_verdicts") or []
        return sum(x == what for x in v), len([x for x in v if x])
    for what, col in (("SUPPORTED", "claims_supported"), ("CONTRADICTED", "claims_contradicted"), ("UNVERIFIABLE", "claims_unverifiable")):
        df[col] = [cv(s, q, what)[0] for s, q in zip(df["system"], df["qid"])]
    df["claims_judged"] = [cv(s, q, "SUPPORTED")[1] for s, q in zip(df["system"], df["qid"])]
    return df


def summarize(df: pd.DataFrame, group: Sequence[str] = ("system",), n_boot: int = 1000) -> pd.DataFrame:
    """Aggregate per-question rows into a metrics table (means with bootstrap 95% CIs for the headline rates)."""
    out = []
    for key, g in df.groupby(list(group), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group, key))
        row["n"] = len(g)
        row["abstention_rate"] = g["abstained"].mean()
        att = g[~g["abstained"]]
        for col in ("rougeL_f1", "bertscore_f1", "em", "token_f1", "contains", "tqa_truthful_sim", "cite_precision", "cite_hit"):
            if col in g and g[col].notna().any():
                row[col] = g[col].mean()
                row[f"{col}_attempted"] = att[col].mean() if len(att) and att[col].notna().any() else np.nan
        for col in ("free_correct", "free_incorrect"):
            if col in g:
                row[col] = g[col].mean()
        for col in ("cost_calls", "cost_prompt_tokens", "cost_completion_tokens", "cost_total_tokens", "cost_latency", "n_final_claims", "answer_words"):
            if col in g:
                row[col] = g[col].mean()
        if "grade" in g and g["grade"].notna().any():
            sq = M.simpleqa_metrics(g["grade"].dropna().tolist())
            row.update({k: v for k, v in sq.items() if k != "n"})
            correct = (g["grade"].dropna() == "CORRECT").astype(float).values
            _, lo, hi = M.bootstrap_ci(correct, n_boot)
            row["correct_ci_lo"], row["correct_ci_hi"] = lo, hi
            inc = (g["grade"].dropna() == "INCORRECT").astype(float).values
            _, lo, hi = M.bootstrap_ci(inc, n_boot)
            row["incorrect_ci_lo"], row["incorrect_ci_hi"] = lo, hi
            tot = g["claims_judged"].sum()
            if tot:
                row["claim_support_rate"] = g["claims_supported"].sum() / tot          # FActScore-style atomic precision
                row["claim_contradiction_rate"] = g["claims_contradicted"].sum() / tot
                # FActScore proper averages the per-answer precision; both are reported
                per = (g["claims_supported"] / g["claims_judged"].where(g["claims_judged"] > 0)).dropna()
                row["factscore"] = per.mean() if len(per) else np.nan
        out.append(row)
    return pd.DataFrame(out)
