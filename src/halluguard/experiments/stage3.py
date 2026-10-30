"""Stage 3 - evidence-guided correction and the controlled ablation study (systems A-G, F-CoVe).

All systems answer the *same* questions (paired design).  Quality is measured three ways, from cheap to expensive:
  1. judge-free:  HaluEval -> gold answer contained in the answer;  TruthfulQA -> closer to correct than to incorrect references
  2. overlap   :  BERTScore / ROUGE-L against the reference answer (kept from Stage 1 for continuity)
  3. judge     :  an independent, stronger LLM grades each answer CORRECT / INCORRECT / NOT_ATTEMPTED against *gold*
                  references and labels every atomic claim SUPPORTED / CONTRADICTED / UNVERIFIABLE (FActScore-style precision)
Cost (calls, prompt / completion tokens, latency) comes from the stages each system actually depends on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import metrics as M
from .. import plotting as P
from ..config import Cfg, results_dir
from ..evaluation import add_judgments, free_metrics, kind_of, reference_answer, summarize
from ..factory import load_calibration
from ..judge import judge_output
from ..pipeline import SYSTEMS, FactualityPipeline, QuestionState, SystemOutput
from ..report import build_report
from ..utils import get_logger, read_jsonl, write_json, write_jsonl
from .common import Resources, banner, load_resources, map_questions, run_meta

log = get_logger()

CONTRASTS = [  # (label, baseline system, system) - each isolates ONE component
    ("Retrieval (B vs A)", "A", "B"),
    ("Self-consistency filter (C vs A)", "A", "C"),
    ("CoVe verification (D vs A)", "A", "D"),
    ("Evidence verification (E vs B)", "B", "E"),
    ("Correction, RARR (F vs E)", "E", "F"),
    ("Hybrid detector (G vs F)", "F", "G"),
    ("Correction, CoVe (F-CoVe vs E)", "E", "F_cove"),
    ("Full pipeline vs RAG (F vs B)", "B", "F"),
    ("Full pipeline vs plain LLM (F vs A)", "A", "F"),
]


# =========================================================================== judging
def judge_all(res: Resources, outputs: Sequence[SystemOutput], items: Dict[str, dict], workers: int):
    work = list(outputs)
    results, quota_err, failed = map_questions(lambda o: judge_output(res.llm, items[o.qid], kind_of(items[o.qid]), o), work, workers, "LLM judge")
    judgments = {(o.system, o.qid): r for o, r in zip(work, results) if r is not None}
    return judgments, quota_err, failed


# =========================================================================== statistics
def paired_contrasts(df: pd.DataFrame, use_judge: bool, n_boot: int) -> pd.DataFrame:
    """Paired (same-question) differences with bootstrap CIs and McNemar tests, overall and per dataset."""
    ok, bad = ("correct", "incorrect") if use_judge else ("free_correct", "free_incorrect")
    d = df.copy()
    if use_judge:
        d = d[d["grade"].notna()]
        d["correct"] = (d["grade"] == "CORRECT").astype(float)
        d["incorrect"] = (d["grade"] == "INCORRECT").astype(float)
    else:
        d["free_correct"] = d["free_correct"].astype(float)
        d["free_incorrect"] = d["free_incorrect"].astype(float)
    rows = []
    for scope in ["all"] + sorted(d["dataset"].unique()):
        sub = d if scope == "all" else d[d["dataset"] == scope]
        pc = sub.pivot(index="qid", columns="system", values=ok)
        pi = sub.pivot(index="qid", columns="system", values=bad)
        for label, a, b in CONTRASTS:
            if a not in pc or b not in pc:
                continue
            m = pc[[a, b]].dropna().index
            if len(m) < 3:
                continue
            dc, clo, chi = M.paired_bootstrap_diff(pc.loc[m, a].values, pc.loc[m, b].values, n_boot)
            di, ilo, ihi = M.paired_bootstrap_diff(pi.loc[m, a].values, pi.loc[m, b].values, n_boot)
            mc = M.mcnemar_exact(pc.loc[m, a].values > 0.5, pc.loc[m, b].values > 0.5)
            rows.append({"scope": scope, "contrast": label, "n": len(m), "delta_correct": dc, "correct_lo": clo, "correct_hi": chi,
                         "delta_incorrect": di, "incorrect_lo": ilo, "incorrect_hi": ihi, "mcnemar_p": mc["p_value"],
                         "only_baseline_correct": mc["only_a"], "only_system_correct": mc["only_b"], "outcome": "judge" if use_judge else "judge-free"})
    return pd.DataFrame(rows)


def group_costs(outputs: Sequence[SystemOutput]) -> pd.DataFrame:
    acc: Dict[tuple, List[dict]] = {}
    for o in outputs:
        for g, c in o.cost_by_group.items():
            acc.setdefault((o.system, g), []).append(c)
    rows = []
    for (s, g), cs in acc.items():
        n_sys = sum(1 for o in outputs if o.system == s)
        rows.append({"system": s, "group": g, "calls": sum(c["calls"] for c in cs) / n_sys, "total_tokens": sum(c["total_tokens"] for c in cs) / n_sys,
                     "prompt_tokens": sum(c["prompt_tokens"] for c in cs) / n_sys, "completion_tokens": sum(c["completion_tokens"] for c in cs) / n_sys,
                     "latency": sum(c["latency"] for c in cs) / n_sys})
    return pd.DataFrame(rows)


# =========================================================================== qualitative examples
def write_examples(outputs: Sequence[SystemOutput], items: Dict[str, dict], df: pd.DataFrame, path, k_good: int = 10, k_bad: int = 4) -> None:
    by = {(o.system, o.qid): o for o in outputs}
    qids = list(dict.fromkeys(o.qid for o in outputs))
    gcol = "grade" if "grade" in df and df["grade"].notna().any() else None

    def grade(s, q):
        r = df[(df["system"] == s) & (df["qid"] == q)]
        if not len(r):
            return "?"
        return r.iloc[0]["grade"] if gcol else ("correct" if r.iloc[0]["free_correct"] else "incorrect" if r.iloc[0]["free_incorrect"] else "abstained")

    def good(q):
        f = by.get(("F", q))
        return f is not None and (f.n_rewritten > 0 or f.n_removed > 0)

    def worse(q):
        a, f = grade("A", q), grade("F", q)
        return ("F" in {s for s, _ in by}) and a in ("CORRECT", "correct") and f in ("INCORRECT", "incorrect", "NOT_ATTEMPTED", "abstained")

    lines = ["# Qualitative examples: initial vs. verified / corrected answers\n",
             "Shown: questions where the full pipeline (F) edited the answer, then cases where it made things worse (honest failure analysis).\n"]
    def block(q):
        it = items[q]
        out = [f"\n---\n### {q} ({kind_of(it)})\n**Q:** {it['question']}\n\n**Reference:** {reference_answer(it)}\n"]
        for s in ("A", "B", "F"):
            o = by.get((s, q))
            if o is None:
                continue
            out.append(f"- **{s}** ({grade(s, q)}): {o.final_answer_cited}")
        f = by.get(("F", q))
        if f:
            out.append("\n| claim (from the verified answer) | label | flagged | action | revised |\n|---|---|---|---|---|")
            for c in f.claims:
                out.append(f"| {c['text']} | {c['label']} | {c['flagged']} | {c['action']} | {c['revised'] or ''} |")
            if f.sources:
                out.append("\nSources: " + " ".join(f"[{s['n']}] {(s.get('title') + ': ' if s.get('title') else '') + s['text'][:160]}..." for s in f.sources))
        return "\n".join(out)

    for q in [q for q in qids if good(q)][:k_good]:
        lines.append(block(q))
    lines.append("\n\n## Cases where the full pipeline did worse than the plain LLM\n")
    for q in [q for q in qids if worse(q)][:k_bad]:
        lines.append(block(q))
    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================================== driver
def run(cfg: Cfg, res: Optional[Resources] = None, systems: Optional[Sequence[str]] = None, n: Optional[int] = None, use_judge: Optional[bool] = None) -> dict:
    banner(cfg, "Stage 3")
    res = res or load_resources(cfg)
    out_dir = results_dir(cfg, "stage3")
    n = n or cfg.stage3.n_per_dataset
    systems = list(systems or cfg.stage3.systems)
    use_judge = cfg.stage3.use_llm_judge if use_judge is None else use_judge
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise ValueError(f"Unknown systems {unknown}; choose from {list(SYSTEMS)}")
    pipe = FactualityPipeline(cfg, res.llm, res.retriever, res.nli, load_calibration(cfg))
    items = res.splits["halueval_e2e"][:n] + res.splits["tqa_e2e"][:n]
    by_id = {it["id"]: it for it in items}
    log.info("Stage 3: %d questions x %d systems %s", len(items), len(systems), systems)

    def work(it: dict) -> Dict[str, SystemOutput]:
        return pipe.run_all(QuestionState(it["id"], it["question"]), systems)

    results, quota_err, failed = map_questions(work, items, cfg.llm.workers, "Stage 3 ablation")
    outputs = [o for r in results if r for o in r.values()]
    write_jsonl(out_dir / "outputs.jsonl", [o.to_dict() for o in outputs])
    if quota_err:
        write_json(out_dir / "meta.json", run_meta(cfg, interrupted=True, n_failed=failed))
        raise quota_err
    # keep only questions answered by every system (paired comparison)
    complete = {q for q in {o.qid for o in outputs} if {o.system for o in outputs if o.qid == q} == set(systems)}
    outputs = [o for o in outputs if o.qid in complete]

    df = free_metrics(outputs, by_id, res.embedder, cfg, res.doc_pids)
    judge_failed, judge_err = 0, None
    judged = False
    if use_judge and len(df):
        judgments, judge_err, judge_failed = judge_all(res, outputs, by_id, cfg.llm.workers)
        df = add_judgments(df, judgments)
        judged = df["grade"].notna().any()
        write_json(out_dir / "judgments.json", {f"{s}|{q}": v for (s, q), v in judgments.items()})
    df.to_csv(out_dir / "metrics_per_question.csv", index=False)

    summ = summarize(df, ("dataset", "system"), cfg.stage3.bootstrap)
    summ_all = summarize(df, ("system",), cfg.stage3.bootstrap)
    summ.to_csv(out_dir / "summary_by_dataset.csv", index=False)
    summ_all.to_csv(out_dir / "summary_overall.csv", index=False)
    if "tqa_has_wiki" in df:
        summarize(df[df["dataset"] == "truthfulqa"], ("tqa_has_wiki", "system"), cfg.stage3.bootstrap).to_csv(out_dir / "summary_truthfulqa_by_wiki_coverage.csv", index=False)
    contrasts = paired_contrasts(df, judged, cfg.stage3.bootstrap)
    contrasts.to_csv(out_dir / "contrasts.csv", index=False)
    gc = group_costs(outputs)
    gc.to_csv(out_dir / "cost_by_stage_group.csv", index=False)

    # ---- figures
    if judged:
        P.fig_grades(summ, out_dir / "grades.png")
        P.fig_metric_ci(summ, "correct", "correct_ci_lo", "correct_ci_hi", "CORRECT rate (95% CI)", out_dir / "correct_ci.png")
        P.fig_metric_ci(summ, "incorrect", "incorrect_ci_lo", "incorrect_ci_hi", "INCORRECT rate = hallucination (95% CI)", out_dir / "incorrect_ci.png")
        if "claim_support_rate" in summ:
            P.fig_claim_rates(summ, out_dir / "claim_rates.png")
    P.fig_pareto(summ_all, "f_score" if judged else "free_correct", "SimpleQA F-score" if judged else "judge-free correct rate", out_dir / "pareto.png")
    P.fig_cost(summ_all, gc, out_dir / "cost.png")
    all_c = contrasts[contrasts["scope"] == "all"]
    if len(all_c):
        P.fig_marginal(all_c, out_dir / "marginal_contributions.png")
    P.fig_flag_stats(df, out_dir / "flag_stats.png")
    P.fig_heatmap(summ, ["correct", "incorrect", "f_score", "claim_support_rate"] if judged else ["free_correct", "free_incorrect", "abstention_rate"], out_dir / "heatmap.png")
    cal = pipe.cal
    claim_scores = {name: [c["scores"]["head"] for o in outputs if SYSTEMS[o.system]["detector"] == name for c in o.claims]
                    for name in ("selfcheck", "cove", "evidence", "hybrid")}
    claim_scores = {k: v for k, v in claim_scores.items() if v}
    if claim_scores:
        P.fig_threshold_retention(claim_scores, {k: cal.heads[k].threshold for k in claim_scores}, out_dir / "threshold_retention.png")
    write_examples(outputs, by_id, df, out_dir / "examples.md")

    fb = {s: float(np.mean([o.initial_source.endswith("(fallback)") for o in outputs if o.system == s])) for s in systems if SYSTEMS[s]["init"] == "R"}
    write_json(out_dir / "meta.json", run_meta(cfg, n_per_dataset=n, systems=systems, n_questions=len(complete), n_failed=failed, judge_failed=judge_failed,
                                               judged=bool(judged), judge_equals_generator=(cfg.llm.judge_model == cfg.llm.model),
                                               calibrated=cal.calibrated, rag_fallback_rate=fb, llm_stats=res.llm.stats_table()))
    build_report(cfg)
    log.info("Stage 3 done -> %s", out_dir)
    if judge_err:
        raise judge_err
    return {"df": df, "summary": summ, "summary_overall": summ_all, "contrasts": contrasts, "group_costs": gc, "outputs": outputs, "dir": out_dir}
