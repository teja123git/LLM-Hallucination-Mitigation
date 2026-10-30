"""Stage 1 - datasets, knowledge base, retrieval quality, Baseline A (LLM only) and Baseline B (hybrid RAG)."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .. import metrics as M
from .. import plotting as P
from ..config import Cfg, results_dir
from ..evaluation import free_metrics, gold_pids_for, kind_of, reference_answer, summarize
from ..pipeline import FactualityPipeline, QuestionState
from ..utils import get_logger, write_json, write_jsonl
from .common import Resources, banner, load_resources, map_questions, run_meta

log = get_logger()
KS = (1, 3, 5, 10)


def retrieval_eval(res: Resources, items: List[dict]) -> pd.DataFrame:
    """Recall@k / MRR of bm25, dense and hybrid retrieval for questions whose gold passage(s) are known."""
    rows = []
    for kind in ("halueval", "truthfulqa"):
        qs = [it for it in items if kind_of(it) == kind and gold_pids_for(it, res.doc_pids)]
        if not qs:
            continue
        for mode in ("bm25", "dense", "hybrid"):
            ranked = [[h.pid for h in res.retriever.search(it["question"], k=max(KS), mode=mode)] for it in qs]
            gold = [gold_pids_for(it, res.doc_pids) for it in qs]
            rows.append({"dataset": kind, "mode": mode, **M.retrieval_metrics(ranked, gold, KS)})
    return pd.DataFrame(rows)


def run(cfg: Cfg, res: Optional[Resources] = None) -> dict:
    banner(cfg, "Stage 1")
    res = res or load_resources(cfg)
    out_dir = results_dir(cfg, "stage1")
    n = cfg.stage1.n_per_dataset
    items = res.splits["halueval_e2e"][:n] + res.splits["tqa_e2e"][:n]
    by_id = {it["id"]: it for it in items}
    log.info("Stage 1: %d HaluEval + %d TruthfulQA questions", len(res.splits["halueval_e2e"][:n]), len(res.splits["tqa_e2e"][:n]))

    # ---- retrieval quality
    ret = retrieval_eval(res, items)
    ret.to_csv(out_dir / "retrieval_metrics.csv", index=False)
    P.fig_retrieval(ret, out_dir / "retrieval.png")
    P.fig_eda(pd.DataFrame(res.splits["tqa_e2e"][:n]), pd.DataFrame(res.splits["halueval_e2e"][:n]), out_dir / "eda.png")

    # ---- Baselines A and B (the same code path as the Stage 3 ablation => identical prompts => shared LLM cache)
    pipe = FactualityPipeline(cfg, res.llm, res.retriever, nli=None)  # type: ignore[arg-type]  (no NLI needed for plain generation)

    def work(it: dict):
        st = QuestionState(it["id"], it["question"])
        outs = pipe.run_all(st, ["A", "B"], eval_claims=False)
        hits = st.results.get("retrieve:q", [])
        cov = M.context_coverage([h["text"] for h in hits], reference_answer(it))
        return outs, cov

    results, quota_err, failed = map_questions(work, items, cfg.llm.workers, "Stage 1 baselines A/B")
    outputs, cov_rows = [], []
    for it, r in zip(items, results):
        if r is None:
            continue
        outs, cov = r
        outputs += list(outs.values())
        cov_rows.append({"qid": it["id"], "dataset": kind_of(it), **cov})
    write_jsonl(out_dir / "outputs.jsonl", [o.to_dict() for o in outputs])

    df = free_metrics(outputs, by_id, res.embedder, cfg, res.doc_pids)
    df.to_csv(out_dir / "metrics_per_question.csv", index=False)
    summ = summarize(df, ("dataset", "system"))
    summ_all = summarize(df, ("system",))
    summ.to_csv(out_dir / "summary_by_dataset.csv", index=False)
    summ_all.to_csv(out_dir / "summary_overall.csv", index=False)
    cov = pd.DataFrame(cov_rows)
    cov_summary = cov.groupby("dataset")[["token_recall", "contains"]].mean().reset_index() if len(cov) else cov
    cov_summary.to_csv(out_dir / "context_coverage.csv", index=False)
    if len(df):
        P.fig_stage1_quality(df, out_dir / "quality.png")

    write_json(out_dir / "meta.json", run_meta(cfg, n_per_dataset=n, n_failed=failed, interrupted=bool(quota_err), llm_stats=res.llm.stats_table()))
    log.info("Stage 1 done -> %s", out_dir)
    if quota_err:
        raise quota_err
    return {"df": df, "summary": summ, "summary_overall": summ_all, "retrieval": ret, "coverage": cov_summary, "dir": out_dir}
