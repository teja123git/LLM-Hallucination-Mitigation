"""Stage 2 - claim extraction and hallucination detection, benchmarked against annotated labels.

Datasets / tasks
  * HaluEval-QA  : answer-level binary detection.  Every question yields TWO labelled answers (right = 0 /
                   hallucinated = 1).  ``dev`` split -> calibrate thresholds and fit the hybrid head; ``test`` split -> report.
  * RAGTruth     : response-level binary detection of *naturally generated* RAG hallucinations (zero-shot transfer: the
                   detectors calibrated on HaluEval are applied unchanged; the provided context is the evidence).
  * FEVER        : 3-way claim verification (SUPPORTS / REFUTES / NOT ENOUGH INFO) + binary "unsupported" detection.

Detectors: nli | llm | selfcheck | cove | evidence (nli+llm) | hybrid (nli+llm+selfcheck)  - see ``detection.py``.
Extra experiments (all cheap because the LLM calls are cached): SelfCheck vs. number of samples, retrieval mode
(bm25 / dense / hybrid) vs. detection quality, gold vs. retrieved evidence.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import data as D
from .. import metrics as M
from .. import plotting as P
from .. import prompts
from ..config import Cfg, results_dir
from ..detection import HEAD_FEATURES, Calibration, DetectorHead, aggregate, fit_head
from ..factory import calibration_path, is_mock
from ..kb import build_pool_kb
from ..nli import nli_against_evidence, selfcheck_per_sample
from ..pipeline import FactualityPipeline, QuestionState, ptxt
from ..retrieval import chunk_text
from ..utils import get_logger, read_jsonl, write_json, write_jsonl
from .common import Resources, banner, load_resources, map_questions, run_meta

log = get_logger()
RAW_FEATURES = ["s_nli", "s_llm", "s_sc", "s_cove"]


# =========================================================================== feature extraction
def answer_features(pipe: FactualityPipeline, st: QuestionState, which: str, names: Sequence[str], how: str) -> dict:
    """Claim-level signals -> one answer-level record (max/mean over claims; an answer without claims has nothing to flag)."""
    rows = pipe.features(st, which, names)
    out: dict = {"claims": [r["claim"] for r in rows],
                 "top_evidence": [ptxt(r["evidence"][0]) if r["evidence"] else "" for r in rows],
                 "claim_scores": [{k: v for k, v in r["feats"].items() if v is not None} for r in rows]}
    for f in RAW_FEATURES:
        out[f] = aggregate([r["feats"][f] for r in rows], how) if f in names else None
    return out


# =========================================================================== HaluEval
def _halueval_rows(pipe: FactualityPipeline, res: Resources, it: dict, split: str, cfg: Cfg) -> List[dict]:
    how = cfg.stage2.aggregate
    use = set(cfg.stage2.detectors)
    names = ["s_nli", "s_llm"] + (["s_sc"] if use & {"selfcheck", "hybrid"} else []) + (["s_cove"] if "cove" in use else [])
    st = QuestionState(it["id"], it["question"])
    st.answers = {"R": it["right_answer"], "H": it["hallucinated_answer"]}
    st_g = QuestionState(it["id"] + "#gold", it["question"], fixed_evidence=[it["knowledge"]])
    st_g.answers = dict(st.answers)
    rows = []
    for which, label in (("R", 0), ("H", 1)):
        r = {"qid": it["id"], "split": split, "which": which, "label": label, "question": it["question"], "answer": st.answers[which],
             "knowledge": it["knowledge"]}
        r.update(answer_features(pipe, st, which, names, how))
        # gold-evidence (oracle) NLI: how much does retrieval quality limit detection?
        g = answer_features(pipe, st_g, which, ["s_nli"], how)
        r["s_nli_gold"] = g["s_nli"]
        # retrieval-mode ablation (local compute only; hybrid = main run)
        for mode in ("bm25", "dense"):
            vals = []
            for c in r["claims"]:
                ev = [h.full_text for h in res.retriever.search(c, k=cfg.retrieval.evidence_k, mode=mode)]
                vals.append(1.0 - nli_against_evidence(pipe.nli, c, ev, cfg.nli.window_words, cfg.nli.window_overlap)["entail"])
            r[f"s_nli_{mode}"] = aggregate(vals, how)
        # SelfCheck vs. number of samples (re-uses the cached samples; local NLI only)
        if "s_sc" in names:
            smp = pipe.samples(st)
            per_claim = [selfcheck_per_sample(pipe.nli, c, smp) for c in r["claims"]]
            for n in cfg.stage2.n_samples_curve:
                vals = [float(p[:n].mean()) if len(p) else 0.5 for p in per_claim]
                r[f"s_sc_n{n}"] = aggregate(vals, how)
        rows.append(r)
    return rows


def _fit_and_eval(dev: pd.DataFrame, test: pd.DataFrame, detectors: Sequence[str], cal: Calibration, criterion: str = "youden") -> tuple[pd.DataFrame, Dict[str, tuple], Dict[str, np.ndarray]]:
    """Fit detector heads on dev (thresholds + logistic weights), evaluate on test. Returns metrics, test score curves, score arrays."""
    ytr, yte = dev["label"].values, test["label"].values
    mrows, curves = [], {}
    for name in detectors:
        feats = HEAD_FEATURES[name]
        if any(dev[f].isna().any() for f in feats):
            log.warning("Detector %r skipped (feature missing)", name)
            continue
        head = fit_head(name, dev[feats].values, ytr, criterion)
        cal.heads[name] = head
        te = np.array([head.score(dict(zip(feats, r))) for r in test[feats].values])
        mrows.append({"detector": name, **M.detection_metrics(yte, te, head.threshold)})
        curves[name] = (yte, te)
    # raw-feature variants (threshold fitted on dev, same protocol)
    variants = {"nli_gold": "s_nli_gold", "nli_bm25": "s_nli_bm25", "nli_dense": "s_nli_dense"}
    for name, col in variants.items():
        if col in dev and dev[col].notna().all():
            thr = M.best_threshold(ytr, dev[col].values, criterion)
            mrows.append({"detector": name, **M.detection_metrics(yte, test[col].values, thr)})
            if name == "nli_gold":
                curves[name] = (yte, test[col].values)
    return pd.DataFrame(mrows), curves, {}


def run_halueval(cfg: Cfg, res: Resources, pipe: FactualityPipeline, out_dir) -> Optional[dict]:
    dev_items = res.splits["halueval_dev"][: cfg.stage2.halueval.n_dev]
    test_items = res.splits["halueval_test"][: cfg.stage2.halueval.n_test]
    feat_path = out_dir / "features_halueval.jsonl"
    quota_err, failed = None, 0
    if cfg.stage2.get("reuse_halueval") and feat_path.exists():
        rows = read_jsonl(feat_path)          # no LLM calls: re-fit / re-evaluate from the stored features (e.g. after a quota stop)
        log.info("Reusing %d stored HaluEval feature rows from %s (no LLM calls)", len(rows), feat_path)
    else:
        work = [(it, "dev") for it in dev_items] + [(it, "test") for it in test_items]
        results, quota_err, failed = map_questions(lambda p: _halueval_rows(pipe, res, p[0], p[1], cfg), work, cfg.llm.workers, "Stage 2 HaluEval detection")
        rows = [r for rs in results if rs for r in rs]
        write_jsonl(feat_path, rows)
        if quota_err:
            raise quota_err
    df = pd.DataFrame(rows)
    dev, test = df[df["split"] == "dev"], df[df["split"] == "test"]
    cal = Calibration.default()
    metrics_df, curves, _ = _fit_and_eval(dev, test, cfg.stage2.detectors, cal, cfg.stage2.calibration)
    cal.calibrated = True
    if "nli" in cal.heads:   # 3-way (FEVER-style) decision: entail >= 1 - threshold(s_nli)
        cal.nli_entail_threshold = float(1.0 - cal.heads["nli"].threshold)
    cal.save(calibration_path(cfg))
    metrics_df.insert(0, "dataset", "halueval")
    # ---- SelfCheck sample-count curve
    curve_rows = []
    for n in cfg.stage2.n_samples_curve:
        col = f"s_sc_n{n}"
        if col in test and test[col].notna().all():
            m = M.detection_metrics(test["label"].values, test[col].values, M.best_threshold(dev["label"].values, dev[col].values, cfg.stage2.calibration))
            curve_rows.append({"n_samples": n, "auroc": m["auroc"], "auprc": m["auprc"], "f1": m["f1"]})
    curve = pd.DataFrame(curve_rows)
    if len(curve):
        curve.to_csv(out_dir / "selfcheck_n_curve.csv", index=False)
        P.fig_lines(curve.melt("n_samples", ["auroc", "auprc", "f1"], "metric", "value"), "n_samples", "value", "metric",
                    "SelfCheck-NLI vs. number of samples (HaluEval test)", out_dir / "selfcheck_n_curve.png", xlabel="stochastic samples")
    # ---- retrieval mode vs detection
    mode_df = metrics_df[metrics_df["detector"].isin(["nli", "nli_bm25", "nli_dense", "nli_gold"])].copy()
    mode_df["evidence"] = mode_df["detector"].map({"nli": "hybrid (default)", "nli_bm25": "bm25", "nli_dense": "dense", "nli_gold": "gold paragraph (oracle)"})
    mode_df.to_csv(out_dir / "retrieval_mode_vs_detection.csv", index=False)
    # ---- figures
    ordered = [d for d in cfg.stage2.detectors if d in curves] + (["nli_gold"] if "nli_gold" in curves else [])
    P.fig_roc_pr({k: curves[k] for k in ordered}, "HaluEval test", out_dir / "roc_pr_halueval.png")
    P.fig_score_hist({k: curves[k] for k in ordered if k in cal.heads}, out_dir / "score_hist_halueval.png")
    cms = {}
    for k in ordered:
        if k in cal.heads:
            y, s = curves[k]
            cms[k] = np.array([[((s < cal.heads[k].threshold) & (y == 0)).sum(), ((s >= cal.heads[k].threshold) & (y == 0)).sum()],
                               [((s < cal.heads[k].threshold) & (y == 1)).sum(), ((s >= cal.heads[k].threshold) & (y == 1)).sum()]])
    if cms:
        P.fig_confusions(cms, ["right", "hallucinated"], out_dir / "confusion_halueval.png")
    _error_analysis(test, cal, out_dir)
    return {"metrics": metrics_df, "calibration": cal, "curve": curve, "features": df, "failed": failed}


def _error_analysis(test: pd.DataFrame, cal: Calibration, out_dir, k: int = 12) -> None:
    """Markdown report with the most confident false positives / false negatives of the best-AUROC head."""
    best = max((n for n in cal.heads if n in ("evidence", "hybrid", "nli", "llm") and all(f in test and test[f].notna().all() for f in cal.heads[n].features)),
               key=lambda n: M.detection_metrics(test["label"].values, [cal.heads[n].score(dict(zip(cal.heads[n].features, r))) for r in test[cal.heads[n].features].values])["auroc"],
               default=None)
    if best is None:
        return
    h = cal.heads[best]
    t = test.copy()
    t["score"] = [h.score(dict(zip(h.features, r))) for r in t[h.features].values]
    fp = t[(t["label"] == 0) & (t["score"] >= h.threshold)].sort_values("score", ascending=False).head(k)
    fn = t[(t["label"] == 1) & (t["score"] < h.threshold)].sort_values("score").head(k)
    lines = [f"# Error analysis - detector `{best}` on HaluEval test (threshold {h.threshold:.3f})\n"]
    for title, d in (("False positives: correct answers flagged as hallucinated", fp), ("False negatives: hallucinations that were missed", fn)):
        lines.append(f"\n## {title} ({len(d)} shown)\n")
        for _, r in d.iterrows():
            lines.append(f"- **Q:** {r['question']}\n  - answer: `{r['answer']}`  | score {r['score']:.3f}\n  - claim(s): {r['claims']}\n  - top evidence: {str(r['top_evidence'])[:300]}\n  - gold paragraph: {r['knowledge'][:300]}\n")
    (out_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


# =========================================================================== RAGTruth
def _ragtruth_row(pipe: FactualityPipeline, cfg: Cfg, rec: dict, names: Sequence[str]) -> dict:
    windows: List[str] = []
    for p in rec["context"]:
        windows += chunk_text(p, cfg.nli.window_words, cfg.nli.window_overlap)
    sp = prompts.rag(rec["question"], rec["context"]) if (cfg.stage2.selfcheck_ragtruth and "s_sc" in names) else None
    st = QuestionState(rec["id"], rec["question"], fixed_evidence=windows, sample_prompt=sp)
    st.answers = {"X": rec["response"]}
    r = {"qid": rec["id"], "label": int(rec["hallucinated"]), "question": rec["question"], "answer": rec["response"], "model": rec["model"],
         "label_types": rec["label_types"], "n_spans": rec["n_spans"]}
    r.update(answer_features(pipe, st, "X", names, cfg.stage2.aggregate))
    return r


def run_ragtruth(cfg: Cfg, res: Resources, pipe: FactualityPipeline, cal: Calibration, out_dir) -> Optional[dict]:
    n = cfg.stage2.ragtruth.n
    if not n:
        return None
    recs = D.load_ragtruth(cfg, "test", tuple(cfg.stage2.ragtruth.tasks))
    recs = D.sample_ragtruth(recs, n, cfg.stage2.ragtruth.pos_frac, cfg.project.seed)
    names = ["s_nli", "s_llm"] + (["s_sc"] if cfg.stage2.selfcheck_ragtruth else [])
    results, quota_err, failed = map_questions(lambda r: _ragtruth_row(pipe, cfg, r, names), recs, cfg.llm.workers, "Stage 2 RAGTruth detection")
    rows = [r for r in results if r]
    write_jsonl(out_dir / "features_ragtruth.jsonl", rows)
    if quota_err:
        raise quota_err
    df = pd.DataFrame(rows)
    y = df["label"].values
    mrows, curves = [], {}
    for name in ("nli", "llm", "evidence") + (("hybrid",) if "s_sc" in names else ()):
        if name not in cal.heads:
            continue
        h = cal.heads[name]
        s = np.array([h.score(dict(zip(h.features, r))) for r in df[h.features].values])
        mrows.append({"dataset": "ragtruth", "detector": name, **M.detection_metrics(y, s, h.threshold),
                      "f1_at_best_threshold": M.detection_metrics(y, s, M.best_f1_threshold(y, s))["f1"]})
        curves[name] = (y, s)
    if curves:
        P.fig_roc_pr(curves, "RAGTruth (zero-shot from HaluEval)", out_dir / "roc_pr_ragtruth.png")
    return {"metrics": pd.DataFrame(mrows), "features": df, "failed": failed}


# =========================================================================== FEVER
def run_fever(cfg: Cfg, res: Resources, nli, llm, cal: Calibration, out_dir) -> Optional[dict]:
    n = cfg.stage2.fever.n
    if not n:
        return None
    try:
        items = D.load_fever(cfg, "validation")
    except Exception as e:
        log.warning("FEVER unavailable (%s) - skipping the FEVER experiment. Set stage2.fever.n=0 to silence this.", e)
        return None
    sample = D.sample_fever(items, n, cfg.project.seed)
    sample_ids = {x["id"] for x in sample}
    extra = [x for x in items if x["id"] not in sample_ids and x["evidence"]][:2000]   # distractor evidence for the retrieval pool
    docs = []
    for x in sample + extra:
        for j, ev in enumerate(x["evidence"]):
            docs.append((f"{x['id']}:{j}", ev["title"], ev["text"]))
    retr, _ = build_pool_kb(cfg, docs, res.embedder, "fever")
    pipe = FactualityPipeline(cfg, llm, retr, nli, cal)
    how = cfg.stage2.aggregate

    def work(it: dict) -> dict:
        gold = [f"{e['title']}: {e['text']}" if e["title"] else e["text"] for e in it["evidence"]]
        out = {"qid": it["id"], "claim": it["claim"], "label": it["label"]}
        for variant, fixed in (("retrieved", None), ("gold", gold if it["label"] != "NOT ENOUGH INFO" else None)):
            st = QuestionState(f"{it['id']}#{variant}", "", fixed_evidence=fixed)
            st.preset_claims = {"X": [it["claim"]]}
            row = pipe.features(st, "X", ["s_nli", "s_llm"])[0]
            nl, lv = row["detail"]["nli"], row["detail"]["llm"]
            out[f"s_nli_{variant}"], out[f"s_llm_{variant}"] = row["feats"]["s_nli"], row["feats"]["s_llm"]
            out[f"entail_{variant}"], out[f"contra_{variant}"] = nl["entail"], nl["contra"]
            out[f"llm_verdict_{variant}"] = lv["verdict"]
            out[f"top_evidence_{variant}"] = ptxt(row["evidence"][0]) if row["evidence"] else ""
        return out

    results, quota_err, failed = map_questions(work, sample, cfg.llm.workers, "Stage 2 FEVER verification")
    rows = [r for r in results if r]
    write_jsonl(out_dir / "features_fever.jsonl", rows)
    if quota_err:
        raise quota_err
    df = pd.DataFrame(rows)
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    llm_map = {"SUPPORTED": "SUPPORTS", "CONTRADICTED": "REFUTES", "NOT_ENOUGH_INFO": "NOT ENOUGH INFO"}
    summary, cms = {}, {}
    for variant in ("retrieved", "gold"):
        pred_nli = [("SUPPORTS" if e >= cal.nli_entail_threshold else "REFUTES" if (c >= cal.nli_contra_threshold and c > e) else "NOT ENOUGH INFO")
                    for e, c in zip(df[f"entail_{variant}"], df[f"contra_{variant}"])]
        pred_llm = [llm_map[v] for v in df[f"llm_verdict_{variant}"]]
        for name, pred in (("nli", pred_nli), ("llm", pred_llm)):
            m = M.multiclass_metrics(df["label"].tolist(), pred, labels)
            summary[f"{name}_{variant}"] = m
            cms[f"{name} ({variant} evidence)"] = np.array(m["confusion"])
    # binary: "claim is NOT supported" (REFUTES + NEI) as the positive class, heads fitted on HaluEval dev
    yb = (df["label"] != "SUPPORTS").astype(int).values
    bin_rows = []
    for name in ("nli", "llm", "evidence"):
        if name in cal.heads:
            h = cal.heads[name]
            for variant in ("retrieved", "gold"):
                s = np.array([h.score({"s_nli": a, "s_llm": b}) for a, b in zip(df[f"s_nli_{variant}"], df[f"s_llm_{variant}"])])
                bin_rows.append({"dataset": f"fever ({variant} evidence)", "detector": name, **M.detection_metrics(yb, s, h.threshold)})
    write_json(out_dir / "fever_3way.json", summary)
    P.fig_confusions(cms, ["SUP", "REF", "NEI"], out_dir / "fever_confusion.png")
    return {"metrics": pd.DataFrame(bin_rows), "three_way": summary, "failed": failed}


# =========================================================================== driver
def run(cfg: Cfg, res: Optional[Resources] = None) -> dict:
    banner(cfg, "Stage 2")
    res = res or load_resources(cfg)
    out_dir = results_dir(cfg, "stage2")
    pipe = FactualityPipeline(cfg, res.llm, res.retriever, res.nli)
    hv = run_halueval(cfg, res, pipe, out_dir)
    cal = hv["calibration"]
    pipe.cal = cal
    rt = run_ragtruth(cfg, res, pipe, cal, out_dir)
    fv = run_fever(cfg, res, res.nli, res.llm, cal, out_dir)
    parts = [hv["metrics"]] + [x["metrics"] for x in (rt, fv) if x]
    allm = pd.concat(parts, ignore_index=True)
    allm.to_csv(out_dir / "detection_metrics.csv", index=False)
    main = allm[allm["detector"].isin(["nli", "llm", "selfcheck", "cove", "evidence", "hybrid"])]
    main = main.assign(dataset=main["dataset"].str.replace(r" \(.*\)", "", regex=True))
    if len(main):
        P.fig_detector_bars(main.drop_duplicates(["dataset", "detector"]), path=out_dir / "detector_comparison.png")
    write_json(out_dir / "meta.json", run_meta(cfg, n_halueval_dev=cfg.stage2.halueval.n_dev, n_halueval_test=cfg.stage2.halueval.n_test,
                                               llm_stats=res.llm.stats_table()))
    log.info("Stage 2 done -> %s", out_dir)
    return {"metrics": allm, "halueval": hv, "ragtruth": rt, "fever": fv, "calibration": cal, "dir": out_dir}
