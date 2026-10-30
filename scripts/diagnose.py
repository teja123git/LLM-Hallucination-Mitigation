#!/usr/bin/env python
"""Sanity-check a (partial) run and print ONE compact report you can paste when asking for help.
Contains no secrets. Usage:  python scripts/diagnose.py [--results results]"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
R = Path(ap.parse_args().results)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.float_format", "{:.3f}".format)
flags = []


def flag(level, msg):
    flags.append((level, msg))


def jread(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def lines(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def meta_block(stage):
    m = jread(R / stage / "meta.json")
    print(f"\n===== {stage}: meta =====")
    if not m:
        print("(no meta.json - this stage did not finish writing results)")
        flag("INFO", f"{stage}: no meta.json (stage not finished or quota stopped it early)")
        return None
    print({k: m.get(k) for k in ("timestamp", "mock", "model", "judge_model", "n_failed", "interrupted", "judge_failed", "judged", "judge_equals_generator")})
    if m.get("mock"):
        flag("FAIL", f"{stage}: this is a MOCK run - not real results")
    if m.get("n_failed"):
        flag("WARN", f"{stage}: {m['n_failed']} items failed with LLM errors and were skipped")
    if m.get("interrupted"):
        flag("INFO", f"{stage}: interrupted by the daily quota - re-run the same command later to finish")
    st = m.get("llm_stats") or {}
    if st:
        t = pd.DataFrame(st).T[["calls", "cached_calls", "prompt_tokens", "completion_tokens"]]
        print(t.to_string())
        for task, r in t.iterrows():
            if r["calls"] and r["completion_tokens"] / r["calls"] > 600:
                flag("WARN", f"{stage}: task '{task}' uses {r['completion_tokens'] / r['calls']:.0f} completion tokens/call (reasoning overhead?) - consider reasoning_effort: low")
    return m


# ------------------------------------------------------------------ Stage 1
meta_block("stage1")
out = lines(R / "stage1" / "outputs.jsonl")
if out:
    empty = sum(1 for o in out if not o["final_answer"].strip())
    print(f"\nanswers: {len(out)} | empty answers: {empty}")
    if empty:
        flag("FAIL", f"stage1: {empty} EMPTY answers (reasoning model used the whole token budget?)")
    err = [o for o in out if o["final_answer"].startswith("Error")]
    if err:
        flag("FAIL", "stage1: answers that look like error strings")
    print("\nexample answers:")
    for o in out[:4]:
        print(f"  [{o['system']}] Q: {o['question'][:80]}\n       -> {o['final_answer'][:160]}")
ab = [o for o in out if o["system"] == "B" and o["abstained"]]
if ab:
    print(f"\nRAG (B) abstained on {len(ab)} questions; first ones:")
    for o in ab[:8]:
        print(f"  Q: {o['question'][:85]}\n     -> {o['final_answer'][:150]}")
s1 = R / "stage1" / "summary_by_dataset.csv"
if s1.exists():
    d = pd.read_csv(s1)
    cols = [c for c in ["dataset", "system", "n", "abstention_rate", "contains", "tqa_truthful_sim", "bertscore_f1", "rougeL_f1", "cost_calls", "cost_total_tokens"] if c in d]
    print("\n", d[cols].to_string(index=False))
    for _, r in d.iterrows():
        if r["system"] == "B" and r["abstention_rate"] > 0.5 and r["dataset"] == "halueval":
            flag("WARN", f"stage1: RAG abstains on {r['abstention_rate']:.0%} of HaluEval questions - retrieval or prompt problem?")
if s1.exists():
    d1 = pd.read_csv(s1)
    t = d1[(d1.dataset == "truthfulqa") & (d1.system == "B")]
    if len(t) and t["abstention_rate"].iloc[0] > 0.7:
        flag("WARN", f"stage1: RAG abstains on {t['abstention_rate'].iloc[0]:.0%} of TruthfulQA questions - Wikipedia pages are probably missing from the knowledge base "
                     "(check the 'Wikipedia pages available' log line / data/cache/wikipedia, see docs/DATA.md)")
wk = list((Path("data") / "cache" / "wikipedia").glob("*.json"))
if wk:
    ok = sum(1 for f in wk if json.loads(f.read_text(encoding="utf-8")).get("text"))
    print(f"\nWikipedia cache: {len(wk)} files, {ok} with text (expected ~357 files, most with text)")
    if ok < 0.5 * len(wk) or ok < 100:
        flag("WARN", f"stage1: only {ok} Wikipedia pages have text - fetching failed (HTTP 403 = set kb.wikipedia_user_agent with your e-mail)")
else:
    print("\nWikipedia cache: none found (data/cache/wikipedia is missing)")
    flag("WARN", "stage1: no Wikipedia pages cached - TruthfulQA RAG has no evidence")
rm = R / "stage1" / "retrieval_metrics.csv"
if rm.exists():
    d = pd.read_csv(rm)
    if "truthfulqa" not in set(d["dataset"]):
        flag("WARN", "stage1: no retrieval metrics for TruthfulQA (none of the sampled questions has its Wikipedia source in the KB)")
    print("\n", d.round(3).to_string(index=False))
    h = d[(d.dataset == "halueval") & (d["mode"] == "hybrid")]
    if len(h) and h["recall@5"].iloc[0] < 0.7:
        flag("WARN", f"stage1: hybrid Recall@5 on HaluEval is only {h['recall@5'].iloc[0]:.2f} (expected well above 0.7)")
    b, de = d[(d.dataset == "halueval") & (d["mode"] == "bm25")], d[(d.dataset == "halueval") & (d["mode"] == "dense")]
    if len(b) and len(de) and len(h) and h["recall@5"].iloc[0] + 0.02 < max(b["recall@5"].iloc[0], de["recall@5"].iloc[0]):
        flag("INFO", "stage1: hybrid is worse than its best single stream on HaluEval - worth discussing in the report")

# ------------------------------------------------------------------ Stage 2
meta_block("stage2")
feats = lines(R / "stage2" / "features_halueval.jsonl")
if feats:
    df = pd.DataFrame(feats)
    print(f"\nHaluEval feature rows: {len(df)} (dev {int((df.split == 'dev').sum())}, test {int((df.split == 'test').sum())})")
    nocl = int((df["claims"].str.len() == 0).sum())
    print("answers with zero claims:", nocl)
    if nocl > 0.1 * len(df):
        flag("WARN", f"stage2: {nocl}/{len(df)} answers produced no claims (claim extraction failing?)")
    for f in ("s_nli", "s_llm", "s_sc", "s_cove"):
        if f in df and df[f].notna().any():
            g = df.groupby("label")[f].mean()
            print(f"  mean {f}: right={g.get(0, float('nan')):.3f}  hallucinated={g.get(1, float('nan')):.3f}")
            if g.get(1, 0) <= g.get(0, 1):
                flag("WARN", f"stage2: {f} is NOT higher for hallucinated answers - this signal is not working")
    if "s_llm" in df and df["s_llm"].notna().any() and (df["s_llm"] == 0.5).mean() > 0.6:
        flag("WARN", "stage2: LLM verifier says NOT_ENOUGH_INFO for >60% of answers (evidence/prompt/empty-output problem?)")
dm = R / "stage2" / "detection_metrics.csv"
if dm.exists():
    d = pd.read_csv(dm)
    print("\n", d[[c for c in ["dataset", "detector", "n", "threshold", "accuracy", "precision", "recall", "f1", "auroc", "auprc"] if c in d]].round(3).to_string(index=False))
    for _, r in d.iterrows():
        if r["detector"] in ("nli", "llm", "selfcheck", "cove", "evidence", "hybrid") and r["auroc"] < 0.55:
            flag("WARN", f"stage2: detector {r['detector']} on {r['dataset']} has AUROC {r['auroc']:.2f} (~chance)")
else:
    flag("INFO", "stage2: detection_metrics.csv not written yet (Stage 2 must finish; it writes calibration.json too)")
cal = jread(R / "stage2" / "calibration.json")
if cal:
    th = {k: round(v["threshold"], 3) for k, v in cal["heads"].items()}
    print("\ncalibrated thresholds:", th, "| calibrated:", cal.get("calibrated"))
    for k, v in th.items():
        if v >= 0.95 and k in ("selfcheck", "cove", "nli", "llm", "evidence", "hybrid"):
            flag("INFO", f"stage2: detector {k} has threshold {v} - it will flag almost nothing (weak detector or too few dev samples)")
        if v <= 0.0:
            flag("FAIL", f"stage2: detector {k} threshold {v} flags EVERY claim")

# ------------------------------------------------------------------ Stage 3
m3 = meta_block("stage3")
s3 = R / "stage3" / "summary_overall.csv"
if s3.exists():
    d = pd.read_csv(s3)
    print("\n", d[[c for c in ["system", "n", "abstention_rate", "correct", "incorrect", "not_attempted", "f_score", "free_correct", "cost_calls", "cost_total_tokens"] if c in d]].round(3).to_string(index=False))

print("\n================ VERDICT ================")
if not flags:
    print("No problems detected.")
for lvl in ("FAIL", "WARN", "INFO"):
    for l, msg in flags:
        if l == lvl:
            print(f"[{l}] {msg}")
print("\nSafe to push code? yes if there is no [FAIL] above. Push real results only after Stages 2 and 3 have finished (meta.json present, not 'interrupted').")
