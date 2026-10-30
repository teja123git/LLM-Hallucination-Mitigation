"""Rough token / call budget estimator (so you know what a run will cost *before* starting it).

The constants are order-of-magnitude averages (prompt + completion tokens) for Llama-3.1-8B on these prompts; real usage
is measured exactly and stored in ``results/<stage>/meta.json`` (``llm_stats``) after a run.
"""
from __future__ import annotations

from typing import Dict, Sequence

from .config import Cfg
from .pipeline import SYSTEMS

# stage -> (LLM calls, tokens) per question
STAGE = {
    "answer:A": (1, 110), "answer:B": (1, 1000), "claims": (1, 430), "samples": (None, None),   # samples scale with n_samples
    "llmv": (1.5, 900), "cove": (3, 1200), "revise": (0.4, 400), "covecorr": (0.4, 1200), "judge": (1, 600),
}
# Groq free tier, per model (checked Oct 2026): tokens/day and requests/day. Check console.groq.com/settings/limits for yours.
FREE_TIER = {"openai/gpt-oss-20b": {"tpd": 200_000, "rpd": 1_000}, "openai/gpt-oss-120b": {"tpd": 200_000, "rpd": 1_000}}
REASONING_TOKENS_PER_CALL = 150   # ASSUMPTION: hidden reasoning at reasoning_effort=low; measure yours in results/<stage>/meta.json (llm_stats)


def _days(model: str, tokens: float, calls: float) -> float:
    lim = FREE_TIER.get(model, {"tpd": 200_000, "rpd": 1_000})
    return max(tokens / lim["tpd"], calls / lim["rpd"])


def _reasoning(cfg: Cfg, calls: float) -> float:
    return calls * REASONING_TOKENS_PER_CALL if cfg.llm.model.startswith("openai/gpt-oss") else 0.0


def stage3_estimate(cfg: Cfg, n_per_dataset: int | None = None, systems: Sequence[str] | None = None) -> Dict[str, float]:
    n_q = 2 * (n_per_dataset or cfg.stage3.n_per_dataset)
    systems = list(systems or cfg.stage3.systems)
    ns = cfg.selfcheck.n_samples
    stages = {"answer:A": STAGE["answer:A"], "claims_A": STAGE["claims"]}
    if any(SYSTEMS[s]["init"] in ("B", "R") for s in systems):
        stages.update({"answer:B": STAGE["answer:B"], "claims_B": STAGE["claims"]})
    if any(SYSTEMS[s]["detector"] in ("selfcheck", "hybrid") for s in systems):
        stages["samples"] = (ns, ns * 110)
    if any(SYSTEMS[s]["detector"] == "cove" for s in systems):
        stages["cove"] = STAGE["cove"]
    if any(SYSTEMS[s]["detector"] in ("evidence", "hybrid") for s in systems):
        stages["llmv"] = STAGE["llmv"]
    if any(SYSTEMS[s]["action"] == "rarr" for s in systems):
        stages["revise"] = STAGE["revise"]
    if any(SYSTEMS[s]["action"] == "cove" for s in systems):
        stages["covecorr"] = STAGE["covecorr"]
    calls = sum(c for c, _ in stages.values()) * n_q
    tokens = sum(t for _, t in stages.values()) * n_q
    jc = len(systems) * 0.9 * n_q if cfg.stage3.use_llm_judge else 0
    jt = jc * STAGE["judge"][1]
    return {"generator_calls": calls, "generator_tokens": tokens + _reasoning(cfg, calls), "judge_calls": jc, "judge_tokens": jt + _reasoning(cfg, jc),
            "generator_days_free_tier": _days(cfg.llm.model, tokens + _reasoning(cfg, calls), calls),
            "judge_days_free_tier": _days(cfg.llm.judge_model, jt + _reasoning(cfg, jc), jc)}


def stage2_estimate(cfg: Cfg) -> Dict[str, float]:
    h = cfg.stage2.halueval
    n_ans = 2 * (h.n_dev + h.n_test)
    calls = n_ans * (1 + STAGE["llmv"][0])
    tok = n_ans * (STAGE["claims"][1] + STAGE["llmv"][1])
    if "cove" in cfg.stage2.detectors:
        calls += n_ans * 3
        tok += n_ans * STAGE["cove"][1]
    if set(cfg.stage2.detectors) & {"selfcheck", "hybrid"}:
        calls += (h.n_dev + h.n_test) * cfg.selfcheck.n_samples
        tok += (h.n_dev + h.n_test) * cfg.selfcheck.n_samples * 110
    calls += cfg.stage2.fever.n + cfg.stage2.ragtruth.n * 3
    tok += cfg.stage2.fever.n * 700 + cfg.stage2.ragtruth.n * (STAGE["claims"][1] + 2 * 800)
    tok += _reasoning(cfg, calls)
    return {"generator_calls": calls, "generator_tokens": tok, "generator_days_free_tier": _days(cfg.llm.model, tok, calls)}


def stage1_estimate(cfg: Cfg) -> Dict[str, float]:
    calls = 4 * cfg.stage1.n_per_dataset
    tok = 2 * cfg.stage1.n_per_dataset * (STAGE["answer:A"][1] + STAGE["answer:B"][1]) + _reasoning(cfg, calls)
    return {"generator_calls": calls, "generator_tokens": tok, "generator_days_free_tier": _days(cfg.llm.model, tok, calls)}


def print_budget(cfg: Cfg) -> None:
    rows = [("Stage 1", stage1_estimate(cfg)), ("Stage 2", stage2_estimate(cfg)), ("Stage 3", stage3_estimate(cfg))]
    print(f"\nRough budget at current config (generator = {cfg.llm.model}, judge = {cfg.llm.judge_model}):")
    print(f"{'stage':8s} {'gen calls':>10s} {'gen tokens':>12s} {'~days (free)':>13s} | {'judge calls':>11s} {'judge tokens':>13s} {'~days (free)':>13s}")
    for name, e in rows:
        print(f"{name:8s} {e['generator_calls']:>10,.0f} {e['generator_tokens']:>12,.0f} {e['generator_days_free_tier']:>13.1f} | "
              f"{e.get('judge_calls', 0):>11,.0f} {e.get('judge_tokens', 0):>13,.0f} {e.get('judge_days_free_tier', 0):>13.1f}")
    print("Days = max(tokens/day limit, requests/day limit) on the Groq FREE tier (200K tokens and 1,000 requests per model per day) - an estimate.")
    print("Runs are resumable: every finished LLM call is cached, so you can stop (or hit the daily quota) and re-run the same command.\n")
