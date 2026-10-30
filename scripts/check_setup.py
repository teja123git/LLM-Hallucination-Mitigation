#!/usr/bin/env python
"""Verify your environment before a long run: API key, models, data, retrieval and a rough token budget."""
import time

from _cli import base_parser, load

ap = base_parser(__doc__)
args = ap.parse_args()
cfg = load(args)
ok = True


def step(name, fn):
    global ok
    t = time.time()
    try:
        r = fn()
        print(f"[ OK ] {name} ({time.time() - t:.1f}s){' - ' + str(r) if r else ''}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


from halluguard import data as D  # noqa: E402
from halluguard.factory import make_llm  # noqa: E402
from halluguard.nli import make_nli  # noqa: E402
from halluguard.retrieval import make_embedder  # noqa: E402
from halluguard.budget import print_budget  # noqa: E402

llm = {}
step("Groq API key + model " + cfg.llm.model, lambda: llm.setdefault("c", make_llm(cfg)) and llm["c"].chat("Reply with one word: ok", max_tokens=5, temperature=0).text)
step("judge model " + cfg.llm.judge_model, lambda: llm["c"].chat("Reply with one word: ok", model=llm["c"].judge_model, max_tokens=5, temperature=0).text if "c" in llm else None)
step("sentence embedder", lambda: make_embedder(cfg).encode(["hello world"]).shape)
step("NLI model (+ label-mapping sanity check)", lambda: make_nli(cfg).sanity_check())
step("HaluEval / TruthfulQA / RAGTruth download", lambda: (len(D.load_halueval_qa(cfg)), len(D.load_truthfulqa(cfg)), len(D.load_ragtruth(cfg, "test", ("QA",)))))
try:
    step("FEVER (optional)", lambda: len(D.load_fever(cfg)))
except Exception:
    pass
try:
    from bert_score import score  # noqa: F401
    print("[ OK ] bert-score importable")
except Exception as e:
    print(f"[WARN] bert-score not importable ({e}); BERTScore columns will be skipped")
if not cfg.project.get("mock") and "c" in llm and hasattr(llm["c"], "list_models"):
    try:
        avail = llm["c"].list_models()
        for m in (cfg.llm.model, cfg.llm.judge_model):
            if m not in avail:
                print(f"[WARN] '{m}' is NOT in your Groq model list. Chat models available to your key: " + ", ".join(x for x in avail if "whisper" not in x and "tts" not in x and "guard" not in x))
    except Exception as e:
        print(f"[WARN] could not list models: {e}")
print_budget(cfg)
print("All good - run `python scripts/run_all.py`." if ok else "Fix the failures above first.")
