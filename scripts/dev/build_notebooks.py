#!/usr/bin/env python
"""Regenerates the three notebooks (01/02/03) from the cell definitions below.  Notebooks are thin: all logic lives in
src/halluguard so the notebooks, the CLI scripts and the tests execute exactly the same code."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = '''import os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")          # figures are written to results/ and shown with IPython.display.Image
from IPython.display import Image, display, Markdown
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.float_format", "{:.3f}".format)

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

# ---- choose how to run -----------------------------------------------------------------------------------------
PROFILE = None          # None = configs/default.yaml | "configs/quick.yaml" = small sanity run (recommended first)
MOCK = os.environ.get("HALLUGUARD_MOCK") == "1"      # offline smoke test with FAKE models - numbers are meaningless
from halluguard.config import load_config
cfg = load_config(PROFILE, mock=MOCK)
print("mock:", MOCK, "| generator:", cfg.llm.model, "| judge:", cfg.llm.judge_model, "| results ->", cfg.project.results_dir)

def show(name, stage):
    p = Path(cfg.project.results_dir) / stage / name
    display(Image(filename=str(p))) if p.exists() else print("(missing)", p)
'''

nb1 = [
md('''# Stage 1 - Data, knowledge base, retrieval and baselines
**DA624 - Improving Factuality of LLMs through RAG and Self-Verification**

This notebook re-implements Stage 1 on top of the shared `halluguard` package and **fixes several problems of the first version** (see `docs/CHANGES_FROM_STAGE1.md`):

| Problem in the first version | Fix |
|---|---|
| Knowledge base contained TruthfulQA's *correct and incorrect answer strings* (RAG could read the answer key) | KB = source documents only: HaluEval paragraphs (pooled) + Wikipedia pages cited by TruthfulQA |
| Failed API calls were stored as `"Error: ..."` answers and scored | failures raise; items are skipped and counted |
| `sleep(0.5)` = 120 req/min vs. a 30 req/min limit | token- and request-aware limiter, persistent cache, resumable runs |
| Recall@k / MRR / context coverage / prompt-vs-completion tokens promised but not computed | implemented |
| BM25 tokenisation left punctuation attached; zero-score docs entered the fusion | regex tokenizer + stop-words; zero scores dropped |

**Run order:** setup -> data -> knowledge base & retrieval -> baselines A/B -> evaluation. First time? use `PROFILE = "configs/quick.yaml"`.'''),
code(SETUP),
md('## 1. Datasets and splits\nHaluEval-QA (10,000 questions with gold paragraph, right and hallucinated answers) and TruthfulQA (790 questions, 475 cite Wikipedia). The three HaluEval pools (dev / test / end-to-end) are **disjoint** so detector calibration never sees the questions used for the final evaluation.'),
code('''from halluguard.experiments.common import load_resources
res = load_resources(cfg)                        # downloads data on first use, builds / loads the knowledge base
print({k: len(v) for k, v in res.splits.items()})
print("TruthfulQA questions with a Wikipedia source:", sum(bool(x["wiki_titles"]) for x in res.tqa), "/", len(res.tqa))
pd.DataFrame(res.splits["halueval_e2e"][:3])[["id", "question", "right_answer", "hallucinated_answer"]]'''),
md('## 2. Knowledge base & hybrid retrieval\nDense (MiniLM + FAISS) and sparse (BM25) rankings are fused with Reciprocal Rank Fusion, `score(d) = sum 1/(k + rank)`, `k = 60`. Passage count and a retrieval sanity check:'),
code('''print("passages:", res.retriever.size)
q = res.splits["halueval_e2e"][0]
print("Q:", q["question"], "| gold answer:", q["right_answer"])
for h in res.retriever.search(q["question"], k=3):
    print(f"  [{h.rank}] rrf={h.score:.4f} dense_rank={h.dense_rank} bm25_rank={h.sparse_rank} | {h.text[:110]}...")'''),
md('## 3. Baselines A (LLM only) and B (hybrid RAG) + retrieval evaluation\nOne call runs both baselines on the end-to-end pool, computes retrieval Recall@k / MRR (bm25 vs. dense vs. hybrid), context coverage, BERTScore, ROUGE-L, EM/F1, abstention rate, tokens and latency, and writes everything to `results/stage1/`.'),
code('''from halluguard.experiments import stage1
r1 = stage1.run(cfg, res)
r1["retrieval"]'''),
md('### Retrieval quality\nHaluEval has a known gold paragraph per question, so Recall@k and MRR are exact. TruthfulQA is evaluated at page level where a Wikipedia source exists.'),
code('show("retrieval.png", "stage1"); r1["coverage"]'),
md('### Baseline comparison\n`abstention_rate` matters: Baseline B is instructed to say when the context is insufficient, which is *not* a hallucination but also not an answer. Use the attempted-only columns and Stage 3\'s judge grades (which score abstentions separately) for a fair comparison.'),
code('''cols = ["dataset","system","n","abstention_rate","bertscore_f1","rougeL_f1","contains","tqa_truthful_sim","cost_calls","cost_prompt_tokens","cost_completion_tokens","cost_latency"]
r1["summary"][[c for c in cols if c in r1["summary"]]]'''),
code('show("quality.png", "stage1")'),
md('### Dataset EDA\n'),
code('show("eda.png", "stage1")'),
md('''## 4. What to look for / caveats
* `contains` (HaluEval) and `tqa_truthful_sim` (TruthfulQA) are judge-free correctness proxies; BERTScore / ROUGE against a 2-word entity answer are weak signals and are kept only for continuity with the original report.
* The KB is an *oracle-in-pool* setting: the gold paragraph is present among ~10k distractors. Retrieval is therefore easier than open-web RAG - report it as such.
* TruthfulQA questions without a Wikipedia source (`tqa_has_wiki = False`) have no gold evidence in the KB; RAG should abstain there, and Stage 3 reports them separately.
* Next: `02_hallucination_detection.ipynb`.'''),
]

nb2 = [
md('''# Stage 2 - Claim extraction and hallucination detection
Pipeline: **answer -> atomic claims -> detector score per claim -> answer-level decision**. Every score is in [0,1] (higher = more likely hallucinated).

| signal | definition | needs |
|---|---|---|
| `s_nli` | 1 - max entailment of the claim given retrieved evidence (DeBERTa-v3 NLI cross-encoder) | retrieval |
| `s_llm` | LLM verifier verdict: SUPPORTED 0 / NOT_ENOUGH_INFO 0.5 / CONTRADICTED 1 | retrieval |
| `s_sc` | SelfCheckGPT-NLI: mean over N samples (T=0.7) of P(contra)/(P(contra)+P(entail)) | N extra samples |
| `s_cove` | Chain-of-Verification without retrieval: independent verification questions | 3 LLM calls |

Detectors: `nli`, `llm`, `selfcheck`, `cove`, `evidence` (nli+llm), `hybrid` (nli+llm+selfcheck; logistic regression). Thresholds are fitted on the HaluEval **dev** split with **Youden's J** (prevalence-invariant - F1-optimal thresholds collapse to "flag everything" for weak detectors on a 50/50 benchmark) and reported on the disjoint **test** split. RAGTruth and FEVER are zero-shot transfer.'''),
code(SETUP),
code('''from halluguard.budget import print_budget
print_budget(cfg)            # rough token budget BEFORE you start (runs are resumable: finished LLM calls are cached)'''),
md('## 1. Claim extraction, one example\nA bare entity answer ("Arthur\'s Magazine") is meaningless without its question, so claims are *decontextualised*.'),
code('''from halluguard.experiments.common import load_resources
from halluguard.pipeline import FactualityPipeline, QuestionState
res = load_resources(cfg)
pipe = FactualityPipeline(cfg, res.llm, res.retriever, res.nli)
it = res.splits["halueval_dev"][0]
st = QuestionState(it["id"], it["question"]); st.answers = {"R": it["right_answer"], "H": it["hallucinated_answer"]}
for which in ("R", "H"):
    rows = pipe.features(st, which, ["s_nli", "s_llm", "s_sc"])
    print(f"[{which}] answer = {st.answers[which]!r}")
    for r in rows:
        print("   claim:", r["claim"]); print("   scores:", {k: round(v, 3) for k, v in r["feats"].items() if v is not None})
        print("   top evidence:", (r["evidence"][0]["text"][:120] + "...") if r["evidence"] else "-")'''),
md('## 2. Run the full Stage 2 benchmark\nHaluEval (answer level, dev/test), RAGTruth QA (response level), FEVER (3-way). Writes `results/stage2/` incl. `calibration.json`, which Stage 3 loads.'),
code('''from halluguard.experiments import stage2
r2 = stage2.run(cfg, res)
cols = ["dataset","detector","n","pos_rate","threshold","accuracy","precision","recall","f1","auroc","auprc"]
r2["metrics"][cols]'''),
md('### Detector comparison and ROC / PR (HaluEval test)'),
code('show("detector_comparison.png", "stage2"); show("roc_pr_halueval.png", "stage2")'),
md('### Does retrieval quality limit detection? Gold paragraph vs. bm25 / dense / hybrid evidence'),
code('''import pandas as pd
d = pd.read_csv(Path(cfg.project.results_dir) / "stage2" / "retrieval_mode_vs_detection.csv")
d[["evidence","auroc","auprc","f1","precision","recall"]]'''),
md('### How many samples does SelfCheck need?'),
code('show("selfcheck_n_curve.png", "stage2")'),
md('### RAGTruth (naturally generated RAG hallucinations) and FEVER (3-way verification)'),
code('''rt = r2["ragtruth"]; display(rt["metrics"] if rt else "RAGTruth skipped")
show("roc_pr_ragtruth.png", "stage2"); show("fever_confusion.png", "stage2")'''),
md('### Confusion matrices and score distributions (HaluEval test)'),
code('show("confusion_halueval.png", "stage2"); show("score_hist_halueval.png", "stage2")'),
md('### Error analysis\n`results/stage2/error_analysis.md` lists the most confident false positives / false negatives of the best detector together with the retrieved evidence and the gold paragraph.'),
code('print((Path(cfg.project.results_dir) / "stage2" / "error_analysis.md").read_text()[:3500])'),
md('''### Calibrated detector heads (used by Stage 3)'''),
code('''pd.DataFrame([h.to_dict() for h in r2["calibration"].heads.values()])[["name","features","kind","weights","bias","threshold"]]'''),
md('''## Reading the results
* Compare `evidence` / `hybrid` against `nli`, `llm`, `selfcheck`, `cove` - *which signal carries the detection, and does fusion help?* (look at AUROC/AUPRC, which do not depend on thresholds).
* `nli_gold` vs. `nli` shows how much of the remaining error is due to retrieval rather than verification.
* RAGTruth is imbalanced (see `pos_rate`): prefer AUPRC over accuracy.
* Caveat: thresholds are fitted on answer-level HaluEval labels and applied to individual claims in Stage 3 (HaluEval answers are mostly single-claim, so the scales match).'''),
]

nb3 = [
md('''# Stage 3 - Evidence-guided correction and the controlled ablation
Eight systems answer the *same* questions (paired design):

| id | system | what it adds |
|---|---|---|
| A | LLM only | - |
| B | LLM + RAG | retrieval |
| C | LLM + self-consistency | SelfCheck-NLI flags -> flagged claims removed |
| D | LLM + verification | CoVe (no retrieval) flags -> removed |
| E | LLM + RAG + verification | NLI + LLM verifier vs. retrieved evidence -> removed |
| F | E + correction | RARR-style rewrite from evidence, **verify-then-accept** with NLI, citations |
| G | F with the hybrid detector | adds SelfCheck signal to the detector |
| F-CoVe | E + CoVe-style correction | alternative correction strategy |

If RAG abstains, E/F/G verify the parametric answer against retrieved evidence (`pipeline.rag_fallback_to_parametric`). Cost of a system = the stages it depends on (shared stages replay their cost), so latency/tokens/calls are what each system would spend alone.

**Evaluation**: an independent stronger judge grades CORRECT / INCORRECT / NOT_ATTEMPTED against *gold* references (so abstaining cannot inflate scores) and labels each claim SUPPORTED / CONTRADICTED / UNVERIFIABLE (FActScore-style precision). Judge-free proxies are always reported too.'''),
code(SETUP),
code('''from halluguard.budget import print_budget
print_budget(cfg)
# Judge options on a free key (70B judge = 100K tokens/day):  USE_JUDGE=False  |  cfg.llm.judge_model = "llama-3.1-8b-instant" (self-judging, flagged in meta.json)
USE_JUDGE = cfg.stage3.use_llm_judge
SYSTEMS = None          # e.g. ["A","B","E","F"] for a cheaper subset'''),
code('''from halluguard.experiments.common import load_resources
from halluguard.experiments import stage3
res = load_resources(cfg)
r3 = stage3.run(cfg, res, systems=SYSTEMS, use_judge=USE_JUDGE)'''),
md('## 1. Headline table'),
code('''s = r3["summary_overall"]
cols = ["system","n","correct","incorrect","not_attempted","correct_given_attempted","f_score","claim_support_rate","claim_contradiction_rate","factscore","free_correct","free_incorrect","cost_calls","cost_total_tokens","cost_latency"]
s[[c for c in cols if c in s]]'''),
code('show("grades.png", "stage3")'),
md('## 2. Which component helps? Paired contrasts with bootstrap CIs and McNemar tests\nEach row changes exactly one component. Differences whose CI contains 0 are not evidence of an effect at this sample size.'),
code('''c = r3["contrasts"]
c[c.scope == "all"][["contrast","n","delta_correct","correct_lo","correct_hi","delta_incorrect","incorrect_lo","incorrect_hi","mcnemar_p"]]'''),
code('show("marginal_contributions.png", "stage3")'),
md('## 3. Factuality vs. cost (the trade-off asked for in the proposal)'),
code('show("pareto.png", "stage3"); show("cost.png", "stage3")'),
md('## 4. Per-dataset results, claim-level factuality, flagging statistics'),
code('''d = r3["summary"]; d[[c for c in ["dataset","system","correct","incorrect","not_attempted","f_score","claim_support_rate","free_correct","abstention_rate"] if c in d]]'''),
code('show("claim_rates.png", "stage3"); show("flag_stats.png", "stage3"); show("threshold_retention.png", "stage3")'),
md('### TruthfulQA split by whether the KB holds the question\'s Wikipedia source'),
code('''pd.read_csv(Path(cfg.project.results_dir) / "stage3" / "summary_truthfulqa_by_wiki_coverage.csv")[["tqa_has_wiki","system","n","abstention_rate","correct","incorrect","free_correct"]]'''),
md('## 5. Qualitative examples (initial vs. corrected, with citations) - including failures\n'),
code('print((Path(cfg.project.results_dir) / "stage3" / "examples.md").read_text()[:6000])'),
md('## 6. Try your own question\nSame pipeline as the Streamlit demo (`streamlit run app.py`).'),
code('''question = "What happens to you if you eat watermelon seeds?"
from halluguard.pipeline import FactualityPipeline, QuestionState
from halluguard.factory import load_calibration
pipe = FactualityPipeline(cfg, res.llm, res.retriever, res.nli, load_calibration(cfg))
outs = pipe.run_all(QuestionState("demo", question), ["A", "B", "F"])
for sid, o in outs.items():
    print(f"--- {sid} ({o.initial_source}) | flagged {o.n_flagged}, removed {o.n_removed}, rewritten {o.n_rewritten} | {o.cost['calls']:.0f} calls, {o.cost['total_tokens']:.0f} tokens")
    print(o.final_answer_cited)'''),
md('''## Reporting checklist
* State sample sizes (`meta.json`), the judge model and whether judge == generator (`judge_equals_generator`), and that the KB is an oracle-in-pool setting.
* Do not report a run whose `meta.json` says `"mock": true`.
* Interpret contrasts through their CIs; with n~80 questions only large effects are detectable - raise `stage3.n_per_dataset` if the budget allows.'''),
]

def write(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.validate(nb)
    nbf.write(nb, ROOT / name)
    print("wrote", name, len(cells), "cells")

write("01_baselines_and_retrieval.ipynb", nb1)
write("02_hallucination_detection.ipynb", nb2)
write("03_correction_and_ablation.ipynb", nb3)
