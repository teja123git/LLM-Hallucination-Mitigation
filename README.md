# LLM-Hallucination-Mitigation

**Improving factuality of LLMs through Retrieval-Augmented Generation, claim-level verification and evidence-guided correction** (DA624 - NLP with LLMs).

Question: *how much do hallucination detection and self-verification improve factuality over a plain LLM and a standard RAG system, and what do they cost in latency and tokens?*
A controlled, paired ablation of eight systems answers it (which component helps, by how much, at what price).

```
question -> [1 generate] -> [2 atomic claims] -> [3 detect: NLI | LLM verifier | SelfCheck | CoVe | hybrid]
                                                     |  flagged claims
                                    [4 retrieve evidence per claim (BM25 + dense + RRF)]
                                                     v
                              [5 correct: remove / RARR-style rewrite, verify-then-accept] -> [6 answer with [n] citations]
```

| system | A | B | C | D | E | F | G | F-CoVe |
|---|---|---|---|---|---|---|---|---|
| | LLM only | +RAG | +self-consistency | +CoVe verification | +RAG +verification | E + correction (RARR) | F with hybrid detector | E + CoVe correction |

## Quick start
```bash
git clone https://github.com/teja123git/LLM-Hallucination-Mitigation.git && cd LLM-Hallucination-Mitigation
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .         # (pip install -e . is optional; scripts/notebooks work without it)
cp .env.example .env                                        # then put your free key from https://console.groq.com in .env
python scripts/check_setup.py                               # API key, models, datasets, token budget - run this first
python scripts/run_all.py --config configs/quick.yaml       # small end-to-end run (validate your setup)
python scripts/run_all.py                                   # full run (see "Budget" - resumable over several days on a free key)
```
Outputs land in `results/stage1|2|3/` and a consolidated `results/REPORT.md`. Or use the notebooks `01_`, `02_`, `03_` (same code, narrated). Google Colab: clone, `pip install -r requirements.txt`, add `GROQ_API_KEY` to the environment, open a notebook.

**Offline smoke test (no key, no internet, fake models, toy data):** `python scripts/run_all.py --mock` -> writes to `results_mock/`; every figure is watermarked **MOCK DATA**. It checks that the code works; its numbers mean nothing and must never be reported.

## Project structure
```
01_baselines_and_retrieval.ipynb   Stage 1: data, KB, retrieval metrics, Baselines A/B
02_hallucination_detection.ipynb   Stage 2: claims, detectors, benchmarks (HaluEval/RAGTruth/FEVER), calibration, ablations
03_correction_and_ablation.ipynb   Stage 3: correction, 8-system ablation, judge, statistics, cost, examples
app.py                             Streamlit demo: initial vs. verified/corrected answer, claim table, evidence, cost
src/halluguard/                    llm.py (cache, limiter) retrieval.py kb.py data.py claims.py nli.py detection.py correction.py
                                   pipeline.py (systems A-G) judge.py evaluation.py metrics.py plotting.py budget.py report.py
                                   experiments/{stage1,stage2,stage3,common}.py
scripts/                           run_stage1|2|3.py run_all.py check_setup.py prepare_data.py make_toy_data.py dev/build_notebooks.py
configs/                           default.yaml  quick.yaml  mock.yaml
tests/                             45 tests (pytest) - infrastructure, retrieval/KB leakage, metrics, pipeline semantics, e2e mock run
docs/                              DATA.md  METHODS.md  CHANGES_FROM_STAGE1.md  course/ (proposal + Stage 1 report)
data/toy/                          tiny offline datasets for --mock/tests      results/   generated outputs
```

## Running pieces
```bash
python scripts/run_stage1.py [--n 100]                        # baselines + retrieval evaluation
python scripts/run_stage2.py [--n-dev 40 --n-test 80 --skip-fever --skip-ragtruth]
python scripts/run_stage3.py [--n 40] [--systems A,B,E,F] [--no-judge]
python scripts/run_all.py --budget-only                       # print the rough token budget and exit
streamlit run app.py                                          # interactive demo (HALLUGUARD_MOCK=1 for an offline UI preview)
python -m pytest                                              # tests (no network or models needed)
```
Override any setting: `--set llm.model=llama-3.3-70b-versatile --set retrieval.top_k=3`. All knobs are in `configs/default.yaml`.
Run Stage 2 before Stage 3: it writes `results/stage2/calibration.json` (detector thresholds / hybrid weights); Stage 3 warns if it is missing.

## Budget on a free Groq key (rough estimates at default sizes - `run_all.py --budget-only` prints yours)
| stage | generator tokens | ~days at 500K tokens/day | notes |
|---|---|---|---|
| 1 | ~0.2M | 0.4 | 100 questions per dataset |
| 2 | ~0.9M | ~2 | HaluEval 40 dev + 80 test questions, 90 FEVER, 90 RAGTruth |
| 3 | ~0.5M + judge ~0.35M | ~1 + ~3.5 | 40 questions per dataset; the 70B judge has only 100K tokens/day |

Every finished LLM call is cached (`data/cache/llm_cache.sqlite`), so stopping, hitting the daily quota (the script exits with code 3 and a message) or re-running costs nothing - it resumes. `configs/quick.yaml` needs roughly a day's quota in total.
**Judge options:** (a) default - independent 70B judge, slowest on free tier; (b) `--no-judge` - judge-free metrics only; (c) `--set llm.judge_model=llama-3.1-8b-instant` - fast but self-judging (recorded as `judge_equals_generator` in `meta.json`; mention it when reporting); (d) a paid Groq tier (a few cents for the whole project). Limits change - check your console; adjust `llm.rpm/tpm/limits` in the config.

## What you get
* **Stage 1** - retrieval Recall@k/MRR (bm25 vs. dense vs. hybrid), context coverage, A/B quality (EM/F1/contains, BERTScore, ROUGE-L), abstention, tokens (prompt/completion), calls, latency.
* **Stage 2** - Accuracy/Precision/Recall/F1/AUROC/AUPRC for 6 detectors on HaluEval test, RAGTruth, FEVER (+ 3-way confusion); gold-vs-retrieved evidence; bm25/dense/hybrid vs. detection; SelfCheck sample-count curve; ROC/PR curves; error analysis (`error_analysis.md`).
* **Stage 3** - judge grades, hallucination (`incorrect`) rate, abstention-aware F-score, FActScore-style claim precision, citation precision, **paired contrasts** (one component at a time, bootstrap CI + McNemar), cost by stage group, factuality-vs-cost Pareto plots, qualitative before/after examples including failures.

Map to the proposal: objectives 1-6 and deliverables (pipeline, >=4 system variants, tables/plots, ablation + error analysis, interactive demo) are implemented; "a small manually checked subset" is *not* automated - audit ~30 judge decisions by hand from `results/stage3/judgments.json` / `examples.md` before quoting claim-level numbers.

## Important design notes (details: docs/)
* **Leak-free knowledge base** - only source documents (HaluEval paragraphs, Wikipedia pages cited by TruthfulQA); never answer lists. It is an oracle-in-pool setting (gold doc hidden among ~10k distractors) - easier than open-web RAG.
* **No fabricated answers** - API failures raise; items are skipped and counted (`n_failed` in `meta.json`).
* **Abstention-aware grading** - a system cannot look good by refusing everything.
* **Paired design + shared stages** - all systems answer identical questions; costs are still attributed per system.
* Prompts: `src/halluguard/prompts.py`. Models: generator `llama-3.1-8b-instant`, judge `llama-3.3-70b-versatile`, embedder `all-MiniLM-L6-v2`, NLI `cross-encoder/nli-deberta-v3-base` (all configurable; Groq model availability changes over time).

## Troubleshooting
* `GROQ_API_KEY is not set` -> `.env` (see `.env.example`). `Daily quota reached` -> re-run later (progress is cached), use `--no-judge`, or a paid tier.
* DeBERTa tokenizer error -> `pip install sentencepiece protobuf tiktoken`. First run downloads ~2 GB of models (MiniLM, DeBERTa NLI, roberta-large for BERTScore; BERTScore is skipped with a warning if unavailable).
* FEVER "unavailable"/"cannot parse" -> `python scripts/prepare_data.py --check`; or `--skip-fever`.
* Behind a firewall: Wikipedia needs internet once (`kb.tqa_wikipedia: false` disables it; see docs/DATA.md).
* Very first Stage-1 run embeds ~10k passages (a few minutes on CPU); embeddings are cached in `data/kb/`.

## Status / honesty notes
The pipeline and all scripts were exercised end-to-end **offline with fake models** (tests, notebooks, CLI, demo). Components that need the real services - the live Groq client, MiniLM/DeBERTa/BERTScore, Wikipedia fetching and the FEVER loader - were written to documented APIs but could not be run where this was developed; `scripts/check_setup.py` exercises each of them on your machine. **No real experimental results are included**; `results/` is empty until you run the stages.

Academic research project (DA624). Authors: Vamaxi Maheshwari, Venkat Sai Teja Kalluri, Chodabattula Moni Praneeth.
