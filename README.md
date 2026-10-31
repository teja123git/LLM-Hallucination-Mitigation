# LLM-Hallucination-Mitigation

**Improving Factuality of LLMs through Retrieval-Augmented Generation, Claim-Level Verification and Evidence-Guided Correction**

**Course:** DA624 – NLP with LLMs

---

## Project Overview

Large Language Models (LLMs) can generate fluent and convincing responses that contain incorrect, unsupported, or fabricated information. This problem, commonly known as hallucination, limits their reliability in applications where factual accuracy is important.

This project investigates how Retrieval-Augmented Generation (RAG), claim-level hallucination detection, self-verification, and evidence-guided correction can improve the factuality of LLM-generated answers.

**Research Question:**

*How much do hallucination detection and self-verification improve factuality over a plain LLM and a standard RAG system, and what are the associated costs in latency and tokens?*

We conduct a controlled, paired ablation study of eight system configurations to investigate which components improve factuality, by how much, and at what computational cost.

## Methodology

The proposed system follows a multi-stage pipeline:

```text
Question
   |
   v
[1. Initial Answer Generation]
   |
   v
[2. Atomic Claim Extraction]
   |
   v
[3. Hallucination Detection]
   |  NLI | LLM Verifier | SelfCheck | CoVe | Hybrid
   |
   v
[4. Evidence Retrieval]
   |  BM25 + Dense Retrieval + Reciprocal Rank Fusion
   |
   v
[5. Evidence-Guided Correction]
   |  KEEP / REWRITE / REMOVE
   |  Verify rewritten claims before acceptance
   |
   v
[6. Final Answer with Supporting Citations]
```

The system combines retrieval, claim-level verification, and evidence-guided correction to reduce unsupported or incorrect statements in generated responses.

## System Configurations

The project evaluates eight system configurations:

| System | Configuration | Description |
|---|---|---|
| A | LLM Only | Direct response from the language model |
| B | Standard RAG | LLM with retrieved context |
| C | Self-Consistency | LLM with self-consistency-based verification |
| D | CoVe Verification | LLM with Chain-of-Verification |
| E | RAG + Verification | Retrieval combined with claim-level verification |
| F | RAG + Correction | System E with evidence-guided correction using a RARR-style approach |
| G | Hybrid Detection | System F with a hybrid hallucination detector |
| F-CoVe | CoVe Correction | System E with CoVe-based correction |

These configurations allow comparisons between a plain LLM, standard RAG, verification-based approaches, and correction-based approaches.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/teja123git/LLM-Hallucination-Mitigation.git
cd LLM-Hallucination-Mitigation
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

Installing the package in editable mode is optional.

### 4. Configure the API Key

Create a `.env` file using `.env.example`.

**Windows:**

```powershell
Copy-Item .env.example .env
notepad .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key
```

Get a free API key from:

https://console.groq.com/keys

Do not upload your `.env` file or expose your API key publicly.

### 5. Check the Setup

```bash
python scripts/check_setup.py
```

This checks the API key, models, datasets, and approximate token budget.

### 6. Run a Quick Experiment

```bash
python scripts/run_all.py --config configs/quick.yaml
```

### 7. Run the Complete Experiment

```bash
python scripts/run_all.py
```

The complete experiment may take several days depending on API limits and model usage. Completed LLM calls are cached, allowing experiments to resume without repeating previous calls.

## Project Structure

```text
LLM-Hallucination-Mitigation/
│
├── 01_baselines_and_retrieval.ipynb
├── 02_hallucination_detection.ipynb
├── 03_correction_and_ablation.ipynb
│
├── app.py
│
├── src/
│   └── halluguard/
│       ├── llm.py
│       ├── retrieval.py
│       ├── kb.py
│       ├── data.py
│       ├── claims.py
│       ├── nli.py
│       ├── detection.py
│       ├── correction.py
│       ├── pipeline.py
│       ├── judge.py
│       ├── evaluation.py
│       ├── metrics.py
│       ├── plotting.py
│       ├── budget.py
│       ├── report.py
│       └── experiments/
│           ├── stage1.py
│           ├── stage2.py
│           ├── stage3.py
│           └── common.py
│
├── scripts/
│   ├── run_stage1.py
│   ├── run_stage2.py
│   ├── run_stage3.py
│   ├── run_all.py
│   ├── check_setup.py
│   ├── prepare_data.py
│   ├── make_toy_data.py
│   └── dev/
│       └── build_notebooks.py
│
├── configs/
│   ├── default.yaml
│   ├── quick.yaml
│   └── mock.yaml
│
├── tests/
│
├── docs/
│   ├── DATA.md
│   ├── METHODS.md
│   ├── CHANGES_FROM_STAGE1.md
│   └── course/
│
├── data/
│   └── toy/
│
├── results/
│
├── requirements.txt
├── .env.example
└── README.md
```

## Running Individual Stages

The project is organized into three main stages, with a separate Jupyter notebook for each stage.

### Stage 1: Baselines and Retrieval

**Notebook:** `01_baselines_and_retrieval.ipynb`

This stage prepares the datasets and knowledge base, evaluates retrieval performance, and establishes the baseline results for systems A and B.

Run:

```bash
python scripts/run_stage1.py --n 100
```

Evaluation includes:

- BM25, dense, and hybrid retrieval.
- Recall@k and Mean Reciprocal Rank (MRR).
- Context coverage.
- Exact Match and F1.
- BERTScore and ROUGE-L.
- Abstention.
- Prompt and completion tokens.
- API calls and latency.

### Stage 2: Hallucination Detection

**Notebook:** `02_hallucination_detection.ipynb`

This stage extracts atomic claims and evaluates hallucination detection methods using HaluEval, RAGTruth, and FEVER.

Run:

```bash
python scripts/run_stage2.py --n-dev 40 --n-test 80
```

Optional dataset exclusions:

```bash
python scripts/run_stage2.py --skip-fever --skip-ragtruth
```

Evaluation includes:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- AUROC and AUPRC.
- Three-way confusion analysis.
- Gold versus retrieved evidence.
- BM25, dense, and hybrid retrieval comparisons.
- Self-consistency sample-count analysis.
- ROC and Precision-Recall curves.
- Detector error analysis.

### Stage 3: Correction and Ablation

**Notebook:** `03_correction_and_ablation.ipynb`

This stage evaluates evidence-guided correction and compares the different system configurations.

Run:

```bash
python scripts/run_stage3.py --n 40
```

To run selected systems:

```bash
python scripts/run_stage3.py --systems A,B,E,F
```

To disable the independent LLM judge:

```bash
python scripts/run_stage3.py --no-judge
```

Evaluation includes:

- Judge-based answer evaluation.
- Hallucination and incorrect-claim rates.
- Abstention-aware F-score.
- FActScore-style claim precision.
- Citation precision.
- Paired system comparisons.
- Bootstrap confidence intervals.
- McNemar's test.
- Token usage and latency analysis.
- Factuality-versus-cost comparisons.
- Before-and-after correction examples.

**Important:** Run Stage 2 before Stage 3. Stage 2 generates `results/stage2/calibration.json`, which contains detector calibration information used by Stage 3.

## Running the Complete Pipeline

For a smaller validation run:

```bash
python scripts/run_all.py --config configs/quick.yaml
```

For the complete experiment:

```bash
python scripts/run_all.py
```

To run individual stages, use:

```bash
python scripts/run_stage1.py
python scripts/run_stage2.py
python scripts/run_stage3.py
```

## Offline Smoke Test

The project includes a mock mode for testing the software pipeline without an API key, internet access, or actual language model calls.

Run:

```bash
python scripts/run_all.py --mock
```

Mock outputs are saved in:

```text
results_mock/
```

Mock mode uses toy datasets and simulated model responses. It is intended only for testing software functionality.

**Important:** Mock results must never be presented as actual experimental results. All mock-generated figures and outputs are marked as MOCK DATA.

An offline preview of the Streamlit interface can also be launched using:

```bash
HALLUGUARD_MOCK=1 streamlit run app.py
```

## Streamlit Application

The project includes a Streamlit application for demonstrating the hallucination mitigation pipeline.

Run:

```bash
streamlit run app.py
```

The application demonstrates:

- Initial LLM-generated answers.
- Claim extraction.
- Claim-level verification.
- Retrieved supporting evidence.
- Unsupported claim identification.
- Evidence-guided correction.
- Final verified answers with citations.
- Token and cost information.

## Evaluation Metrics

The project uses multiple metrics to evaluate retrieval quality, hallucination detection, answer factuality, and computational cost.

| Evaluation Area | Metrics |
|---|---|
| Retrieval | Recall@k, MRR, context coverage |
| Answer Quality | Exact Match, F1, ROUGE-L, BERTScore |
| Hallucination Detection | Accuracy, Precision, Recall, F1, AUROC, AUPRC |
| Claim-Level Evaluation | FActScore-style claim precision, incorrect-claim rate |
| Citation Evaluation | Citation precision |
| Abstention | Abstention-aware F-score |
| Computational Cost | Prompt tokens, completion tokens, API calls, latency |
| Statistical Analysis | Bootstrap confidence intervals, McNemar's test |

The evaluation compares both factuality improvements and the additional computational cost introduced by verification and correction.

## Datasets

The project uses the following datasets and knowledge sources:

- **HaluEval:** Used for hallucination detection evaluation.
- **RAGTruth:** Used for evaluating hallucinations in retrieval-augmented responses.
- **FEVER:** Used for claim verification against evidence.
- **TruthfulQA:** Used for evaluating factual question answering and retrieval.

The knowledge base is constructed using source documents rather than answer lists. For TruthfulQA, Wikipedia pages cited by the dataset are used as source material.

The experimental setup follows an oracle-in-pool retrieval setting, where relevant gold documents are included among a larger collection of distractor documents. This makes retrieval evaluation controlled but easier than unrestricted open-web retrieval.

For additional information, refer to:

- `docs/DATA.md`
- `docs/METHODS.md`

## Budget and API Usage

The experiments use Groq API calls for language generation and verification. Token consumption depends on the selected configuration, dataset sizes, and judge settings.

Approximate generator-token requirements at the default experiment sizes:

| Stage | Estimated Generator Tokens | Approximate Duration at 500K Tokens/Day |
|---|---:|---:|
| Stage 1 | 0.2M | 0.4 days |
| Stage 2 | 0.9M | Approximately 2 days |
| Stage 3 | 0.5M + approximately 0.35M for judging | Approximately 1 day + 3.5 days for judging |

These are rough estimates and may change with API limits, model selection, and experiment configuration.

To view the estimated token budget:

```bash
python scripts/run_all.py --budget-only
```

### Judge Options

Three configurations are supported:

1. **Independent judge:** Uses the configured 70B judge model. This is the default configuration.
2. **Judge-free evaluation:** Run with `--no-judge`.
3. **Faster self-judging configuration:** Change the judge model to `llama-3.1-8b-instant`.

When the generator and judge use the same model, the experiment metadata records `judge_equals_generator`. This distinction should be mentioned when reporting the result.

### Caching and Resuming

Completed LLM calls are cached in:

```text
data/cache/llm_cache.sqlite
```

This allows experiments to resume without repeating completed API calls.

If the daily API quota is reached, the script exits with a message. The experiment can be continued later using the same configuration.

API limits and model availability may change. Check your Groq console and adjust the relevant settings in `configs/default.yaml` when required.

## Configuration

Experiment parameters can be modified through:

```text
configs/default.yaml
configs/quick.yaml
configs/mock.yaml
```

For example, to change the language model or retrieval configuration:

```bash
python scripts/run_all.py \
  --set llm.model=llama-3.3-70b-versatile \
  --set retrieval.top_k=3
```

The configuration files contain settings for model selection, retrieval, detector thresholds, token limits, API usage, and experiment parameters.

## Testing

The project includes an automated pytest suite covering:

- Infrastructure and configuration.
- Retrieval and knowledge-base functionality.
- Knowledge-base leakage checks.
- Evaluation metrics.
- Pipeline semantics.
- End-to-end mock execution.

Run the tests using:

```bash
python -m pytest
```

The tests are designed to run without external API calls or language models.

## Experimental Limitations

The following limitations should be considered when interpreting the results:

1. **Oracle-in-pool retrieval:** Relevant documents are included in the knowledge base, making retrieval easier than unrestricted open-web retrieval.
2. **Limited evaluation size:** Some experiments use relatively small subsets of the available datasets.
3. **Model dependency:** Results may vary depending on the generator, verifier, and judge models.
4. **API limitations:** Free-tier token limits and rate limits can affect experiment execution.
5. **Judge reliability:** LLM-based evaluation may introduce its own biases and errors.
6. **Manual validation:** A small manually checked subset is not automated. Approximately 30 judge decisions should be manually audited using `results/stage3/judgments.json` and `examples.md` before reporting claim-level conclusions.
7. **Dataset coverage:** TruthfulQA retrieval uses a subset of the available Wikipedia source pages.

These limitations are important when generalizing the results beyond the experimental setup.

## Output Files

Generated experiment results are stored under:

```text
results/
├── stage1/
├── stage2/
├── stage3/
└── REPORT.md
```

The outputs include:

- Retrieval evaluation results.
- Detector performance metrics.
- Calibration information.
- System comparison results.
- Statistical analysis.
- Token and latency measurements.
- Generated plots.
- Qualitative examples.
- Consolidated experiment report.

The exact files generated depend on the experiment configuration and the stages completed.

## Reproducibility

To reproduce the experiments:

1. Clone the repository.
2. Install the required dependencies.
3. Configure the Groq API key.
4. Run `scripts/check_setup.py`.
5. Execute Stage 1 to prepare the knowledge base and evaluate baselines.
6. Execute Stage 2 to calibrate and evaluate the detectors.
7. Execute Stage 3 to evaluate correction and system ablations.
8. Inspect the generated outputs in the `results/` directory.

The notebooks provide a narrated version of the experimental stages, while the Python scripts support reproducible execution from the command line.

For a quick validation:

```bash
python scripts/run_all.py --config configs/quick.yaml
```

For the complete experiment:

```bash
python scripts/run_all.py
```

## Documentation

Additional project documentation is available in the `docs/` directory:

- `docs/DATA.md` – Dataset preparation and knowledge-base construction.
- `docs/METHODS.md` – Methodology and experimental details.
- `docs/CHANGES_FROM_STAGE1.md` – Changes made after the initial project stage.
- `docs/course/` – Initial project proposal and Stage 1 report.

---

**Course Project:** DA624 – NLP with LLMs

**Project:** LLM Hallucination Mitigation

**Repository:** [LLM-Hallucination-Mitigation](https://github.com/teja123git/LLM-Hallucination-Mitigation)
