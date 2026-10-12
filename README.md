# LLM-Hallucination-Mitigation

Improving factuality of Large Language Models through Retrieval-Augmented Generation and Self-Verification. This project investigates whether combining RAG, self-consistency verification, and evidence-guided correction can reliably reduce hallucinations across standard benchmarks.

## Project Structure

```
├── 01_baselines_and_retrieval.ipynb   Stage 1 - Data, Retrieval and Baselines
├── 02_hallucination_detection.ipynb   Stage 2 - Claim Extraction and Verification
├── 03_correction_and_ablation.ipynb   Stage 3 - Correction and Ablation Study
├── data/                              Auto-downloaded datasets (git-ignored)
├── results/                           Evaluation outputs and plots
├── requirements.txt
└── .env                               API key (git-ignored)
```

## Setup

```bash
git clone https://github.com/<your-username>/LLM-Hallucination-Mitigation.git
cd LLM-Hallucination-Mitigation
pip install -r requirements.txt
```

Create a `.env` file in the project root with your free Groq API key:

```
GROQ_API_KEY=your_key_here
```

Get a free key at https://console.groq.com

## Approach

**Stage 1 - Baselines and Retrieval**
- Load TruthfulQA and HaluEval benchmarks
- Build hybrid retrieval engine (Dense + BM25 with Reciprocal Rank Fusion)
- Run Baseline A (LLM only) and Baseline B (RAG-augmented)
- Evaluate with BERTScore, ROUGE-L, Recall@k, MRR, latency

**Stage 2 - Hallucination Detection**
- Atomic claim extraction from generated answers
- Multi-strategy verification: SelfCheckGPT, NLI cross-encoder, LLM-as-Judge

**Stage 3 - Correction and Ablation**
- Evidence-guided claim correction with inline citations
- Six-system controlled ablation comparison (A through F)

## Evaluation Metrics

| Category       | Metrics                                        |
|----------------|------------------------------------------------|
| Factuality     | FActScore, Claim Support Rate, Hallucination Rate |
| Answer Quality | BERTScore F1, ROUGE-L                          |
| Retrieval      | Recall@k, MRR, Context Relevance               |
| Efficiency     | Latency, Token Cost, API Calls                  |

## Tech Stack

| Component        | Tool                                |
|------------------|-------------------------------------|
| LLM              | Groq API (free) - llama-3.1-8b-instant |
| Embeddings       | all-MiniLM-L6-v2 (local)           |
| Vector Store     | FAISS (local)                       |
| Sparse Retrieval | BM25 via rank-bm25                  |

## License

Academic research project (DA624).
