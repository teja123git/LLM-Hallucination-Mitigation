"""Interactive demo: initial answer vs. verified / corrected answer, claim by claim, with evidence and cost.

    streamlit run app.py                      # live: needs GROQ_API_KEY in .env and the datasets (first start builds the KB)
    HALLUGUARD_MOCK=1 streamlit run app.py    # offline UI preview with fake models (answers are meaningless)
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from halluguard.config import load_config  # noqa: E402
from halluguard.experiments.common import load_resources  # noqa: E402
from halluguard.factory import is_mock, load_calibration  # noqa: E402
from halluguard.llm import LLMError  # noqa: E402
from halluguard.pipeline import SYSTEMS, FactualityPipeline, QuestionState  # noqa: E402

MOCK = os.environ.get("HALLUGUARD_MOCK") == "1"
EXAMPLES = [
    "What happens to you if you eat watermelon seeds?",
    "Which magazine was started first, Arthur's Magazine or First for Women?",
    "Can you see the Great Wall of China from space?",
    "What is the spiciest part of a chili pepper?",
]
ICON = {"supported": "✅", "unsupported": "⚠️", "contradicted": "❌"}

st.set_page_config(page_title="HalluGuard - verify & correct LLM answers", layout="wide")


@st.cache_resource(show_spinner="Loading data, building the knowledge base and loading models (first start takes a few minutes)...")
def get_pipeline():
    cfg = load_config(mock=MOCK)
    res = load_resources(cfg)
    return cfg, FactualityPipeline(cfg, res.llm, res.retriever, res.nli, load_calibration(cfg))


def claims_table(claims):
    rows = [{"": ICON.get(c["label"], ""), "claim": c["text"], "verdict": c["label"], "hallucination score": round(c["scores"]["head"], 3),
             "action": c["action"], "revised claim": c["revised"] or ""} for c in claims]
    return pd.DataFrame(rows)


st.title("🛡️ HalluGuard")
st.caption("Generate → extract claims → detect unsupported claims → retrieve evidence → correct → answer with citations")
if MOCK:
    st.error("MOCK MODE - fake LLM, fake NLI and toy data. Answers are meaningless; this only previews the interface.")

try:
    cfg, pipe = get_pipeline()
except Exception as e:  # missing API key etc. -> explain instead of a stack trace
    st.error(f"Could not start the pipeline: {e}")
    st.info("Copy `.env.example` to `.env`, add your free Groq key (https://console.groq.com) and restart. "
            "For an offline preview run `HALLUGUARD_MOCK=1 streamlit run app.py`.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    labels = {k: v["label"] for k, v in SYSTEMS.items()}
    chosen = st.multiselect("Systems to compare", list(SYSTEMS), default=["A", "B", "F"], format_func=lambda k: labels[k])
    pipe.rc.top_k = st.slider("Passages for RAG generation (top-k)", 1, 8, int(pipe.rc.top_k))
    pipe.rc.evidence_k = st.slider("Evidence passages per claim", 1, 6, int(pipe.rc.evidence_k))
    st.markdown(f"**Generator:** `{cfg.llm.model}`  \n**NLI:** `{cfg.nli.model}`  \n**Detector thresholds calibrated:** {'yes' if pipe.cal.calibrated else 'no (defaults - run Stage 2)'}")
    st.caption("Cached LLM calls are re-used, so asking the same question twice is free.")

if "q" not in st.session_state:
    st.session_state["q"] = EXAMPLES[0]
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex):
        st.session_state["q"] = ex
question = st.text_input("Ask a factual question", key="q")

if st.button("Run", type="primary") and question.strip() and chosen:
    st_ = QuestionState("demo", question.strip())
    try:
        with st.spinner("Running the pipeline..."):
            outs = pipe.run_all(st_, chosen)
    except LLMError as e:
        st.error(f"LLM call failed: {e}")
        st.stop()
    st.session_state["outs"] = outs

outs = st.session_state.get("outs")
if outs:
    st.subheader("Answers")
    columns = st.columns(len(outs))
    for col, (sid, o) in zip(columns, outs.items()):
        with col:
            st.markdown(f"##### {SYSTEMS[sid]['label']}")
            if o.abstained:
                st.warning(o.final_answer)
            else:
                st.success(o.final_answer_cited)
            if o.initial_answer and o.initial_answer != o.final_answer:
                with st.expander("Initial answer (before verification)"):
                    st.write(o.initial_answer)
            if o.sources:
                with st.expander(f"Sources ({len(o.sources)})"):
                    for s in o.sources:
                        st.markdown(f"**[{s['n']}]** {('*' + s['title'] + '* — ') if s.get('title') else ''}{s['text'][:400]}")
            c = o.cost
            st.caption(f"{c['calls']:.0f} LLM calls · {c['total_tokens']:.0f} tokens · {c['latency']:.2f}s"
                       + (f" · flagged {o.n_flagged}, removed {o.n_removed}, rewritten {o.n_rewritten}" if o.claims else ""))

    detail = [(sid, o) for sid, o in outs.items() if o.claims]
    if detail:
        st.subheader("Claim-level verification")
        tabs = st.tabs([SYSTEMS[sid]["label"] for sid, _ in detail])
        for tab, (sid, o) in zip(tabs, detail):
            with tab:
                st.caption(f"Detector: **{SYSTEMS[sid]['detector']}** · verified answer: **{o.initial_source}**")
                st.dataframe(claims_table(o.claims), hide_index=True)
                for c in o.claims:
                    with st.expander(f"{ICON.get(c['label'], '')} {c['text']}"):
                        d = c["detail"]
                        if d.get("nli"):
                            st.write(f"NLI vs. evidence → entailment **{d['nli']['entail']:.2f}**, contradiction **{d['nli']['contra']:.2f}**")
                        if d.get("llm"):
                            st.write(f"LLM verifier → **{d['llm']['verdict']}** — {d['llm'].get('reason', '')}")
                        if d.get("cove"):
                            st.write(f"CoVe check → **{d['cove']['verdict']}** (Q: {d['cove']['question']} → A: {d['cove']['answer']})")
                        if c["scores"].get("s_sc") is not None:
                            st.write(f"Self-consistency (SelfCheck-NLI) score: **{c['scores']['s_sc']:.2f}**")
                        for h in c["evidence"]:
                            st.markdown(f"> {('**' + h['title'] + '**: ') if h.get('title') else ''}{h['text'][:500]}")
