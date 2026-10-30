"""All prompt templates in one place (so they are easy to audit, tweak and cite in the report).

Conventions: tasks that need structured output ask for a single JSON *object* (required by Groq's JSON mode)
and are parsed with ``utils.parse_json``; every parser has a deterministic fallback.
"""
from __future__ import annotations

from typing import List, Sequence

NO_INFO = "The context does not contain enough information."

# ------------------------------------------------------------------ generation (Baselines A / B)
DIRECT = """Answer the following question accurately, in one or two complete sentences.

Question: {question}
Answer:"""

RAG = """Answer the question using ONLY the context below. Write one or two complete sentences. If the context does not contain enough information to answer, reply exactly: "{no_info}"

Context:
{context}

Question: {question}
Answer:"""


def direct(question: str) -> str:
    return DIRECT.format(question=question)


def rag(question: str, passages: Sequence[str]) -> str:
    context = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return RAG.format(context=context, question=question, no_info=NO_INFO)


# ------------------------------------------------------------------ claim extraction
CLAIMS = """Decompose the answer below into a list of atomic factual claims.

Rules:
- Each claim is ONE short, self-contained declarative sentence stating a single fact.
- Resolve pronouns and short answers using the question, so each claim can be understood without seeing the question. Example: question "Which magazine was started first, Arthur's Magazine or First for Women?" with answer "Arthur's Magazine" gives the claim "Arthur's Magazine was started before First for Women."
- Do not add facts that are not stated in the answer. Skip opinions, hedges and meta-statements.
- If the answer declines to answer or says it does not know, return an empty list.
- Return at most {max_claims} claims.

Question: {question}
Answer: {answer}

Respond with JSON only, in exactly this form: {{"claims": ["claim one", "claim two"]}}"""


def claims(question: str, answer: str, max_claims: int = 5) -> str:
    return CLAIMS.format(question=question, answer=answer, max_claims=max_claims)


# ------------------------------------------------------------------ LLM-based verification against evidence
VERIFY = """You are a careful fact-checker. Decide whether the CLAIM is supported by the EVIDENCE passages. Use only the evidence and ignore your own knowledge.

Evidence:
{evidence}

Claim: {claim}

Verdicts:
- SUPPORTED: the evidence clearly states or directly implies the claim.
- CONTRADICTED: the evidence states something that conflicts with the claim.
- NOT_ENOUGH_INFO: the evidence neither supports nor contradicts the claim.

Respond with JSON only, in exactly this form: {{"verdict": "SUPPORTED", "evidence_ids": [1], "reason": "one short sentence"}}"""


def verify(claim: str, evidence: Sequence[str]) -> str:
    ev = "\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence)) or "(no evidence found)"
    return VERIFY.format(evidence=ev, claim=claim)


# ------------------------------------------------------------------ Chain-of-Verification (parametric, no retrieval)
COVE_PLAN = """Below is a question and numbered claims taken from an answer. For each claim write ONE short verification question that checks the key fact of the claim WITHOUT stating the claim's assertion in the question, so that answering it is an independent check.

Question: {question}
Claims:
{claims}

Respond with JSON only, in exactly this form: {{"questions": [{{"claim_id": 1, "question": "..."}}]}}"""

COVE_ANSWER = """Answer each question below independently, using your own knowledge. Give a short factual answer (one sentence). If you do not know, answer "unknown".

Questions:
{questions}

Respond with JSON only, in exactly this form: {{"answers": [{{"id": 1, "answer": "..."}}]}}"""

COVE_ANSWER_EVIDENCE = """Answer each question below using ONLY the evidence listed under it. Give a short factual answer (one sentence). If the evidence does not answer the question, answer "unknown".

{blocks}

Respond with JSON only, in exactly this form: {{"answers": [{{"id": 1, "answer": "..."}}]}}"""

COVE_JUDGE = """For each item compare the CLAIM with the independently produced verification answer and decide whether the claim is consistent with it.
- CONSISTENT: the verification answer agrees with the claim.
- INCONSISTENT: the verification answer conflicts with the claim.
- UNCERTAIN: the verification answer is "unknown" or does not address the claim.

Items:
{items}

Respond with JSON only, in exactly this form: {{"verdicts": [{{"id": 1, "verdict": "CONSISTENT"}}]}}"""

COVE_REVISE = """Question: {question}

Draft answer: {answer}

Verification questions with evidence-based answers:
{qa}

Write the final answer. Keep only facts that agree with the verification answers, correct facts that conflict with them, and drop anything that could not be verified. Use one or two complete sentences. If nothing can be answered reliably, reply exactly: "{abstain}"
Final answer:"""


def cove_plan(question: str, claims: Sequence[str]) -> str:
    return COVE_PLAN.format(question=question, claims="\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims)))


def cove_answer(questions: Sequence[str]) -> str:
    return COVE_ANSWER.format(questions="\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions)))


def cove_answer_evidence(questions: Sequence[str], evidence: Sequence[Sequence[str]]) -> str:
    blocks = []
    for i, (q, ev) in enumerate(zip(questions, evidence)):
        ev_txt = "\n".join(f"   - {e}" for e in ev) or "   - (no evidence found)"
        blocks.append(f"{i + 1}. Question: {q}\n   Evidence:\n{ev_txt}")
    return COVE_ANSWER_EVIDENCE.format(blocks="\n".join(blocks))


def cove_judge(items: Sequence[tuple]) -> str:
    lines = [f"{i + 1}. CLAIM: {c} | VERIFICATION QUESTION: {q} | VERIFICATION ANSWER: {a}" for i, (c, q, a) in enumerate(items)]
    return COVE_JUDGE.format(items="\n".join(lines))


def cove_revise(question: str, answer: str, qa: Sequence[tuple], abstain: str) -> str:
    return COVE_REVISE.format(question=question, answer=answer, abstain=abstain,
                              qa="\n".join(f"- Q: {q}\n  A: {a}" for q, a in qa) or "(none)")


# ------------------------------------------------------------------ RARR-style evidence-guided revision
REVISE = """You are editing factual claims that could not be verified. For each claim use ONLY its own evidence passages to choose one action:
- KEEP: the evidence supports the claim as written.
- REWRITE: the evidence supports a corrected version of the claim. Write the corrected claim as ONE sentence, changing as little as possible.
- REMOVE: the evidence does not allow a reliable correction.

{items}

Respond with JSON only, in exactly this form: {{"edits": [{{"id": 1, "action": "KEEP", "revised_claim": "", "evidence_ids": [1]}}]}}"""


def revise(items: Sequence[tuple]) -> str:
    """items: (claim, verdict_hint, [evidence passages])"""
    blocks = []
    for i, (claim, hint, ev) in enumerate(items):
        ev_txt = "\n".join(f"[{j + 1}] {e}" for j, e in enumerate(ev)) or "(no evidence found)"
        blocks.append(f"### Claim {i + 1}: {claim}\n(automatic check: {hint})\nEvidence for claim {i + 1}:\n{ev_txt}")
    return REVISE.format(items="\n\n".join(blocks))


# ------------------------------------------------------------------ reference-guided grading (evaluation only)
JUDGE = """You are grading a candidate answer to a question against trusted reference information.

Question: {question}

Reference information (treat as ground truth):
{reference}

Candidate answer:
{answer}

Atomic claims extracted from the candidate answer:
{claims}

Task 1 - grade the whole answer:
- CORRECT: it gives the key correct information and does not contradict the reference.
- INCORRECT: it asserts something that contradicts the reference, or gives a wrong answer.
- NOT_ATTEMPTED: it declines to answer or says the information is unavailable, without asserting anything wrong.
Task 2 - label each claim:
- SUPPORTED: stated or implied by the reference information.
- CONTRADICTED: conflicts with the reference information.
- UNVERIFIABLE: neither supported nor contradicted by the reference information.

Respond with JSON only, in exactly this form: {{"grade": "CORRECT", "claims": [{{"id": 1, "verdict": "SUPPORTED"}}]}}"""


def judge(question: str, reference: str, answer: str, claims: Sequence[str]) -> str:
    claims_txt = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims)) or "(none)"
    return JUDGE.format(question=question, reference=reference, answer=answer, claims=claims_txt)
