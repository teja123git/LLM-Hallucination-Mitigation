"""Stage 5/6 - evidence-guided correction and answer composition with citations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import prompts
from .llm import BaseLLM
from .utils import parse_json, truncate_words


@dataclass
class Edit:
    action: str                       # KEEP | REWRITE | REMOVE
    revised: str = ""
    evidence_ids: List[int] = field(default_factory=list)   # 0-based indices into the claim's evidence list


def revise_claims(llm: BaseLLM, items: Sequence[Tuple[str, str, Sequence[str]]], evidence_words: int = 120) -> List[Edit]:
    """RARR-style revision.  items = (claim, automatic-check hint, evidence texts).  One LLM call per answer.
    Anything the model fails to address is REMOVEd (the conservative default)."""
    items = list(items)
    if not items:
        return []
    prompt = prompts.revise([(c, h, [truncate_words(e, evidence_words) for e in ev]) for c, h, ev in items])
    obj = parse_json(llm.chat(prompt, task="revise", json_mode=True, temperature=0.0, max_tokens=120 * len(items) + 80).text)
    edits: Dict[int, Edit] = {}
    if isinstance(obj, dict):
        for it in obj.get("edits", []) or []:
            try:
                i = int(it["id"]) - 1
                act = str(it.get("action", "REMOVE")).upper()
                act = "KEEP" if act.startswith("KEEP") else "REWRITE" if act.startswith("REWRITE") else "REMOVE"
                ids = [int(x) - 1 for x in it.get("evidence_ids", []) if str(x).isdigit()]
                edits[i] = Edit(act, str(it.get("revised_claim", "") or "").strip(), ids)
            except Exception:
                continue
    out = []
    for i in range(len(items)):
        e = edits.get(i, Edit("REMOVE"))
        if e.action == "REWRITE" and not e.revised:
            e = Edit("REMOVE")
        out.append(e)
    return out


def cove_correct(llm: BaseLLM, retriever, question: str, draft: str, flagged_claims: Sequence[str],
                 evidence_k: int, evidence_words: int, abstain: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Chain-of-Verification with retrieval: plan verification questions for the flagged claims, answer each from
    evidence retrieved *for that question*, then write a revised final answer.  3 LLM calls."""
    flagged = list(flagged_claims)
    obj = parse_json(llm.chat(prompts.cove_plan(question, flagged), task="cove_plan", json_mode=True, temperature=0.0, max_tokens=300).text)
    vq: Dict[int, str] = {}
    if isinstance(obj, dict):
        for it in obj.get("questions", []) or []:
            try:
                vq[int(it["claim_id"]) - 1] = str(it["question"]).strip()
            except Exception:
                continue
    questions = [vq.get(i) or f"Is it true that {c.rstrip('.')}?" for i, c in enumerate(flagged)]
    evid = [[truncate_words(h.full_text, evidence_words) for h in retriever.search(q, k=evidence_k)] for q in questions]
    obj = parse_json(llm.chat(prompts.cove_answer_evidence(questions, evid), task="cove_answer", json_mode=True, temperature=0.0, max_tokens=400).text)
    va: Dict[int, str] = {}
    if isinstance(obj, dict):
        for it in obj.get("answers", []) or []:
            try:
                va[int(it["id"]) - 1] = str(it["answer"]).strip()
            except Exception:
                continue
    qa = [(q, va.get(i, "unknown")) for i, q in enumerate(questions)]
    text = llm.chat(prompts.cove_revise(question, draft, qa, abstain), task="cove_revise", temperature=0.0, max_tokens=200).text.strip()
    return text or abstain, qa


def compose_answer(claims: Sequence[Tuple[str, Sequence[int]]], sources: Sequence[Dict], abstain: str) -> Tuple[str, str, List[Dict]]:
    """Join claims into a final answer.

    ``claims`` = [(claim_text, [source indices into ``sources``])].  Returns
    (plain_text, text_with_[n]_markers, numbered_source_list)."""
    if not claims:
        return abstain, abstain, []
    order: Dict[int, int] = {}
    plain, cited = [], []
    for text, srcs in claims:
        text = text.strip()
        plain.append(text)
        marks = ""
        for s in srcs:
            if s not in order:
                order[s] = len(order) + 1
            marks += f"[{order[s]}]"
        cited.append(f"{text}{(' ' + marks) if marks else ''}")
    numbered = [dict(sources[s], n=n) for s, n in sorted(order.items(), key=lambda kv: kv[1])]
    return " ".join(plain), " ".join(cited), numbered
