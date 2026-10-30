"""Stage 2 - atomic, decontextualised claim extraction."""
from __future__ import annotations

from typing import List

from . import prompts
from .llm import BaseLLM
from .utils import is_abstention, parse_json, split_sentences


def _fallback(question: str, answer: str, max_claims: int) -> List[str]:
    answer = (answer or "").strip()
    if not answer:
        return []
    if len(answer.split()) <= 6:  # bare entity answers ("Arthur's Magazine") are meaningless without the question
        return [f'The answer to the question "{question}" is {answer.rstrip(".")}.']
    return split_sentences(answer)[:max_claims]


def extract_claims(llm: BaseLLM, question: str, answer: str, max_claims: int = 5) -> List[str]:
    """Decompose ``answer`` into at most ``max_claims`` self-contained claims (empty list for abstentions)."""
    if is_abstention(answer):
        return []
    res = llm.chat(prompts.claims(question, answer, max_claims), task="claims", json_mode=True, temperature=0.0, max_tokens=300)
    obj = parse_json(res.text)
    claims = obj.get("claims") if isinstance(obj, dict) else obj if isinstance(obj, list) else None
    if isinstance(claims, list):
        claims = [str(c).strip() for c in claims if isinstance(c, (str, int, float)) and str(c).strip()]
    else:
        claims = []
    seen, uniq = set(), []
    for c in claims:
        if c.lower() not in seen:
            seen.add(c.lower())
            uniq.append(c)
    return uniq[:max_claims] or _fallback(question, answer, max_claims)
