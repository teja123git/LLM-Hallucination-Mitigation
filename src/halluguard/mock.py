"""Offline fake LLM for smoke tests (``configs/mock.yaml`` / ``HALLUGUARD_MOCK=1``).

It answers every task type with deterministic, *structurally valid* output so that all prompts, parsers, pipeline
branches, metrics and plots can be exercised without an API key or internet access.  Its content is meaningless:
**results produced in mock mode must never be reported** (all mock figures are watermarked).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple

from .llm import BaseLLM
from .prompts import NO_INFO
from .utils import STOPWORDS, split_sentences, tokenize


def _h(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest(), 16)


def _content(text: str) -> List[str]:
    return [t for t in tokenize(text, True) if len(t) > 2]


def _overlap(a: str, b: str) -> float:
    ta, tb = set(_content(a)), set(_content(b))
    return len(ta & tb) / len(ta) if ta else 0.0


def _between(prompt: str, start: str, end: Optional[str] = None) -> str:
    i = prompt.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = prompt.find(end, i) if end else -1
    return prompt[i:j if j > 0 else None].strip()


class MockLLM(BaseLLM):
    def __init__(self, **kw):
        kw.setdefault("model", "mock-llm")
        super().__init__(**kw)

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _direct_answer(question: str) -> str:
        words = [w.capitalize() for w in _content(question)[-3:]] or ["Unknown"]
        extra = f", which dates back to {1900 + _h(question) % 100}" if _h(question, "y") % 2 == 0 else ""
        return f"The answer is {' '.join(words)}{extra}."

    def _complete(self, *, prompt, system, model, temperature, max_tokens, seed, json_mode, task):
        text = getattr(self, f"_t_{task}", self._t_generic)(prompt, seed)
        return text, len(prompt) // 4, max(1, len(text) // 4)

    # ---- tasks -------------------------------------------------------------
    def _t_generic(self, prompt, seed):
        return "ok"

    def _t_ping(self, prompt, seed):
        return "ok"

    def _t_direct(self, prompt, seed):
        return self._direct_answer(_between(prompt, "Question:", "Answer:"))

    def _t_samples(self, prompt, seed):
        q = _between(prompt, "Question:", "Answer:")
        base = self._direct_answer(q)
        if _h(q, seed) % 5 < 3:
            return base
        return " ".join(reversed(base.split()))

    def _t_rag(self, prompt, seed):
        q = _between(prompt, "Question:", "Answer:")
        ctx = _between(prompt, "Context:", "Question:")
        best, score = "", 0.0
        for line in ctx.splitlines():
            for s in split_sentences(re.sub(r"^\[\d+\]\s*", "", line)):
                o = _overlap(q, s)
                if o > score:
                    best, score = s, o
        return best if score >= 0.2 else NO_INFO

    def _t_claims(self, prompt, seed):
        ans = _between(prompt, "Answer:", "Respond with JSON")
        sents = split_sentences(ans) or [ans]
        return json.dumps({"claims": [s.rstrip(".") + "." for s in sents if s][:5]})

    def _t_llm_verify(self, prompt, seed):
        ev = _between(prompt, "Evidence:", "Claim:")
        claim = _between(prompt, "Claim:", "Verdicts:")
        o = _overlap(claim, ev)
        if o >= 0.7:
            v = "SUPPORTED"
        elif o >= 0.4 and _h(claim) % 3 == 0:
            v = "CONTRADICTED"
        else:
            v = "NOT_ENOUGH_INFO"
        return json.dumps({"verdict": v, "evidence_ids": [1] if v != "NOT_ENOUGH_INFO" else [], "reason": "mock"})

    def _t_cove_plan(self, prompt, seed):
        claims = re.findall(r"^\d+\.\s*(.+)$", _between(prompt, "Claims:", "Respond with JSON"), re.M)
        return json.dumps({"questions": [{"claim_id": i + 1, "question": f"What is known about: {' '.join(_content(c)[:4])}?"} for i, c in enumerate(claims)]})

    def _t_cove_answer(self, prompt, seed):
        qs = re.findall(r"^\s*\d+\.\s*(?:Question:\s*)?(.+)$", prompt, re.M)
        qs = [q for q in qs if "?" in q]
        return json.dumps({"answers": [{"id": i + 1, "answer": "unknown" if _h(q) % 3 == 0 else q.replace("What is known about:", "It is about").rstrip("?") + "."} for i, q in enumerate(qs)]})

    def _t_cove_judge(self, prompt, seed):
        n = len(re.findall(r"^\d+\.\s*CLAIM:", prompt, re.M))
        vs = ["CONSISTENT", "INCONSISTENT", "UNCERTAIN"]
        return json.dumps({"verdicts": [{"id": i + 1, "verdict": vs[_h(prompt, i) % 3]} for i in range(n)]})

    def _t_revise(self, prompt, seed):
        blocks = re.split(r"### Claim \d+:", prompt)[1:]
        edits = []
        for i, b in enumerate(blocks):
            claim = b.split("\n", 1)[0].strip()
            ev = [re.sub(r"^\[\d+\]\s*", "", l) for l in b.splitlines() if re.match(r"^\[\d+\]", l)]
            ev_text = " ".join(ev)
            if not ev:
                edits.append({"id": i + 1, "action": "REMOVE", "revised_claim": "", "evidence_ids": []})
            elif _overlap(claim, ev_text) >= 0.6:
                edits.append({"id": i + 1, "action": "KEEP", "revised_claim": "", "evidence_ids": [1]})
            else:
                sent = (split_sentences(ev[0]) or [ev[0]])[0]
                edits.append({"id": i + 1, "action": "REWRITE", "revised_claim": " ".join(sent.split()[:25]), "evidence_ids": [1]})
        return json.dumps({"edits": edits})

    def _t_cove_revise(self, prompt, seed):
        qa = re.findall(r"A: (.+)", prompt)
        good = [a for a in qa if a.strip().lower() != "unknown"]
        return good[0] if good else "I could not find enough reliable evidence to answer this question."

    def _t_judge(self, prompt, seed):
        ref = _between(prompt, "Reference information (treat as ground truth):", "Candidate answer:")
        ans = _between(prompt, "Candidate answer:", "Atomic claims extracted")
        claims = re.findall(r"^\d+\.\s*(.+)$", _between(prompt, "Atomic claims extracted from the candidate answer:", "Task 1"), re.M)
        if NO_INFO.lower() in ans.lower() or "could not find enough" in ans.lower():
            return json.dumps({"grade": "NOT_ATTEMPTED", "claims": []})
        o = _overlap(ans, ref)
        grade = "CORRECT" if o >= 0.5 else "INCORRECT"
        cl = [{"id": i + 1, "verdict": "SUPPORTED" if _overlap(c, ref) >= 0.5 else "CONTRADICTED" if _h(c) % 2 else "UNVERIFIABLE"} for i, c in enumerate(claims)]
        return json.dumps({"grade": grade, "claims": cl})
