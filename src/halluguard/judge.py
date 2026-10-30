"""Reference-guided LLM grading of final answers and their atomic claims (evaluation only).

Independence from the system under test:
  * a different, stronger model (``llm.judge_model``) than the generator / verifier (``llm.model``);
  * the judge sees *gold* references (HaluEval: right answer + gold paragraph; TruthfulQA: curated lists of correct /
    incorrect answers) - never the retrieved evidence or any detector output.
Grades follow SimpleQA: CORRECT / INCORRECT / NOT_ATTEMPTED, so abstaining cannot inflate the score.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import prompts
from .llm import BaseLLM
from .pipeline import SystemOutput
from .utils import is_abstention, parse_json

GRADES = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
CLAIM_VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")


def reference_block(item: dict, kind: str) -> str:
    if kind == "halueval":
        return f"Reference answer: {item['right_answer']}\nSupporting passage: {item['knowledge']}"
    ok = "\n".join(f"- {a}" for a in item["correct_answers"][:8])
    bad = "\n".join(f"- {a}" for a in item["incorrect_answers"][:8])
    return (f"Correct answers (the candidate is correct if it agrees with these):\n{ok}\n"
            f"Incorrect answers (common misconceptions - a candidate asserting any of these is INCORRECT):\n{bad}")


def judge_output(llm: BaseLLM, item: dict, kind: str, out: SystemOutput) -> Dict:
    """Returns {grade, claim_verdicts, judged} - ``judged`` False if the judge reply could not be parsed."""
    if out.abstained or is_abstention(out.final_answer):
        return {"grade": "NOT_ATTEMPTED", "claim_verdicts": [], "judged": True}
    prompt = prompts.judge(item["question"], reference_block(item, kind), out.final_answer, out.final_claims)
    res = llm.chat(prompt, task="judge", model=llm.judge_model, json_mode=True, temperature=0.0, max_tokens=60 + 25 * len(out.final_claims) + 40)
    obj = parse_json(res.text)
    if not isinstance(obj, dict):
        return {"grade": None, "claim_verdicts": [], "judged": False}
    g = str(obj.get("grade", "")).upper().replace(" ", "_")
    grade = next((x for x in GRADES if g.startswith(x[:4])), None)
    verdicts: List[Optional[str]] = [None] * len(out.final_claims)
    for it in obj.get("claims", []) or []:
        try:
            i = int(it["id"]) - 1
            v = str(it["verdict"]).upper()
            verdicts[i] = "SUPPORTED" if v.startswith("SUPPORT") else "CONTRADICTED" if v.startswith("CONTRA") else "UNVERIFIABLE"
        except Exception:
            continue
    return {"grade": grade, "claim_verdicts": verdicts, "judged": grade is not None}
