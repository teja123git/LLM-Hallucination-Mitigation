"""Hallucination detectors (claim level) and their calibration.

Every detector yields a score in [0,1] - higher means "more likely hallucinated / unsupported":

  s_nli    = 1 - max-entailment(claim | retrieved evidence)         (evidence-based, NLI cross-encoder)
  s_llm    = {SUPPORTED: 0, NOT_ENOUGH_INFO: 0.5, CONTRADICTED: 1}   (evidence-based, LLM verifier)
  s_sc     = SelfCheckGPT-NLI consistency vs. stochastic samples     (no external evidence)
  s_cove   = {CONSISTENT: 0, UNCERTAIN: 0.5, INCONSISTENT: 1}        (Chain-of-Verification, parametric, no retrieval)

Named detectors ("heads") combine these features; the multi-feature heads are logistic regressions fitted on the
HaluEval *dev* split in Stage 2.  A claim is *flagged* when ``score >= threshold`` (threshold = F1-optimal on dev).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import prompts
from .llm import BaseLLM
from .utils import parse_json, truncate_words

LLM_SCORE = {"SUPPORTED": 0.0, "NOT_ENOUGH_INFO": 0.5, "CONTRADICTED": 1.0}
COVE_SCORE = {"CONSISTENT": 0.0, "UNCERTAIN": 0.5, "INCONSISTENT": 1.0}

HEAD_FEATURES: Dict[str, List[str]] = {
    "nli": ["s_nli"],
    "llm": ["s_llm"],
    "selfcheck": ["s_sc"],
    "cove": ["s_cove"],
    "evidence": ["s_nli", "s_llm"],
    "hybrid": ["s_nli", "s_llm", "s_sc"],
}
DEFAULT_THRESHOLD = {"nli": 0.5, "llm": 0.5, "selfcheck": 0.5, "cove": 0.75, "evidence": 0.5, "hybrid": 0.5}


# --------------------------------------------------------------------------- LLM verifier
def _norm_verdict(raw: str) -> str:
    u = (raw or "").upper().replace(" ", "_")
    if u.startswith("SUPPORT") or u == "ENTAILED":
        return "SUPPORTED"
    if u.startswith("CONTRA") or u.startswith("REFUT") or u == "FALSE":
        return "CONTRADICTED"
    return "NOT_ENOUGH_INFO"


def llm_verify_claim(llm: BaseLLM, claim: str, evidence: Sequence[str], evidence_words: int = 120) -> Dict:
    """One call per claim; no call at all when there is no evidence."""
    if not evidence:
        return {"verdict": "NOT_ENOUGH_INFO", "evidence_ids": [], "reason": "no evidence retrieved", "score": LLM_SCORE["NOT_ENOUGH_INFO"]}
    res = llm.chat(prompts.verify(claim, [truncate_words(e, evidence_words) for e in evidence]), task="llm_verify",
                   json_mode=True, temperature=0.0, max_tokens=120)
    obj = parse_json(res.text)
    if isinstance(obj, dict) and "verdict" in obj:
        verdict = _norm_verdict(str(obj["verdict"]))
        ids = [int(i) - 1 for i in obj.get("evidence_ids", []) if str(i).isdigit() and 0 < int(i) <= len(evidence)]
        reason = str(obj.get("reason", ""))
    else:  # unparseable -> keyword fallback
        t = res.text.upper()
        verdict = "CONTRADICTED" if "CONTRADICT" in t else "SUPPORTED" if "SUPPORT" in t and "NOT" not in t.split("SUPPORT")[0][-6:] else "NOT_ENOUGH_INFO"
        ids, reason = [], "unparsed"
    return {"verdict": verdict, "evidence_ids": ids, "reason": reason, "score": LLM_SCORE[verdict]}


# --------------------------------------------------------------------------- Chain-of-Verification (no retrieval)
def cove_verify(llm: BaseLLM, question: str, claims: Sequence[str]) -> List[Dict]:
    """CoVe 'factored' variant: plan verification questions -> answer them independently (the draft is NOT shown)
    -> compare each claim with its independent answer.  3 LLM calls per answer, regardless of the number of claims."""
    claims = list(claims)
    if not claims:
        return []
    obj = parse_json(llm.chat(prompts.cove_plan(question, claims), task="cove_plan", json_mode=True, temperature=0.0, max_tokens=300).text)
    vq = {}
    if isinstance(obj, dict):
        for it in obj.get("questions", []) or []:
            try:
                vq[int(it["claim_id"]) - 1] = str(it["question"]).strip()
            except Exception:
                continue
    questions = [vq.get(i) or f"Is it true that {c.rstrip('.')}?" for i, c in enumerate(claims)]
    obj = parse_json(llm.chat(prompts.cove_answer(questions), task="cove_answer", json_mode=True, temperature=0.0, max_tokens=400).text)
    va = {}
    if isinstance(obj, dict):
        for it in obj.get("answers", []) or []:
            try:
                va[int(it["id"]) - 1] = str(it["answer"]).strip()
            except Exception:
                continue
    answers = [va.get(i, "unknown") for i in range(len(claims))]
    obj = parse_json(llm.chat(prompts.cove_judge(list(zip(claims, questions, answers))), task="cove_judge", json_mode=True, temperature=0.0, max_tokens=200).text)
    vd = {}
    if isinstance(obj, dict):
        for it in obj.get("verdicts", []) or []:
            try:
                v = str(it["verdict"]).upper()
                vd[int(it["id"]) - 1] = "CONSISTENT" if v.startswith("CONSIST") else "INCONSISTENT" if v.startswith("INCONS") else "UNCERTAIN"
            except Exception:
                continue
    out = []
    for i in range(len(claims)):
        v = vd.get(i, "UNCERTAIN")
        out.append({"verdict": v, "question": questions[i], "answer": answers[i], "score": COVE_SCORE[v]})
    return out


# --------------------------------------------------------------------------- detector heads
@dataclass
class DetectorHead:
    name: str
    features: List[str]
    weights: List[float] = field(default_factory=list)   # empty => unweighted mean of features
    bias: float = 0.0
    kind: str = "mean"                                    # mean | logreg
    threshold: float = 0.5

    def available(self, feats: Dict[str, Optional[float]]) -> bool:
        return all(feats.get(f) is not None for f in self.features)

    def score(self, feats: Dict[str, Optional[float]]) -> float:
        x = np.array([feats[f] for f in self.features], dtype=float)
        if self.kind == "logreg":
            z = float(np.dot(self.weights, x) + self.bias)
            return float(1.0 / (1.0 + np.exp(-z)))
        return float(x.mean())

    def flagged(self, feats: Dict[str, Optional[float]]) -> bool:
        return self.score(feats) >= self.threshold

    def to_dict(self) -> dict:
        return {"name": self.name, "features": self.features, "weights": list(map(float, self.weights)), "bias": float(self.bias),
                "kind": self.kind, "threshold": float(self.threshold)}

    @staticmethod
    def from_dict(d: dict) -> "DetectorHead":
        return DetectorHead(d["name"], d["features"], d.get("weights", []), d.get("bias", 0.0), d.get("kind", "mean"), d.get("threshold", 0.5))


@dataclass
class Calibration:
    heads: Dict[str, DetectorHead]
    nli_entail_threshold: float = 0.5     # 3-way decision (FEVER-style): entail >= t => SUPPORTS
    nli_contra_threshold: float = 0.5
    calibrated: bool = False

    @staticmethod
    def default() -> "Calibration":
        return Calibration({n: DetectorHead(n, f, threshold=DEFAULT_THRESHOLD[n]) for n, f in HEAD_FEATURES.items()})

    def head(self, name: str) -> DetectorHead:
        return self.heads[name]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"heads": {k: v.to_dict() for k, v in self.heads.items()}, "nli_entail_threshold": self.nli_entail_threshold,
                       "nli_contra_threshold": self.nli_contra_threshold, "calibrated": self.calibrated}, f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "Calibration":
        p = Path(path)
        if not p.exists():
            return Calibration.default()
        d = json.loads(p.read_text(encoding="utf-8"))
        base = Calibration.default()
        base.heads.update({k: DetectorHead.from_dict(v) for k, v in d["heads"].items()})
        base.nli_entail_threshold = d.get("nli_entail_threshold", 0.5)
        base.nli_contra_threshold = d.get("nli_contra_threshold", 0.5)
        base.calibrated = bool(d.get("calibrated", True))
        return base


def fit_head(name: str, X: np.ndarray, y: np.ndarray, criterion: str = "youden") -> DetectorHead:
    """Fit a detector head on (answer-level) features ``X`` and labels ``y`` (1 = hallucinated).
    The decision threshold maximises Youden's J (``criterion="youden"``, default) or F1 (``"f1"``)."""
    from .metrics import best_threshold

    feats = HEAD_FEATURES[name]
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(feats) == 1 or len(np.unique(y)) < 2:
        head = DetectorHead(name, feats, kind="mean")
        scores = X.mean(axis=1)
    else:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X, y)
        head = DetectorHead(name, feats, weights=lr.coef_[0].tolist(), bias=float(lr.intercept_[0]), kind="logreg")
        scores = np.array([head.score(dict(zip(feats, row))) for row in X])
    head.threshold = float(best_threshold(y, scores, criterion))
    return head


def aggregate(scores: Sequence[float], how: str = "max") -> float:
    """Claim scores -> answer score (an answer with no claims, e.g. an abstention, has nothing to flag)."""
    if not len(scores):
        return 0.0
    return float(np.max(scores) if how == "max" else np.mean(scores))
