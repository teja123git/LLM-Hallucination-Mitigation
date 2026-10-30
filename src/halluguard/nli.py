"""Natural-language-inference utilities: claim-vs-evidence verification and SelfCheckGPT-NLI consistency."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from .costs import local_timer
from .retrieval import chunk_text
from .utils import get_logger, softmax, tokenize

log = get_logger()
# Column order of every probability matrix returned by NLI models in this project:
ENTAIL, NEUTRAL, CONTRA = 0, 1, 2


class NLIModel:
    name = "base"

    def predict(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:  # (n, 3): entail, neutral, contra
        raise NotImplementedError

    def sanity_check(self) -> bool:
        pairs = [("A man is playing a guitar on stage.", "A man is playing an instrument."),
                 ("The sky is blue today.", "The sky is not blue today."),
                 ("The company was founded in 1998 by two students.", "The company sells fresh bananas.")]
        p = self.predict([a for a, _ in pairs], [b for _, b in pairs])
        ok = p[0].argmax() == ENTAIL and p[1].argmax() == CONTRA and p[2].argmax() == NEUTRAL
        if not ok:
            log.warning("NLI sanity check FAILED (probabilities: %s). Check the label mapping of %s.", np.round(p, 2).tolist(), self.name)
        return bool(ok)


class HFNLIModel(NLIModel):
    """HuggingFace sequence-classification NLI model (default: cross-encoder/nli-deberta-v3-base)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", batch_size: int = 16, max_length: int = 512, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_name
        self._torch = torch
        self.batch_size, self.max_length = batch_size, max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(self.device)
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        if not {"entailment", "neutral", "contradiction"} <= set(id2label.values()):
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}  # documented order of the cross-encoder/nli-* models
        inv = {v: k for k, v in id2label.items()}
        self._cols = [inv["entailment"], inv["neutral"], inv["contradiction"]]

    def predict(self, premises, hypotheses):
        torch = self._torch
        out: List[np.ndarray] = []
        premises, hypotheses = list(premises), list(hypotheses)
        with local_timer(), torch.no_grad():
            for i in range(0, len(premises), self.batch_size):
                enc = self.tok(premises[i : i + self.batch_size], hypotheses[i : i + self.batch_size], truncation=True,
                               max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float().cpu().numpy()
                out.append(softmax(logits, axis=-1)[:, self._cols])
        return np.concatenate(out, axis=0) if out else np.zeros((0, 3))


_NEG = {"not", "no", "never", "none", "n't", "neither", "nor", "without", "cannot"}


class LexicalNLI(NLIModel):
    """Deterministic overlap-based pseudo-NLI used ONLY for offline tests / mock mode (no model download)."""

    name = "lexical-mock"

    def predict(self, premises, hypotheses):
        rows = []
        with local_timer():
            for p, h in zip(premises, hypotheses):
                pt, ht = set(tokenize(p, True)), tokenize(h, True)
                if not ht:
                    rows.append([0.0, 1.0, 0.0])
                    continue
                cov = sum(t in pt for t in ht) / len(ht)
                neg_p = any(w in tokenize(p) or w in p.lower() for w in _NEG)
                neg_h = any(w in tokenize(h) or w in h.lower() for w in _NEG)
                nums_p, nums_h = set(re.findall(r"\d+", p)), set(re.findall(r"\d+", h))
                conflict = (neg_p != neg_h and cov > 0.5) or (nums_h and nums_p and not (nums_h & nums_p) and cov > 0.4)
                if conflict:
                    e, c = 0.1 * cov, 0.5 + 0.4 * cov
                else:
                    e, c = cov ** 1.5, 0.02
                n = max(0.0, 1.0 - e - c)
                v = np.array([e, n, c]) + 1e-6
                rows.append((v / v.sum()).tolist())
        return np.array(rows, dtype="float64").reshape(-1, 3)


def make_nli(cfg) -> NLIModel:
    if cfg.project.get("mock"):
        return LexicalNLI()
    return HFNLIModel(cfg.nli.model, cfg.nli.batch_size, cfg.nli.max_length)


# --------------------------------------------------------------------------- claim vs evidence
def nli_against_evidence(nli: NLIModel, claim: str, evidence: Sequence[str], window_words: int = 180, overlap: int = 30) -> Dict[str, float]:
    """Score ``claim`` against evidence passages.  Long passages are split into windows; the claim's
    entailment / contradiction score is the max over windows (SummaC-style), ``best_idx`` the index of the
    evidence passage that entails it most (used for citations)."""
    wins: List[str] = []
    owner: List[int] = []
    for i, e in enumerate(evidence):
        for w in chunk_text(e, window_words, overlap) or []:
            wins.append(w)
            owner.append(i)
    if not wins:
        return {"entail": 0.0, "neutral": 1.0, "contra": 0.0, "best_idx": -1}
    p = nli.predict(wins, [claim] * len(wins))
    j = int(np.argmax(p[:, ENTAIL]))
    return {"entail": float(p[:, ENTAIL].max()), "neutral": float(p[j, NEUTRAL]), "contra": float(p[:, CONTRA].max()), "best_idx": owner[j]}


def selfcheck_per_sample(nli: NLIModel, claim: str, samples: Sequence[str]) -> np.ndarray:
    """Per-sample SelfCheckGPT-NLI scores  P(contra) / (P(contra) + P(entail))  (one value per non-empty sample)."""
    samples = [s for s in samples if s and s.strip()]
    if not samples:
        return np.zeros(0)
    p = nli.predict(samples, [claim] * len(samples))
    pc, pe = p[:, CONTRA], p[:, ENTAIL]
    return pc / np.maximum(pc + pe, 1e-9)


def selfcheck_nli(nli: NLIModel, claim: str, samples: Sequence[str]) -> float:
    """SelfCheckGPT-NLI score in [0,1] (higher = more likely hallucinated): mean of the per-sample scores.
    With no usable sample the score is 0.5 (no information)."""
    s = selfcheck_per_sample(nli, claim, samples)
    return float(s.mean()) if len(s) else 0.5
