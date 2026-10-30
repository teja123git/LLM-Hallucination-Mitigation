"""All metrics: detection, retrieval, QA / text overlap, abstention-aware answer grading, statistical tests."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .utils import STOPWORDS, tokenize


# =========================================================================== detection
def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_f1_threshold(y: Sequence[int], scores: Sequence[float]) -> float:
    """Threshold t (flag iff score >= t) maximising F1 of the positive (hallucinated) class.
    Ties are broken towards the *higher* threshold (flag less => remove fewer claims)."""
    y, s = np.asarray(y, int), np.asarray(scores, float)
    if y.sum() == 0 or len(y) == 0:
        return 0.5
    best_t, best_f = 0.5, -1.0
    for t in sorted(set(np.round(s, 6)), reverse=True):
        pred = s >= t - 1e-9
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        f = _prf(tp, fp, fn)[2]
        if f > best_f + 1e-12:
            best_t, best_f = float(t), f
    return best_t


def best_youden_threshold(y: Sequence[int], scores: Sequence[float]) -> float:
    """Threshold maximising Youden's J = TPR - FPR (flag iff score >= t).  Unlike F1 it does not depend on the class
    prevalence, so it does not collapse to 'flag everything' on a 50/50 benchmark with a weak detector.
    Ties go to the higher threshold; a detector with no signal (J <= 0) falls back to 0.5."""
    y, s = np.asarray(y, int), np.asarray(scores, float)
    P, N = int((y == 1).sum()), int((y == 0).sum())
    if P == 0 or N == 0:
        return 0.5
    best_t, best_j = 0.5, 1e-12
    for t in sorted(set(np.round(s, 6)), reverse=True):
        pred = s >= t - 1e-9
        j = (pred & (y == 1)).sum() / P - (pred & (y == 0)).sum() / N
        if j > best_j + 1e-12:
            best_t, best_j = float(t), float(j)
    return best_t


def best_threshold(y: Sequence[int], scores: Sequence[float], criterion: str = "youden") -> float:
    return best_youden_threshold(y, scores) if criterion == "youden" else best_f1_threshold(y, scores)


def detection_metrics(y: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    """Binary detection quality (positive class = hallucinated). AUROC/AUPRC are threshold-free."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, s = np.asarray(y, int), np.asarray(scores, float)
    pred = s >= threshold - 1e-9
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    p, r, f = _prf(tp, fp, fn)
    both = len(np.unique(y)) == 2
    return {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else float("nan"), "threshold": float(threshold),
            "accuracy": (tp + tn) / max(1, len(y)), "precision": p, "recall": r, "f1": f,
            "auroc": float(roc_auc_score(y, s)) if both else float("nan"),
            "auprc": float(average_precision_score(y, s)) if both else float("nan"),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {"accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), "macro_f1": float(f.mean()),
            "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])} for i, l in enumerate(labels)},
            "confusion": cm.tolist(), "labels": list(labels)}


# =========================================================================== retrieval
def retrieval_metrics(ranked: Sequence[Sequence[int]], gold: Sequence[Iterable[int]], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    """Recall@k (= hit-rate: any gold passage in the top-k) and MRR over queries with a known gold set."""
    gold = [set(g) for g in gold]
    out: Dict[str, float] = {}
    n = max(1, len(ranked))
    for k in ks:
        out[f"recall@{k}"] = sum(bool(set(r[:k]) & g) for r, g in zip(ranked, gold)) / n
    rr = []
    for r, g in zip(ranked, gold):
        pos = next((i for i, pid in enumerate(r, 1) if pid in g), None)
        rr.append(1.0 / pos if pos else 0.0)
    out["mrr"] = float(np.mean(rr)) if rr else 0.0
    out["n"] = len(ranked)
    return out


def context_coverage(passages: Sequence[str], reference: str) -> Dict[str, float]:
    """How much of the reference answer is present in the retrieved context?
    ``token_recall``: fraction of the reference's content words found in the context;
    ``contains``: 1 if the whole normalised reference string occurs verbatim."""
    ctx = " ".join(passages)
    toks = [t for t in tokenize(reference, True)]
    ctx_set = set(tokenize(ctx))
    rec = sum(t in ctx_set for t in toks) / len(toks) if toks else 0.0
    return {"token_recall": rec, "contains": float(normalize_answer(reference) in normalize_answer(ctx)) if reference.strip() else 0.0}


# =========================================================================== text / QA metrics
def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation / articles / extra whitespace."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def contains_match(pred: str, gold: str) -> float:
    """Lenient QA accuracy for verbose LLM answers: does the normalised gold answer occur inside the prediction?"""
    g = normalize_answer(gold)
    return float(bool(g) and g in normalize_answer(pred))


_rouge = None


def rouge_l(pred: str, ref: str) -> Dict[str, float]:
    global _rouge
    if _rouge is None:
        from rouge_score import rouge_scorer

        _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    s = _rouge.score(str(ref), str(pred))["rougeL"]
    return {"precision": s.precision, "recall": s.recall, "f1": s.fmeasure}


def bertscore_f1(preds: Sequence[str], refs: Sequence[str], model_type: Optional[str] = None, rescale: bool = True) -> np.ndarray:
    """BERTScore F1 (empty strings are replaced so the scorer never crashes)."""
    from bert_score import score as bs

    c = [p.strip() or "no answer" for p in preds]
    r = [x.strip() or "no reference" for x in refs]
    kw = dict(lang="en", verbose=False)
    if model_type:
        kw["model_type"] = model_type
    try:
        _, _, f = bs(c, r, rescale_with_baseline=rescale, **kw)
    except Exception:  # baseline file missing for this model => raw scores
        _, _, f = bs(c, r, **kw)
    return f.numpy()


# =========================================================================== abstention-aware grading (SimpleQA style)
def simpleqa_metrics(grades: Sequence[str]) -> Dict[str, float]:
    """grades in {CORRECT, INCORRECT, NOT_ATTEMPTED}.  Rewards neither guessing nor blanket abstention:
    ``correct_given_attempted`` = precision of what the system chose to answer, ``f_score`` = harmonic mean of
    ``correct`` (recall-like) and ``correct_given_attempted``."""
    g = [x for x in grades if x in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")]
    n = len(g)
    if n == 0:
        return {k: float("nan") for k in ("n", "correct", "incorrect", "not_attempted", "correct_given_attempted", "f_score")}
    c, i, na = (g.count("CORRECT") / n, g.count("INCORRECT") / n, g.count("NOT_ATTEMPTED") / n)
    cga = c / (c + i) if (c + i) > 0 else 0.0
    f = 2 * c * cga / (c + cga) if (c + cga) > 0 else 0.0
    return {"n": n, "correct": c, "incorrect": i, "not_attempted": na, "correct_given_attempted": cga, "f_score": f}


# =========================================================================== statistics
def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                 stat: Callable[[np.ndarray], float] = np.mean) -> Tuple[float, float, float]:
    """(point estimate, lower, upper) percentile bootstrap over items."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[i]) for i in idx])
    return float(stat(v)), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean(b - a) with 95% CI over paired items (same questions answered by both systems)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_ci((b - a)[m], n_boot=n_boot, seed=seed)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired binary outcomes (system A vs. system B)."""
    from scipy.stats import binomtest

    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = float(binomtest(min(only_a, only_b), n, 0.5).pvalue) if n > 0 else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": p}
