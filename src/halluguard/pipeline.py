"""The factuality pipeline and the ablation systems A-G.

Pipeline stages (proposal, Section 4)::

    1 generation  ->  2 claim extraction  ->  3 detection  ->  4 retrieval  ->  5 correction  ->  6 final answer

Systems (one controlled ablation, every system answers the *same* questions):

    A       LLM only                                   direct answer
    B       LLM + RAG                                  hybrid retrieval, grounded answer
    C       LLM + self-consistency                     SelfCheckGPT-NLI flags claims, flagged claims are removed
    D       LLM + verification                         Chain-of-Verification (parametric, no retrieval), flagged claims removed
    E       LLM + RAG + verification                   claims checked against retrieved evidence (NLI + LLM verifier), flagged removed
    F       LLM + RAG + verification + correction      flagged claims are rewritten from evidence (RARR-style), verify-then-accept
    G       F with the full hybrid detector            NLI + LLM verifier + self-consistency fused by a calibrated logistic head
    F_cove  F with a CoVe-style correction strategy    strategy comparison (answer-level revision instead of claim-level edits)

Efficiency / controlled comparison
----------------------------------
All systems share intermediate results through ``QuestionState``: the direct answer, the RAG answer, the claims, the
retrieved evidence, the NLI scores, ... are each computed once per question.  The cost of a system is the sum of the
cost of the stages *it logically depends on* (cached results replay their original cost), so reported latency / tokens
/ calls are exactly what that system would spend if it were run on its own.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import prompts
from .claims import extract_claims
from .config import Cfg
from .correction import Edit, compose_answer, cove_correct, revise_claims
from .costs import Cost, track_cost
from .detection import HEAD_FEATURES, Calibration, cove_verify, llm_verify_claim
from .llm import BaseLLM
from .nli import NLIModel, nli_against_evidence, selfcheck_nli
from .retrieval import HybridRetriever, local_rank
from .utils import is_abstention, sha1

SYSTEMS: Dict[str, Dict[str, Any]] = {
    "A": dict(label="A: LLM only", init="A", detector=None, action="none"),
    "B": dict(label="B: LLM + RAG", init="B", detector=None, action="none"),
    "C": dict(label="C: LLM + self-consistency", init="A", detector="selfcheck", action="filter"),
    "D": dict(label="D: LLM + verification (CoVe)", init="A", detector="cove", action="filter"),
    "E": dict(label="E: LLM + RAG + verification", init="R", detector="evidence", action="filter"),
    "F": dict(label="F: + correction (RARR)", init="R", detector="evidence", action="rarr"),
    "G": dict(label="G: hybrid detector + correction", init="R", detector="hybrid", action="rarr"),
    "F_cove": dict(label="F-CoVe: + correction (CoVe)", init="R", detector="evidence", action="cove"),
}

STAGE_GROUPS = [("answer", "generation"), ("retrieve", "retrieval"), ("evidence", "retrieval"), ("claims", "claims"),
                ("samples", "self-consistency"), ("selfcheck", "self-consistency"), ("nli", "verification"),
                ("llmv", "verification"), ("cove", "verification"), ("revise", "correction"), ("covecorr", "correction")]


def stage_group(name: str) -> str:
    head = name.split(":")[0]
    return next((g for p, g in STAGE_GROUPS if head == p), "other")


def ptxt(h: dict) -> str:
    return f"{h['title']}: {h['text']}" if h.get("title") else h["text"]


# =========================================================================== per-question state
class QuestionState:
    """Memoises stage results for one question and records, per stage, what it cost and what it depended on."""

    def __init__(self, qid: str, question: str, fixed_evidence: Optional[Sequence[str]] = None, sample_prompt: Optional[str] = None):
        self.qid, self.question = qid, question
        self.answers: Dict[str, str] = {}                  # externally supplied answers (HaluEval right/hallucinated, RAGTruth response)
        self.preset_claims: Dict[str, List[str]] = {}      # externally supplied claims (FEVER)
        self.fixed_evidence = list(fixed_evidence) if fixed_evidence is not None else None
        self.sample_prompt = sample_prompt
        self.results: Dict[str, Any] = {}
        self.costs: Dict[str, Cost] = {}
        self.deps: Dict[str, Set[str]] = defaultdict(set)
        self._stack: List[str] = []
        self._touched: Set[str] = set()

    def begin(self) -> None:
        self._touched = set()

    def _touch(self, name: str) -> None:
        if name in self._touched:
            return
        self._touched.add(name)
        for d in self.deps.get(name, ()):
            self._touch(d)

    def run(self, name: str, fn: Callable[[], Any], count: bool = True) -> Any:
        if count:
            self._touch(name)
        if self._stack:
            self.deps[self._stack[-1]].add(name)
        if name not in self.results:
            self._stack.append(name)
            try:
                with track_cost(isolate=True) as c:
                    val = fn()
            finally:
                self._stack.pop()
            self.results[name] = val
            self.costs[name] = c
        return self.results[name]

    def touched_cost(self) -> Cost:
        return Cost.sum([self.costs[n] for n in self._touched if n in self.costs])

    def touched_cost_by_group(self) -> Dict[str, dict]:
        agg: Dict[str, Cost] = defaultdict(Cost)
        for n in self._touched:
            if n in self.costs:
                agg[stage_group(n)] = agg[stage_group(n)] + self.costs[n]
        return {g: c.to_dict() for g, c in agg.items()}


# =========================================================================== records
@dataclass
class ClaimRecord:
    text: str
    scores: Dict[str, Optional[float]] = field(default_factory=dict)   # s_nli, s_llm, s_sc, s_cove, head score
    detail: Dict[str, Any] = field(default_factory=dict)               # nli probabilities, llm verdict / reason, cove Q&A
    label: str = "supported"                                           # supported | unsupported | contradicted
    flagged: bool = False
    evidence: List[dict] = field(default_factory=list)
    cited: List[int] = field(default_factory=list)                     # indices into ``evidence``
    action: str = "kept"                                               # kept | removed | rewritten
    revised: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SystemOutput:
    system: str
    qid: str
    question: str
    initial_source: str = ""        # which answer was verified: A | B | A(fallback)
    initial_answer: str = ""
    final_answer: str = ""
    final_answer_cited: str = ""
    sources: List[dict] = field(default_factory=list)
    claims: List[dict] = field(default_factory=list)
    final_claims: List[str] = field(default_factory=list)   # atomic claims of the final answer (for claim-level evaluation)
    abstained: bool = False
    n_flagged: int = 0
    n_removed: int = 0
    n_rewritten: int = 0
    cost: Dict[str, float] = field(default_factory=dict)
    cost_by_group: Dict[str, dict] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================================== pipeline
class FactualityPipeline:
    def __init__(self, cfg: Cfg, llm: BaseLLM, retriever: Optional[HybridRetriever], nli: NLIModel, calibration: Optional[Calibration] = None):
        self.cfg, self.llm, self.retr, self.nli = cfg, llm, retriever, nli
        self.cal = calibration or Calibration.default()
        self.p, self.rc = cfg.pipeline, cfg.retrieval

    # ------------------------------------------------------------------ generation
    def answer_A(self, st: QuestionState) -> str:
        return st.run("answer:A", lambda: self.llm.chat(prompts.direct(st.question), task="direct", temperature=self.cfg.llm.temperature).text)

    def retrieve_question(self, st: QuestionState) -> List[dict]:
        return st.run("retrieve:q", lambda: [h.to_dict() for h in self.retr.search(st.question, k=self.rc.top_k)])

    def answer_B(self, st: QuestionState) -> str:
        def fn() -> str:
            hits = self.retrieve_question(st)
            return self.llm.chat(prompts.rag(st.question, [ptxt(h) for h in hits]), task="rag", temperature=self.cfg.llm.temperature).text
        return st.run("answer:B", fn)

    def _answer(self, st: QuestionState, which: str, count: bool = True) -> str:
        if which in st.answers:
            return st.answers[which]
        return self.answer_A(st) if which == "A" else self.answer_B(st)

    def initial(self, st: QuestionState, mode: str) -> Tuple[str, str, bool]:
        """(answer key, text, used_fallback).  Mode R = RAG, falling back to the parametric answer if RAG abstains."""
        if mode == "A":
            return "A", self.answer_A(st), False
        b = self.answer_B(st)
        if mode == "B":
            return "B", b, False
        if is_abstention(b) and self.p.rag_fallback_to_parametric:
            return "A", self.answer_A(st), True
        return "B", b, False

    # ------------------------------------------------------------------ claims & evidence
    def claims(self, st: QuestionState, which: str, count: bool = True) -> List[str]:
        if which in st.preset_claims:
            return st.preset_claims[which]
        return st.run(f"claims:{which}", lambda: extract_claims(self.llm, st.question, self._answer(st, which), self.cfg.claims.max_claims), count)

    def text_claims(self, st: QuestionState, text: str) -> List[str]:
        """Claims of an arbitrary final text (evaluation overhead - never charged to a system)."""
        return st.run(f"claims:txt:{sha1(text)[:12]}", lambda: extract_claims(self.llm, st.question, text, self.cfg.claims.max_claims), count=False)

    def evidence(self, st: QuestionState, which: str) -> List[List[dict]]:
        def fn() -> List[List[dict]]:
            out = []
            for c in self.claims(st, which):
                if st.fixed_evidence is not None:
                    idx = local_rank(c, st.fixed_evidence, self.rc.evidence_k)
                    out.append([{"pid": -1 - i, "title": "", "text": st.fixed_evidence[i], "score": 0.0, "source": "fixed", "rank": r + 1}
                                for r, i in enumerate(idx)])
                else:
                    q = c if self.rc.claim_query == "claim" else f"{st.question} {c}"
                    out.append([h.to_dict() for h in self.retr.search(q, k=self.rc.evidence_k)])
            return out
        return st.run(f"evidence:{which}", fn)

    # ------------------------------------------------------------------ detectors' raw signals
    def nli_stage(self, st: QuestionState, which: str) -> List[dict]:
        def fn():
            cl, ev = self.claims(st, which), self.evidence(st, which)
            return [nli_against_evidence(self.nli, c, [ptxt(h) for h in e], self.cfg.nli.window_words, self.cfg.nli.window_overlap) for c, e in zip(cl, ev)]
        return st.run(f"nli:{which}", fn)

    def llm_stage(self, st: QuestionState, which: str) -> List[dict]:
        def fn():
            cl, ev = self.claims(st, which), self.evidence(st, which)
            return [llm_verify_claim(self.llm, c, [ptxt(h) for h in e], self.p.evidence_words) for c, e in zip(cl, ev)]
        return st.run(f"llmv:{which}", fn)

    def samples(self, st: QuestionState) -> List[str]:
        prompt = st.sample_prompt or prompts.direct(st.question)
        return st.run("samples", lambda: [r.text for r in self.llm.samples(prompt, self.cfg.selfcheck.n_samples, task="samples")])

    def selfcheck_stage(self, st: QuestionState, which: str) -> List[float]:
        def fn():
            cl, smp = self.claims(st, which), self.samples(st)
            return [selfcheck_nli(self.nli, c, smp) for c in cl]
        return st.run(f"selfcheck:{which}", fn)

    def cove_stage(self, st: QuestionState, which: str) -> List[dict]:
        return st.run(f"cove:{which}", lambda: cove_verify(self.llm, st.question, self.claims(st, which)))

    def features(self, st: QuestionState, which: str, names: Sequence[str]) -> List[dict]:
        """Per claim: raw feature values (None when not requested) and the details behind them."""
        cl = self.claims(st, which)
        want = set(names)
        ev = self.evidence(st, which) if want & {"s_nli", "s_llm"} else [[] for _ in cl]
        nl = self.nli_stage(st, which) if "s_nli" in want else None
        lv = self.llm_stage(st, which) if "s_llm" in want else None
        sc = self.selfcheck_stage(st, which) if "s_sc" in want else None
        cv = self.cove_stage(st, which) if "s_cove" in want else None
        rows = []
        for i, c in enumerate(cl):
            rows.append({"claim": c, "evidence": ev[i],
                         "feats": {"s_nli": (1.0 - nl[i]["entail"]) if nl else None, "s_llm": lv[i]["score"] if lv else None,
                                   "s_sc": sc[i] if sc else None, "s_cove": cv[i]["score"] if cv else None},
                         "detail": {"nli": nl[i] if nl else None, "llm": lv[i] if lv else None, "cove": cv[i] if cv else None}})
        return rows

    # ------------------------------------------------------------------ detection
    def analyze(self, st: QuestionState, which: str, head: str) -> List[ClaimRecord]:
        H = self.cal.head(head)
        recs: List[ClaimRecord] = []
        for row in self.features(st, which, HEAD_FEATURES[head]):
            f, d = row["feats"], row["detail"]
            score = H.score(f)
            flagged = score >= H.threshold
            nli, lv, cv = d["nli"], d["llm"], d["cove"]
            contradicted = bool((lv and lv["verdict"] == "CONTRADICTED") or (cv and cv["verdict"] == "INCONSISTENT")
                                or (nli and nli["contra"] >= self.cal.nli_contra_threshold and nli["contra"] > nli["entail"]))
            label = "supported" if not flagged else "contradicted" if contradicted else "unsupported"
            recs.append(ClaimRecord(row["claim"], dict(f, head=score), d, label, flagged, row["evidence"]))
        return recs

    @staticmethod
    def _best_citation(rec: ClaimRecord) -> List[int]:
        nli, lv = rec.detail.get("nli"), rec.detail.get("llm")
        if nli and nli["best_idx"] >= 0 and nli["entail"] >= 0.3:
            return [nli["best_idx"]]
        if lv and lv["verdict"] == "SUPPORTED" and lv["evidence_ids"]:
            return lv["evidence_ids"][:2]
        return []

    # ------------------------------------------------------------------ correction
    def _rarr(self, st: QuestionState, which: str, head: str, flagged: List[ClaimRecord]) -> None:
        """Revise flagged claims from *their own* evidence, then verify-then-accept with NLI."""
        def fn():
            edits = revise_claims(self.llm, [(r.text, r.label, [ptxt(h) for h in r.evidence]) for r in flagged], self.p.evidence_words)
            out = []
            for r, e in zip(flagged, edits):
                if e.action == "REMOVE":
                    out.append({"action": "REMOVE", "revised": "", "accepted": False, "entail": 0.0, "best_idx": -1})
                    continue
                cand = r.text if e.action == "KEEP" else e.revised
                chk = nli_against_evidence(self.nli, cand, [ptxt(h) for h in r.evidence], self.cfg.nli.window_words, self.cfg.nli.window_overlap)
                out.append({"action": e.action, "revised": cand, "accepted": chk["entail"] >= self.p.correction_accept_entail,
                            "entail": chk["entail"], "best_idx": chk["best_idx"]})
            return out
        for r, e in zip(flagged, st.run(f"revise:{which}:{head}", fn)):
            if e["accepted"]:
                r.action = "rewritten" if e["action"] == "REWRITE" else "kept"
                r.revised = e["revised"] if e["action"] == "REWRITE" else None
                r.cited = [e["best_idx"]] if e["best_idx"] >= 0 else []
            else:
                r.action = "removed"

    # ------------------------------------------------------------------ a full system
    def run_system(self, st: QuestionState, sid: str, eval_claims: bool = True) -> SystemOutput:
        spec = SYSTEMS[sid]
        st.begin()
        out = SystemOutput(system=sid, qid=st.qid, question=st.question)
        which, init_text, fallback = self.initial(st, spec["init"])
        out.initial_source = f"{which}(fallback)" if fallback else which
        out.initial_answer = init_text
        abstain = self.p.abstain_text

        if spec["detector"] is None:                                   # --- A, B: plain generation
            out.final_answer = out.final_answer_cited = init_text
            out.abstained = is_abstention(init_text)
            out.final_claims = self.claims(st, which, count=False) if (eval_claims and not out.abstained) else []
        else:
            head = spec["detector"]
            recs = self.analyze(st, which, head)
            flagged = [r for r in recs if r.flagged]
            out.n_flagged = len(flagged)
            for r in recs:
                r.action = "kept" if not r.flagged else "removed"
                r.cited = self._best_citation(r) if (not r.flagged and r.evidence) else []
            final_text: Optional[str] = None
            if spec["action"] == "rarr" and flagged:
                self._rarr(st, which, head, flagged)
            elif spec["action"] == "cove" and flagged:
                final_text, _ = st.run(f"covecorr:{which}:{head}", lambda: cove_correct(
                    self.llm, self.retr, st.question, init_text, [r.text for r in flagged], self.rc.evidence_k, self.p.evidence_words, abstain))
            out.n_removed = sum(r.action == "removed" for r in recs)
            out.n_rewritten = sum(r.action == "rewritten" for r in recs)

            if final_text is not None:                                   # F_cove: free-text answer-level revision
                out.final_answer = out.final_answer_cited = final_text
                out.abstained = is_abstention(final_text)
                out.final_claims = [] if (out.abstained or not eval_claims) else self.text_claims(st, final_text)
            else:                                                        # claim-level systems: compose from surviving claims
                source_idx: Dict[Any, int] = {}
                sources: List[dict] = []
                kept: List[Tuple[str, List[int]]] = []
                for r in recs:
                    if r.action == "removed":
                        continue
                    idxs = []
                    for ci in r.cited:
                        if 0 <= ci < len(r.evidence):
                            h = r.evidence[ci]
                            key = h["pid"] if h["pid"] >= 0 else h["text"]
                            if key not in source_idx:
                                source_idx[key] = len(sources)
                                sources.append(h)
                            idxs.append(source_idx[key])
                    kept.append((r.revised or r.text, idxs))
                plain, cited, numbered = compose_answer(kept, sources, abstain)
                unchanged = not flagged or all(r.action == "kept" and r.revised is None for r in recs)
                out.final_answer = init_text if (unchanged and recs) else plain
                out.final_answer_cited = cited if kept else abstain
                out.sources = numbered
                out.abstained = not kept
                out.final_claims = [t for t, _ in kept]
            out.claims = [r.to_dict() for r in recs]

        out.cost = st.touched_cost().to_dict()
        out.cost_by_group = st.touched_cost_by_group()
        return out

    def run_all(self, st: QuestionState, systems: Sequence[str], eval_claims: bool = True) -> Dict[str, SystemOutput]:
        return {s: self.run_system(st, s, eval_claims) for s in systems}
