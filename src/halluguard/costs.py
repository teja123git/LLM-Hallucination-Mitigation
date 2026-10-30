"""Cost accounting (LLM calls, tokens, latency) with nestable thread-local trackers.

Every LLM call (also cache hits - they replay the originally recorded usage) and every local
computation (retrieval, NLI) reports to all active trackers of the calling thread.  The pipeline
uses one tracker per stage so that the cost of each ablation system can be assembled from the
stages it actually needs, even though intermediate results are shared between systems.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List


@dataclass
class Cost:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency: float = 0.0     # seconds spent waiting for the API (original latency is replayed for cache hits)
    local_latency: float = 0.0   # seconds of local compute (retrieval / NLI)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def latency(self) -> float:
        return self.llm_latency + self.local_latency

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            self.calls + other.calls,
            self.cached_calls + other.cached_calls,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.llm_latency + other.llm_latency,
            self.local_latency + other.local_latency,
        )

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["latency"] = self.latency
        return d

    @staticmethod
    def sum(costs: List["Cost"]) -> "Cost":
        out = Cost()
        for c in costs:
            out = out + c
        return out


_state = threading.local()


def _stack() -> List[Cost]:
    if not hasattr(_state, "stack"):
        _state.stack = []
    return _state.stack


@contextmanager
def track_cost(isolate: bool = False) -> Iterator[Cost]:
    """Track cost inside the ``with`` block.  ``isolate=True`` hides the block from every enclosing tracker
    (used for pipeline stages, so that a stage which triggers a dependency is not charged for it twice)."""
    c = Cost()
    st = _stack()
    saved = None
    if isolate:
        saved = list(st)
        st.clear()
    st.append(c)
    try:
        yield c
    finally:
        if isolate:
            st.clear()
            st.extend(saved or [])
        elif c in st:
            st.remove(c)


def record_llm(prompt_tokens: int, completion_tokens: int, latency: float, cached: bool) -> None:
    for c in _stack():
        c.calls += 1
        c.cached_calls += int(cached)
        c.prompt_tokens += int(prompt_tokens)
        c.completion_tokens += int(completion_tokens)
        c.llm_latency += float(latency)


def record_local(seconds: float) -> None:
    for c in _stack():
        c.local_latency += float(seconds)


@contextmanager
def local_timer() -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record_local(time.perf_counter() - t0)
