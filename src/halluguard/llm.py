"""LLM access layer.

Design goals (driven by the Groq free tier: 30 RPM / 6K TPM / 500K tokens-per-day):
  * **Never fabricate an answer on failure.**  Stage 1 returned the string "Error: ..." as if it were a
    model answer.  Here failures raise ``LLMError`` and the pipelines record the item as failed.
  * **Token-aware rate limiting** (requests *and* tokens per minute) with a sliding window.
  * **Persistent response cache** (SQLite).  Re-running a script costs nothing, a run that hits the
    daily quota can simply be restarted later, and Stage 1 / Stage 3 share their baseline calls.
    Cache hits replay the originally recorded token usage / latency so that cost accounting stays honest.
  * Prompt tokens and completion tokens are tracked separately, per call and per task.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

from .costs import Cost, record_llm
from .utils import get_logger, sha1

log = get_logger()


class LLMError(RuntimeError):
    """A call failed permanently (after retries)."""


class DailyQuotaExceeded(LLMError):
    """The provider's per-day token/request quota is exhausted. Re-run later: progress is cached."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    cached: bool = False
    model: str = ""
    task: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Sliding-window limiter on requests-per-minute and tokens-per-minute (thread-safe)."""

    def __init__(self, rpm: int, tpm: int, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.rpm, self.tpm = int(rpm), int(tpm)
        self._clock, self._sleep = clock, sleep
        self._events: Deque[List[float]] = deque()  # [timestamp, tokens]
        self._lock = threading.Lock()
        self.total_wait = 0.0

    def _purge(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def acquire(self, est_tokens: int) -> List[float]:
        """Block until a request of ``est_tokens`` may be sent. Returns the event handle for ``adjust``."""
        est = max(1, int(est_tokens))
        while True:
            with self._lock:
                now = self._clock()
                self._purge(now)
                used = sum(e[1] for e in self._events)
                req_ok = len(self._events) < self.rpm
                tok_ok = (used + est <= self.tpm) or not self._events  # a single oversized request must still be able to go
                if req_ok and tok_ok:
                    ev = [now, float(est)]
                    self._events.append(ev)
                    return ev
                wait = max(0.05, 60.0 - (now - self._events[0][0]) + 0.01)
            self._sleep(wait)
            self.total_wait += wait

    def adjust(self, handle: List[float], actual_tokens: int) -> None:
        with self._lock:
            handle[1] = float(actual_tokens)


# ---------------------------------------------------------------------------- cache
class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=60)
        self._lock = threading.Lock()
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:  # pragma: no cover
                pass
            self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._conn.commit()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache(key, value) VALUES (?, ?)", (key, json.dumps(value)))
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------- base class
class BaseLLM:
    """Template: cache lookup -> ``_complete`` -> cache store -> cost accounting."""

    def __init__(self, model: str, judge_model: Optional[str] = None, temperature: float = 0.3, max_tokens: int = 256,
                 sample_temperature: float = 0.7, cache: Optional[ResponseCache] = None, seed: int = 42):
        self.model = model
        self.judge_model = judge_model or model
        self.temperature = temperature
        self.sample_temperature = sample_temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.seed = seed
        self.task_stats: Dict[str, Cost] = defaultdict(Cost)
        self._stats_lock = threading.Lock()

    # subclasses implement this ------------------------------------------------
    def _complete(self, *, prompt: str, system: Optional[str], model: str, temperature: float, max_tokens: int,
                  seed: int, json_mode: bool, task: str) -> Tuple[str, int, int]:
        raise NotImplementedError

    # public API --------------------------------------------------------------
    def chat(self, prompt: str, *, task: str = "generic", system: Optional[str] = None, model: Optional[str] = None,
             temperature: Optional[float] = None, max_tokens: Optional[int] = None, sample_idx: int = 0,
             json_mode: bool = False) -> LLMResult:
        model = model or self.model
        temperature = self.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.max_tokens
        key = sha1(json.dumps({"m": model, "s": system, "p": prompt, "t": temperature, "mt": max_tokens, "i": sample_idx, "j": json_mode}, sort_keys=True))
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                res = LLMResult(hit["text"], hit["prompt_tokens"], hit["completion_tokens"], hit["latency"], True, model, task)
                self._account(res)
                return res
        t0 = time.perf_counter()
        text, pt, ct = self._complete(prompt=prompt, system=system, model=model, temperature=temperature, max_tokens=max_tokens,
                                      seed=self.seed * 1000 + sample_idx, json_mode=json_mode, task=task)
        latency = time.perf_counter() - t0
        res = LLMResult((text or "").strip(), pt, ct, latency, False, model, task)
        if self.cache is not None:
            self.cache.put(key, {"text": res.text, "prompt_tokens": pt, "completion_tokens": ct, "latency": latency})
        self._account(res)
        return res

    def samples(self, prompt: str, n: int, *, task: str = "samples", **kw) -> List[LLMResult]:
        kw.setdefault("temperature", self.sample_temperature)
        return [self.chat(prompt, task=task, sample_idx=i + 1, **kw) for i in range(n)]

    def _account(self, res: LLMResult) -> None:
        record_llm(res.prompt_tokens, res.completion_tokens, res.latency, res.cached)
        with self._stats_lock:
            c = self.task_stats[res.task]
            c.calls += 1
            c.cached_calls += int(res.cached)
            c.prompt_tokens += res.prompt_tokens
            c.completion_tokens += res.completion_tokens
            c.llm_latency += res.latency

    def stats_table(self) -> Dict[str, dict]:
        with self._stats_lock:
            return {k: v.to_dict() for k, v in self.task_stats.items()}


# ---------------------------------------------------------------------------- Groq
_DAILY_RE = re.compile(r"per day|\bTPD\b|\bRPD\b", re.I)
_UNIT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_retry_after(msg: str) -> Optional[float]:
    """Parse 'Please try again in 1m26.4s' / '850ms' / '2.3s' (Groq style) into seconds."""
    m = re.search(r"try again in\s+([0-9a-z\.\s]+)", msg or "", re.I)
    if not m:
        return None
    total, found = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", m.group(1), re.I):
        total += float(num) * _UNIT[unit.lower()]
        found = True
    return total if found else None


class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, rpm: int = 28, tpm: int = 5500, max_retries: int = 6, timeout: float = 60.0,
                 limits: Optional[Dict[str, Dict[str, int]]] = None, reasoning_effort: str = "low", reasoning_headroom: int = 600, **kw):
        super().__init__(**kw)
        try:
            import groq  # noqa: F401
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install the Groq SDK:  pip install groq") from exc
        import groq as _groq

        self._groq = _groq
        self.client = Groq(api_key=api_key, timeout=timeout, max_retries=0)  # we do our own retrying
        self._default_limits = (int(rpm), int(tpm))
        self._limits = {m: (int(v["rpm"]), int(v["tpm"])) for m, v in (limits or {}).items()}
        self._limiters: Dict[str, RateLimiter] = {}
        self._limiter_lock = threading.Lock()
        self.max_retries = max_retries
        self._json_supported = True
        self.reasoning_effort = reasoning_effort        # gpt-oss models always reason; "low" keeps it cheap
        self.reasoning_headroom = int(reasoning_headroom)  # hidden reasoning tokens count against the completion budget

    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        return model.startswith("openai/gpt-oss")

    def list_models(self) -> List[str]:
        return sorted(m.id for m in self.client.models.list().data)

    def limiter_for(self, model: str) -> RateLimiter:
        """Groq enforces quotas per model, so each model gets its own sliding window."""
        with self._limiter_lock:
            if model not in self._limiters:
                self._limiters[model] = RateLimiter(*self._limits.get(model, self._default_limits))
            return self._limiters[model]

    def _complete(self, *, prompt, system, model, temperature, max_tokens, seed, json_mode, task):
        g = self._groq
        limiter = self.limiter_for(model)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        reasoning = self.is_reasoning_model(model)
        budget = max_tokens + (self.reasoning_headroom if reasoning else 0)
        use_json = json_mode and self._json_supported
        last_err: Optional[Exception] = None
        spent_pt = spent_ct = 0          # tokens of discarded (empty) attempts are still charged to the cost accounting
        for attempt in range(self.max_retries + 1):
            est = (len(prompt) + len(system or "")) // 3 + budget
            handle = limiter.acquire(est)
            kwargs = dict(model=model, messages=messages, temperature=temperature, seed=seed)
            if reasoning:
                kwargs["max_completion_tokens"] = budget
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
            else:
                kwargs["max_tokens"] = budget
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self.client.chat.completions.create(**kwargs)
                usage = resp.usage
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                limiter.adjust(handle, pt + ct)
                spent_pt, spent_ct = spent_pt + pt, spent_ct + ct
                content = resp.choices[0].message.content or ""
                if not content.strip() and getattr(resp.choices[0], "finish_reason", "") == "length" and budget < 4 * max_tokens + self.reasoning_headroom:
                    budget *= 2          # the model used the whole budget for hidden reasoning: retry with more room
                    log.warning("Empty answer (reasoning used the whole token budget); retrying with max_completion_tokens=%d", budget)
                    continue
                return content, spent_pt, spent_ct
            except g.RateLimitError as e:
                last_err = e
                msg = str(e)
                if _DAILY_RE.search(msg):
                    raise DailyQuotaExceeded(
                        f"Daily quota reached for {model}. Everything finished so far is cached - re-run the same command later. Provider message: {msg[:300]}"
                    ) from e
                wait = parse_retry_after(msg)
                if wait is None:
                    try:
                        wait = float(e.response.headers.get("retry-after"))
                    except Exception:
                        wait = min(2.0 * (2 ** attempt), 60.0)
                log.warning("Rate limited (attempt %d/%d); sleeping %.1fs", attempt + 1, self.max_retries + 1, wait + 1)
                time.sleep(min(wait + 1.0, 180.0))
            except g.BadRequestError as e:
                last_err = e
                if use_json and ("response_format" in str(e) or "json" in str(e).lower()):
                    log.warning("Model rejected JSON mode; falling back to plain prompting for JSON tasks.")
                    self._json_supported = False
                    use_json = False
                    continue
                raise LLMError(f"Bad request: {e}") from e
            except (g.APIConnectionError, g.APITimeoutError, g.InternalServerError) as e:
                last_err = e
                wait = min(2.0 * (2 ** attempt), 60.0)
                log.warning("Transient API error (%s); retry in %.0fs", type(e).__name__, wait)
                time.sleep(wait)
            except g.NotFoundError as e:
                raise LLMError(f"Model '{model}' is not available to your Groq key (retired, renamed or no access). "
                               f"Run `python scripts/check_setup.py` to list the models you can use, then set llm.model / llm.judge_model in configs/default.yaml.") from e
            except g.AuthenticationError as e:
                raise LLMError("Groq authentication failed - check GROQ_API_KEY in your .env") from e
            except g.APIStatusError as e:
                raise LLMError(f"Groq API error {getattr(e, 'status_code', '?')}: {e}") from e
        raise LLMError(f"Failed after {self.max_retries + 1} attempts: {last_err}")

    def ping(self) -> str:
        return self.chat("Reply with the single word: ok", task="ping", max_tokens=5, temperature=0.0).text
