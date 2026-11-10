"""
LLM backend abstraction — Strategy pattern.

REVISED from the original single-class-with-if/elif design: an abstract
base class (LLMBackend) defines the shared interface, and each backend
(Groq, OpenAI) is its own concrete subclass. LLMClient is now a thin
factory/facade on top, so existing call sites elsewhere in the codebase
(pipeline.py, compressor.py, manager.py) don't need to change how they
ask for a backend — LLMClient().generate(prompt, backend="groq") still
works exactly as before, just backed by real polymorphism instead of an
if/elif chain internally.

Bundled into this same refactor (touching this file anyway):
- generate() now returns a typed LLMResponse dataclass instead of a
  plain dict — result.response / result.latency_ms instead of
  result["response"] / result["latency_ms"], catching a typo'd key at
  editor/type-checker time instead of a runtime KeyError three calls
  later.
- Backend selection now has a real Backend enum. LLMClient.generate()
  still accepts a plain string too ("groq"/"openai") for backward
  compatibility with every existing caller — Backend(backend) converts
  a string to the enum member — so this is additive, not a breaking
  change to the rest of the codebase.

Groq key rotation: the Groq backend sends every request through
pipeline.groq_key_pool (GROQ_API_KEYS=key1,key2,...; falls back to the
single GROQ_API_KEY). The pool handles rate limits (rotate / wait) and
raises DailyLimitExhausted when every key is out for the day.
"""

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from groq import APIConnectionError as GroqConnectionError
from groq import Groq, RateLimitError as GroqRateLimitError
from openai import APIConnectionError as OpenAIConnectionError
from openai import OpenAI, RateLimitError as OpenAIRateLimitError

from pipeline import config
from pipeline.groq_key_pool import DailyLimitExhausted, get_pool  # noqa: F401  (re-exported)


class MissingAPIKeyError(RuntimeError):
    """Raised when a backend is used but its API key is not configured."""


def _key_is_set(value) -> bool:
    """True for a real key; empty values and the .env.example placeholder count as unset."""
    return bool(value) and not str(value).startswith("your-")


class Backend(Enum):
    GROQ = "groq"
    OPENAI = "openai"


@dataclass
class LLMResponse:
    response: str
    latency_ms: float


class LLMBackend(ABC):
    """Shared interface every backend implements, plus the logic that's
    genuinely identical across backends: latency measurement and
    rate-limit retry with backoff. Only the actual SDK call — and which
    exception type means 'rate limited' for that SDK — differs per
    subclass, so only _call() and rate_limit_exc are backend-specific;
    everything else lives here once."""

    rate_limit_exc: type[Exception]  # set by each concrete subclass
    transient_excs: tuple = ()       # connection errors worth retrying (set per subclass)

    @abstractmethod
    def _call(self, prompt: str) -> str:
        """Makes the actual SDK call, returns the raw response text.
        Implemented per-backend — this is the one method each subclass
        must provide."""
        raise NotImplementedError

    def generate(self, prompt: str) -> LLMResponse:
        """Wrapped in perf_counter() (not wall-clock time.time()) since
        perf_counter is monotonic and immune to system clock adjustments
        — the right tool for measuring elapsed duration.

        Retries on rate-limit errors (see _call_with_retry) — any time
        spent sleeping for a retry is subtracted from latency_ms, so a
        rate-limited call doesn't get logged as if the model itself was
        slow. What's measured is real model response time only.
        """
        start = time.perf_counter()
        text, sleep_time = self._call_with_retry(prompt)
        latency_ms = ((time.perf_counter() - start) - sleep_time) * 1000
        return LLMResponse(response=text, latency_ms=latency_ms)

    def _call_with_retry(self, prompt: str, max_retries: int = 40):
        """Calls self._call(prompt), retrying on this backend's rate-limit
        error and on connection errors (a dropped network, a DNS failure).
        The wait is the time the provider asks for when it says so, otherwise
        exponential backoff (2s, 4s, 8s, ...) capped at 60s, so 40 attempts
        tolerate roughly 40 minutes of outage. Other errors (bad key, bad
        request) are raised immediately. Returns (result, total_seconds_slept).

        Free-tier rate limits (Groq: 8000 tokens/minute at time of
        writing) are easy to hit running 20-100 samples back-to-back —
        this is what turns a hard crash mid-experiment into an automatic
        wait-and-continue, needed for anything past a handful of calls.
        """
        total_slept = 0.0
        for attempt in range(max_retries):
            try:
                return self._call(prompt), total_slept
            except (self.rate_limit_exc, *self.transient_excs) as e:
                if attempt == max_retries - 1:
                    raise  # out of retries, let the real error surface
                wait = self._parse_retry_wait(str(e)) or min(2 ** (attempt + 1), 60)
                kind = "rate limit" if isinstance(e, self.rate_limit_exc) else "connection error"
                print(f"  [{kind}] retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                total_slept += wait
        raise RuntimeError("unreachable")  # loop always returns or raises

    @staticmethod
    def _parse_retry_wait(error_message: str) -> float | None:
        """Parses 'try again in 1m47.568s' / '1.185s' / '350ms' style hints."""
        match = re.search(r"try again in ([0-9hms.]+)", error_message)
        if not match:
            return None
        units = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
        total = sum(float(v) * units[u]
                    for v, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1)))
        return total + 0.5 if total else None  # +0.5s safety margin


class GroqBackend(LLMBackend):
    rate_limit_exc = GroqRateLimitError
    transient_excs = (GroqConnectionError,)

    @staticmethod
    def _configured_keys() -> list:
        """Groq keys, in order: groq_keys.txt (one key per line, '#' comments allowed; looked up
        in the repo root, then the current folder, or the path in GROQ_KEYS_FILE), then
        GROQ_API_KEYS / GROQ_API_KEY (comma/space/newline separated). Duplicates are dropped."""
        from pathlib import Path
        raw = []
        root = Path(__file__).resolve().parent.parent
        names = ["groq_keys.txt", "groq_keys.txt.txt"]  # Windows/Notepad often hides a double .txt
        candidates = ([Path(os.environ["GROQ_KEYS_FILE"])] if os.environ.get("GROQ_KEYS_FILE") else []) \
            + [d / n for d in (root, Path.cwd()) for n in names]
        found = next((c for c in candidates if c.is_file()), None)
        if found:
            data = found.read_bytes()
            if data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # PowerShell '>' writes UTF-16
                text = data.decode("utf-16", errors="ignore")
            else:
                text = data.decode("utf-8-sig", errors="ignore")
            text = text.replace("\x00", "")
            file_keys, commented = [], 0
            for line in text.splitlines():
                if line.strip().startswith("#"):  # '#' disables a line
                    commented += len(re.findall(r"gsk_[A-Za-z0-9]+", line))
                else:
                    file_keys.extend(re.findall(r"gsk_[A-Za-z0-9]+", line))
            raw.extend(file_keys)
            print(f"[groq] keys file: {found} ({len(file_keys)} key(s) found in it)", flush=True)
            if commented:
                print(f"[groq] NOTE: {commented} key(s) in that file are commented out with '#' and "
                      "are IGNORED. Remove the '#' to use them.", flush=True)
            if not file_keys:
                print("[groq] WARNING: no usable 'gsk_...' keys in that file; is it empty or saved wrongly?", flush=True)
        else:
            print(f"[groq] keys file NOT found. Looked in: {root} and {Path.cwd()}", flush=True)
        raw.append(os.environ.get("GROQ_API_KEYS", ""))
        raw.append(str(config.GROQ_API_KEY or ""))
        keys = []
        for k in re.split(r"[,\s]+", ",".join(raw)):
            k = k.strip().strip("\"'")
            if _key_is_set(k) and k not in keys:
                keys.append(k)
        if not keys:
            raise MissingAPIKeyError(
                "No Groq key configured (needed for the 'groq' backend). Put one key per line in "
                "groq_keys.txt (repo root), or set GROQ_API_KEYS=key1,key2,... in .env.")
        return keys

    _pool = None

    @property
    def pool(self):
        # Shared process-wide pool, created on first use (so a missing key only matters if
        # this backend is used). Keys are read once, not on every request.
        if GroqBackend._pool is None:
            GroqBackend._pool = get_pool(self._configured_keys())
        return GroqBackend._pool

    def _create(self, client, prompt: str):
        return client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

    def _call(self, prompt: str) -> str:
        response = self.pool.call(lambda c: self._create(c, prompt))
        return response.choices[0].message.content

    def generate(self, prompt: str) -> LLMResponse:
        """Rotation, cooldown waits and failed attempts all happen inside the pool.
        Latency is timed around the successful call only, so it stays comparable
        to earlier results (no sleeping or failed attempts included)."""
        def attempt(client):
            t0 = time.perf_counter()
            resp = self._create(client, prompt)
            return resp, (time.perf_counter() - t0) * 1000

        resp, latency_ms = self.pool.call(attempt)
        return LLMResponse(response=resp.choices[0].message.content, latency_ms=latency_ms)


class OpenAIBackend(LLMBackend):
    rate_limit_exc = OpenAIRateLimitError
    transient_excs = (OpenAIConnectionError,)

    def __init__(self):
        self._client = None  # created on first use, so a missing key only matters if this backend is used

    @property
    def client(self):
        if self._client is None:
            if not _key_is_set(config.OPENAI_API_KEY):
                raise MissingAPIKeyError(
                    "OPENAI_API_KEY is not set (needed for the 'openai' backend). "
                    "Add it to .env or the environment.")
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    def _call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class LLMClient:
    """Thin factory/facade over the two backend classes. Backend objects are
    created up front, but the SDK clients inside them are created lazily on
    first use, so ``LLMClient()`` works with only one API key configured and
    a missing key raises MissingAPIKeyError only when that backend is called."""

    def __init__(self):
        self._backends = {
            Backend.GROQ: GroqBackend(),
            Backend.OPENAI: OpenAIBackend(),
        }

        # A cheap, explicit flag other modules (like the compressor)
        # can check before choosing OpenAI, instead of trying the call
        # and catching an AuthenticationError every time. Treats the
        # unfilled .env.example placeholder as "not configured" too.
        self.has_openai = _key_is_set(config.OPENAI_API_KEY)

    def model_name(self, backend: "Backend | str" = Backend.GROQ) -> str:
        """Return the configured model name for a backend."""
        if isinstance(backend, str):
            backend = Backend(backend)
        return config.GROQ_MODEL if backend is Backend.GROQ else config.OPENAI_MODEL

    def generate(self, prompt: str, backend: "Backend | str" = Backend.GROQ) -> LLMResponse:
        """Send `prompt` to the given backend and return an LLMResponse.
        Accepts either a Backend enum member or a plain string
        ("groq"/"openai") — every existing caller in this codebase
        passes a string, and this keeps that working unchanged while
        Backend exists as the real, typo-proof way to refer to a
        backend going forward."""
        if isinstance(backend, str):
            backend = Backend(backend)  # raises ValueError on an unrecognized string, same as before
        return self._backends[backend].generate(prompt)


if __name__ == "__main__":
    # Quick manual test: "hello world" prompt against both backends.
    client = LLMClient()

    print("--- Groq ---")
    print(client.generate("Say hello world in one short sentence.", backend="groq"))

    print("\n--- OpenAI ---")
    print(client.generate("Say hello world in one short sentence.", backend="openai"))