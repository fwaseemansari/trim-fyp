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
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from groq import Groq, RateLimitError as GroqRateLimitError
from openai import OpenAI, RateLimitError as OpenAIRateLimitError

from pipeline import config


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

    def _call_with_retry(self, prompt: str, max_retries: int = 5):
        """Calls self._call(prompt), retrying on this backend's
        rate_limit_exc with exponential backoff (2s, 4s, 8s, 16s, 32s).
        Returns (result, total_seconds_slept).

        Free-tier rate limits (Groq: 8000 tokens/minute at time of
        writing) are easy to hit running 20-100 samples back-to-back —
        this is what turns a hard crash mid-experiment into an automatic
        wait-and-continue, needed for anything past a handful of calls.
        """
        total_slept = 0.0
        for attempt in range(max_retries):
            try:
                return self._call(prompt), total_slept
            except self.rate_limit_exc as e:
                if attempt == max_retries - 1:
                    raise  # out of retries, let the real error surface
                wait = self._parse_retry_wait(str(e)) or (2 ** (attempt + 1))
                print(f"  [rate limit] retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                total_slept += wait
        raise RuntimeError("unreachable")  # loop always returns or raises

    @staticmethod
    def _parse_retry_wait(error_message: str) -> float | None:
        """Groq's error message includes a suggested wait, e.g. 'Please
        try again in 1.185s'. Use it when present (more accurate than a
        blind exponential guess); fall back to exponential backoff
        otherwise."""
        match = re.search(r"try again in ([\d.]+)s", error_message)
        return float(match.group(1)) + 0.5 if match else None  # +0.5s safety margin


class GroqBackend(LLMBackend):
    rate_limit_exc = GroqRateLimitError

    def __init__(self):
        self._client = Groq(api_key=config.GROQ_API_KEY)

    def _call(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class OpenAIBackend(LLMBackend):
    rate_limit_exc = OpenAIRateLimitError

    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def _call(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class LLMClient:
    """Thin factory/facade over the two backend classes. Both backend
    instances are still created eagerly (unchanged from before this
    refactor) — constructing an SDK client object doesn't make a
    network call, so this costs nothing even if only one backend ends
    up being used."""

    def __init__(self):
        self._backends = {
            Backend.GROQ: GroqBackend(),
            Backend.OPENAI: OpenAIBackend(),
        }

        # A cheap, explicit flag other modules (like the compressor)
        # can check before choosing OpenAI, instead of trying the call
        # and catching an AuthenticationError every time. Treats the
        # unfilled .env.example placeholder as "not configured" too.
        self.has_openai = bool(
            config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your-openai-key-here"
        )

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
