"""
Thin wrapper around the two LLM backends (Groq, OpenAI) so every other
module (compression, context manager, evaluation) calls one interface
— LLMClient.generate() — regardless of which backend is behind it.

This keeps backend-specific SDK details (Groq() vs OpenAI()) out of the
rest of the codebase, so swapping/adding a backend later (e.g. the
optional local HF model) only touches this file.
"""

import re
import time

from groq import Groq, RateLimitError as GroqRateLimitError
from openai import OpenAI, RateLimitError as OpenAIRateLimitError

from pipeline import config


class LLMClient:
    def __init__(self):
        # Both clients are created eagerly. If a key is missing, the
        # SDK will raise only when that backend is actually called,
        # not on import — keeps this usable even with one key set.
        self._groq = Groq(api_key=config.GROQ_API_KEY)
        self._openai = OpenAI(api_key=config.OPENAI_API_KEY)

        # A cheap, explicit flag other modules (like the compressor)
        # can check before choosing OpenAI, instead of trying the call
        # and catching an AuthenticationError every time. Treats the
        # unfilled .env.example placeholder as "not configured" too.
        self.has_openai = bool(
            config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your-openai-key-here"
        )

    def generate(self, prompt: str, backend: str = "groq") -> dict:
        """Send `prompt` to the given backend ('groq' or 'openai') and
        return {"response": str, "latency_ms": float}.

        Wrapped in perf_counter() (not wall-clock time.time()) since
        perf_counter is monotonic and immune to system clock adjustments
        — the right tool for measuring elapsed duration.

        Retries on rate-limit errors (see _call_with_retry) — any time
        spent sleeping for a retry is subtracted from latency_ms, so a
        rate-limited call doesn't get logged as if the model itself was
        slow. What's measured is real model response time only.
        """
        start = time.perf_counter()

        if backend == "groq":
            response, sleep_time = self._call_with_retry(self._generate_groq, prompt, GroqRateLimitError)
        elif backend == "openai":
            response, sleep_time = self._call_with_retry(self._generate_openai, prompt, OpenAIRateLimitError)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'groq' or 'openai'.")

        latency_ms = ((time.perf_counter() - start) - sleep_time) * 1000
        return {"response": response, "latency_ms": latency_ms}

    def _call_with_retry(self, fn, prompt: str, rate_limit_exc, max_retries: int = 5):
        """Calls fn(prompt), retrying on rate_limit_exc with exponential
        backoff (2s, 4s, 8s, 16s, 32s). Returns (result, total_seconds_slept).

        Free-tier rate limits (Groq: 8000 tokens/minute at time of
        writing) are easy to hit running 20-100 samples back-to-back —
        this is what turns a hard crash mid-experiment into an automatic
        wait-and-continue, needed for anything past a handful of calls.
        """
        total_slept = 0.0
        for attempt in range(max_retries):
            try:
                return fn(prompt), total_slept
            except rate_limit_exc as e:
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

    def _generate_groq(self, prompt: str) -> str:
        response = self._groq.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _generate_openai(self, prompt: str) -> str:
        response = self._openai.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    # Quick manual test: "hello world" prompt against both backends.
    client = LLMClient()

    print("--- Groq ---")
    print(client.generate("Say hello world in one short sentence.", backend="groq"))

    print("\n--- OpenAI ---")
    print(client.generate("Say hello world in one short sentence.", backend="openai"))