"""
Thin wrapper around the two LLM backends (Groq, OpenAI) so every other
module (compression, context manager, evaluation) calls one interface
— LLMClient.generate() — regardless of which backend is behind it.

This keeps backend-specific SDK details (Groq() vs OpenAI()) out of the
rest of the codebase, so swapping/adding a backend later (e.g. the
optional local HF model) only touches this file.
"""

import time

from groq import Groq
from openai import OpenAI

from pipeline import config


class LLMClient:
    def __init__(self):
        # Both clients are created eagerly. If a key is missing, the
        # SDK will raise only when that backend is actually called,
        # not on import — keeps this usable even with one key set.
        self._groq = Groq(api_key=config.GROQ_API_KEY)
        self._openai = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, prompt: str, backend: str = "groq") -> dict:
        """Send `prompt` to the given backend ('groq' or 'openai') and
        return {"response": str, "latency_ms": float}.

        Wrapped in perf_counter() (not wall-clock time.time()) since
        perf_counter is monotonic and immune to system clock adjustments
        — the right tool for measuring elapsed duration."""
        start = time.perf_counter()

        if backend == "groq":
            response = self._generate_groq(prompt)
        elif backend == "openai":
            response = self._generate_openai(prompt)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'groq' or 'openai'.")

        latency_ms = (time.perf_counter() - start) * 1000
        return {"response": response, "latency_ms": latency_ms}

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

