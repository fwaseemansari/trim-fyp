"""
Token counting and per-call analysis.

NOTE (stand-in flag): Easha owns this module per the plan (Mon/Tue/Wed
tasks). Built as a stand-in so the pipeline has a real, working
TokenAnalyzer today — hand off / let her review and extend once she
starts.

TOKENIZER DECISION (revised): each model now gets counted with its OWN
real tokenizer instead of approximating both with a single tiktoken
encoding.
- gpt-4o-mini -> tiktoken.encoding_for_model("gpt-4o-mini"). This was
  already tiktoken's correct/real tokenizer for OpenAI models — no
  change in behavior here, just now looked up per-model instead of a
  single hardcoded encoding shared across models.
- openai/gpt-oss-20b (the Groq-served model) -> the model's own
  HuggingFace tokenizer, via transformers.AutoTokenizer. This IS the
  real tokenizer for that model, not an approximation like the earlier
  cl100k_base stand-in was. Loaded lazily (only on first count_tokens()
  call for this model) and cached after, since loading it is a real
  network call + disk read the first time (gpt-oss is Apache-2.0
  licensed and openly downloadable — no HF auth/license-acceptance
  step needed, unlike gated models such as Llama).
"""

import csv
import datetime
import os

import tiktoken

from token_analysis.pricing import estimate_cost

_tokenizer_cache = {}  # model name -> loaded tokenizer object, filled lazily


def _get_tokenizer(model: str):
    """Returns a cached tokenizer for `model`, loading it on first use.
    Each model routes to its own real tokenizer rather than one shared
    approximation — see module docstring."""
    if model in _tokenizer_cache:
        return _tokenizer_cache[model]

    if model == "openai/gpt-oss-20b":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    else:
        # Covers gpt-4o-mini and any other OpenAI model name — tiktoken
        # is the correct, real tokenizer for these (not an approximation).
        try:
            tok = tiktoken.encoding_for_model(model)
        except KeyError:
            # Unknown/unreleased model name tiktoken doesn't recognize
            # yet — cl100k_base is the closest fallback rather than a
            # hard crash, since most recent OpenAI models share it.
            tok = tiktoken.get_encoding("cl100k_base")

    _tokenizer_cache[model] = tok
    return tok


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Token count for `text` using `model`'s own real tokenizer."""
    tok = _get_tokenizer(model)
    return len(tok.encode(text))  # same method name on both tiktoken and HF tokenizer objects


class TokenAnalyzer:
    """Given a prompt/response pair, computes token + cost breakdown,
    and logs runs to a CSV so every module's cost is measured the same
    way (the 'single source of truth' the plan calls for)."""

    def __init__(self, log_path: str = "evaluation/logs/token_log.csv"):
        self.log_path = log_path

    def analyze(self, prompt: str, response: str, model: str) -> dict:
        input_tokens = count_tokens(prompt, model)
        output_tokens = count_tokens(response, model)
        total_tokens = input_tokens + output_tokens
        cost = estimate_cost(model, input_tokens, output_tokens)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
        }

    def log_run(self, prompt: str, response: str, backend: str, model: str, latency_ms: float) -> dict:
        """Analyzes the run and appends one row to the CSV log. Returns
        the analysis dict so callers (pipeline.py) can also use the
        numbers immediately without re-reading the CSV."""
        analysis = self.analyze(prompt, response, model)

        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "backend": backend,
            "model": model,
            "prompt_length_chars": len(prompt),
            "input_tokens": analysis["input_tokens"],
            "output_tokens": analysis["output_tokens"],
            "total_tokens": analysis["total_tokens"],
            "cost_usd": round(analysis["cost_usd"], 6),
            "latency_ms": round(latency_ms, 1),
        }

        file_exists = os.path.isfile(self.log_path)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return analysis

    def compare(self, baseline_log: dict, treatment_log: dict) -> dict:
        """Computes % reduction in tokens/cost and latency delta between
        a baseline run's analysis dict and a treatment (e.g. compressed)
        run's analysis dict. Both are the dicts returned by analyze()/
        log_run() — not file paths, despite the 'log' naming from the
        plan (kept the plan's parameter names for continuity)."""
        token_reduction_pct = (
            (baseline_log["total_tokens"] - treatment_log["total_tokens"])
            / baseline_log["total_tokens"]
            * 100
            if baseline_log["total_tokens"] > 0
            else 0.0
        )
        cost_reduction_pct = (
            (baseline_log["cost_usd"] - treatment_log["cost_usd"])
            / baseline_log["cost_usd"]
            * 100
            if baseline_log["cost_usd"] > 0
            else 0.0
        )
        latency_delta_ms = treatment_log.get("latency_ms", 0) - baseline_log.get("latency_ms", 0)

        return {
            "token_reduction_pct": round(token_reduction_pct, 2),
            "cost_reduction_pct": round(cost_reduction_pct, 2),
            "latency_delta_ms": round(latency_delta_ms, 1),
        }


if __name__ == "__main__":
    # Quick manual check: same 3 sample strings, counted with BOTH
    # real tokenizers, to see how much they actually differ.
    samples = [
        "Hello world.",
        "The quick brown fox jumps over the lazy dog.",
        "TRIM is a token reduction and intelligent management framework.",
    ]
    for s in samples:
        gpt_tokens = count_tokens(s, "gpt-4o-mini")
        groq_tokens = count_tokens(s, "openai/gpt-oss-20b")
        print(f"{s[:45]!r:47} | gpt-4o-mini: {gpt_tokens:>3} | gpt-oss-20b: {groq_tokens:>3}")
