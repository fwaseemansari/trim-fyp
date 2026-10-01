"""
token_analysis/counter.py

Core token-counting function for the Token Analysis Module.

Why this module exists:
    Every other module in TRIM (compression, context management) is measured
    against a single source of truth for "how many tokens is this?". That
    source of truth is this file. If counting logic drifts between modules,
    every downstream metric (cost, compression ratio, etc.) becomes unreliable.

Two different tokenizers, because we have two different LLM backends:
    - OpenAI models (gpt-4o-mini) -> tiktoken. This is OpenAI's own library;
      it doesn't call any API, it just runs the same tokenizer OpenAI's
      models use, locally and for free.
    - Groq-served model (openai/gpt-oss-20b) -> Groq doesn't expose a
      token-counting endpoint, so we use gpt-oss-20b's real tokenizer from
      HuggingFace directly (its own tokenizer, publicly available - no
      access request needed).
"""

import csv
import datetime
import os

import tiktoken
from transformers import AutoTokenizer

from token_analysis.pricing import estimate_cost

# --- Config: model name -> which tokenizer family it belongs to ---
OPENAI_MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
}

# Groq now serves openai/gpt-oss-20b (per the team's pricing.py/config.py),
# not Llama-3.3-70B as the original plan assumed. gpt-oss's own tokenizer
# is public/ungated on HuggingFace, so we use the real thing directly.
GROQ_MODELS = {
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
}

# --- Lazy-loaded caches ---
# tiktoken encoders and HF tokenizers are somewhat expensive to load.
# We don't want to reload them every single call to count_tokens(),
# especially once we're running this over hundreds of samples (Week 4).
_tiktoken_cache = {}
_hf_tokenizer_cache = {}


def _get_tiktoken_encoder(model: str):
    if model not in _tiktoken_cache:
        try:
            _tiktoken_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # tiktoken doesn't recognize gpt-4o-mini by name yet on some
            # versions - fall back to the encoding gpt-4o family actually uses.
            _tiktoken_cache[model] = tiktoken.get_encoding("o200k_base")
    return _tiktoken_cache[model]


def _get_hf_tokenizer(repo: str):
    if repo not in _hf_tokenizer_cache:
        _hf_tokenizer_cache[repo] = AutoTokenizer.from_pretrained(repo)
    return _hf_tokenizer_cache[repo]


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count how many tokens `text` would consume for the given `model`.
    Uses the REAL tokenizer per backend (tiktoken for OpenAI, the actual
    HF tokenizer for the Groq model) rather than approximating both with
    one encoding.

    Args:
        text: the raw string to tokenize (a prompt, a document, anything).
        model: model identifier. Must be a key in OPENAI_MODELS or GROQ_MODELS
               (e.g. "gpt-4o-mini" or "openai/gpt-oss-20b"). Defaults to
               "gpt-4o-mini" so it's a drop-in match for TokenAnalyzer's
               call signature below.

    Returns:
        Integer token count.

    Raises:
        ValueError: if `model` isn't recognized by either backend.
    """
    if model in OPENAI_MODELS:
        encoder = _get_tiktoken_encoder(OPENAI_MODELS[model])
        return len(encoder.encode(text))

    if model in GROQ_MODELS:
        tokenizer = _get_hf_tokenizer(GROQ_MODELS[model])
        # add_special_tokens=False: we want the raw content token count,
        # not inflated by BOS/EOS tokens the chat template would add.
        return len(tokenizer.encode(text, add_special_tokens=False))

    raise ValueError(
        f"Unknown model '{model}'. Known models: "
        f"{list(OPENAI_MODELS) + list(GROQ_MODELS)}"
    )


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
        log_run() - not file paths, despite the 'log' naming from the
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
    # Quick manual test: 5 sample strings, both backends, per the Monday task.
    samples = [
        "What is the capital of France?",
        "Summarize the causes of World War I in two sentences.",
        "def add(a, b):   return a + b",
        "The quick brown fox jumps over the lazy dog, repeatedly, for emphasis.",
        "Explain the difference between supervised and unsupervised learning.",
    ]

    print(f"{'Sample':<60} {'gpt-4o-mini':>12} {'gpt-oss-20b':>13}")
    print("-" * 90)
    for s in samples:
        openai_count = count_tokens(s, "gpt-4o-mini")
        groq_count = count_tokens(s, "openai/gpt-oss-20b")
        label = (s[:57] + "...") if len(s) > 57 else s
        print(f"{label:<60} {openai_count:>12} {groq_count:>13}")