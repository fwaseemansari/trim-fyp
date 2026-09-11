"""
Token counting and per-call analysis.

NOTE (stand-in flag): Easha owns this module per the plan (Mon/Tue/Wed
tasks). Built as a stand-in so the pipeline has a real, working
TokenAnalyzer today — hand off / let her review and extend once she
starts.

DECISION WORTH FLAGGING: the plan specifies tiktoken for OpenAI and
HuggingFace AutoTokenizer (Llama's tokenizer) for Groq, since Groq was
originally serving Llama-3.3-70B. Since GROQ_MODEL is now
openai/gpt-oss-20b (see config.py's model-availability fix), that specific
tokenizer justification no longer applies cleanly, and gpt-oss's real
tokenizer isn't guaranteed to be a quick, ungated download.

Alternative considered: use AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
for exact Groq-side counts. Rejected for now — first run downloads
tokenizer files from the HuggingFace Hub (network-dependent, adds
transformers as a hard runtime dependency for a task that's just
counting).

What's actually done here: tiktoken's cl100k_base encoding is used to
approximate BOTH backends' token counts. This is accurate for OpenAI
and a reasonable (not exact) proxy for Groq — most modern tokenizers
land within ~10-15% of each other on English text. Good enough for
Week 1-2 baseline/comparison numbers, where you're measuring relative
reduction (before vs. after compression) more than absolute counts.
Revisit with a real Groq-side tokenizer before any number goes in the
report as an exact per-backend count.
"""

import csv
import datetime
import os

import tiktoken

from token_analysis.pricing import estimate_cost

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Approximate token count for `text`. `model` is accepted (per the
    plan's function signature) but currently every model routes through
    the same cl100k_base encoding — see module docstring for why."""
    return len(_ENCODING.encode(text))


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
    # Quick manual check: 5 sample strings, printed token counts.
    samples = [
        "Hello world.",
        "The quick brown fox jumps over the lazy dog.",
        "TRIM is a token reduction and intelligent management framework.",
        "A" * 500,
        "",
    ]
    for s in samples:
        print(f"{len(s):>4} chars -> {count_tokens(s)} tokens | {s[:40]!r}")
