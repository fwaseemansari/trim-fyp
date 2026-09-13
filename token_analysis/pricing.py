"""
Hardcoded per-1K-token pricing for cost estimation. These are published
rates, not measured - update if OpenAI/Groq change pricing.

Sources (checked Sep 2026):
- OpenAI gpt-4o-mini: https://openai.com/api/pricing/
- Groq openai/gpt-oss-20b: https://groq.com/pricing (self-serve LLM catalog)
"""

# USD per 1,000 tokens. (input, output) - most providers charge
# differently for prompt tokens vs. generated tokens.
PRICING = {
    "gpt-4o-mini": {
        "input_per_1k": 0.00015,
        "output_per_1k": 0.0006,
    },
    "openai/gpt-oss-20b": {
        "input_per_1k": 0.000075,
        "output_per_1k": 0.0003,
    },
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Returns estimated USD cost for one call. Raises KeyError with a
    clear message if `model` isn't in the PRICING table yet."""
    if model not in PRICING:
        raise KeyError(
            f"No pricing entry for model {model!r}. Add it to PRICING in "
            f"token_analysis/pricing.py before estimating its cost."
        )
    rates = PRICING[model]
    return (
        (input_tokens / 1000) * rates["input_per_1k"]
        + (output_tokens / 1000) * rates["output_per_1k"]
    )