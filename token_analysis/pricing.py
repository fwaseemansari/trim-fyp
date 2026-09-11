"""
Hardcoded per-1K-token pricing for cost estimation. These are published
rates, not measured — update if OpenAI/Groq change pricing.

NOTE (stand-in flag): Easha owns this module per the plan. This is a
placeholder so the pipeline has real cost numbers today — hand off /
replace once she starts her actual work.
"""

# USD per 1,000 tokens. (input, output) — most providers charge
# differently for prompt tokens vs. generated tokens.
PRICING = {
    "gpt-4o-mini": {
        "input_per_1k": 0.00015,
        "output_per_1k": 0.0006,
    },
    # Groq's free tier has no per-token dollar cost today, but we keep a
    # nonzero placeholder rate so cost-reduction % comparisons against
    # OpenAI still mean something, rather than dividing by zero. This
    # mirrors Groq's paid on-demand pricing for a comparable 20B model.
    "openai/gpt-oss-20b": {
        "input_per_1k": 0.00010,
        "output_per_1k": 0.00010,
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
