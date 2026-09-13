"""
tests/test_token_analyzer.py

Thorough manual test for the Token Analysis Module (Mon+Tue tasks):
- count_tokens() behaves sanely across normal, edge, and unusual inputs
- TokenAnalyzer.analyze() produces internally consistent results
- Same checks repeated across both backends (gpt-4o-mini, openai/gpt-oss-20b)

Uses plain assert statements for now (readable, no extra dependency).
Faiqa's Friday pytest setup can absorb these into real pytest tests later
with minimal changes - these already live in tests/, and each check below
maps cleanly to one pytest test function.
"""

from token_analysis.counter import TokenAnalyzer, count_tokens

MODELS = ["gpt-4o-mini", "openai/gpt-oss-20b"]

analyzer = TokenAnalyzer()

passed = 0
failed = 0


def check(description, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {description}")
        passed += 1
    else:
        print(f"  FAIL: {description}")
        failed += 1


# --- Test 1: normal prompt/response pairs, multiple examples ---
print("Test 1: Normal prompt/response pairs")
pairs = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Explain gravity in one sentence.",
     "Gravity is the force that attracts objects with mass toward each other."),
    ("Write a haiku about autumn.",
     "Leaves drift to the ground\nGolden light through empty trees\nAutumn whispers rest"),
]

for model in MODELS:
    print(f" Model: {model}")
    for prompt, response in pairs:
        result = analyzer.analyze(prompt, response, model)
        check(
            f"total_tokens = input + output ({prompt[:30]!r})",
            result["total_tokens"] == result["input_tokens"] + result["output_tokens"],
        )
        check(
            f"cost_usd is positive ({prompt[:30]!r})",
            result["cost_usd"] > 0,
        )

# --- Test 2: edge case - empty strings ---
print("\nTest 2: Empty string edge case")
for model in MODELS:
    result = analyzer.analyze("", "", model)
    check(f"empty prompt+response -> 0 tokens ({model})", result["total_tokens"] == 0)
    check(f"empty prompt+response -> 0 cost ({model})", result["cost_usd"] == 0)

# --- Test 3: edge case - very long input (stress test) ---
print("\nTest 3: Long input")
long_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~1800 words
for model in MODELS:
    result = analyzer.analyze(long_text, "OK.", model)
    check(
        f"long input produces >500 tokens ({model})",
        result["input_tokens"] > 500,
    )

# --- Test 4: unknown model raises a clear error, doesn't crash silently ---
print("\nTest 4: Unknown model handling")
try:
    count_tokens("test", "not-a-real-model")
    check("unknown model raises ValueError", False)
except ValueError:
    check("unknown model raises ValueError", True)

# --- Test 5: compare() gives sensible reduction % for a "compressed" pair ---
print("\nTest 5: compare() sanity check")
baseline = analyzer.analyze(
    "This is a very long and unnecessarily verbose original prompt that says very little.",
    "Response A.",
    "gpt-4o-mini",
)
treatment = analyzer.analyze("Short prompt.", "Response A.", "gpt-4o-mini")
comparison = analyzer.compare(baseline, treatment)
check(
    "compare() shows positive token reduction for a shorter treatment prompt",
    comparison["token_reduction_pct"] > 0,
)

# --- Summary ---
print(f"\n{'='*40}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*40}")