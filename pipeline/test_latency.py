"""
Unit test for LLMClient.generate()'s latency measurement.

Runs 3 sample prompts against Groq and logs the results via
TokenAnalyzer, into evaluation/logs/latency_baseline.csv.

REVISED: this used to write its own Markdown table to
docs/latency_baseline.md via a custom writer function. Once
TokenAnalyzer.log_run() existed (Week 2) and already logged latency_ms
per call to CSV, that meant two separate logging mechanisms doing an
overlapping job — inconsistent, and the kind of thing worth fixing once
noticed rather than leaving as-is. Now this reuses TokenAnalyzer like
every other script does, so there's exactly ONE logging path in the
whole codebase. docs/latency_baseline.md is retired — delete it, its
data now lives in evaluation/logs/latency_baseline.csv going forward.

Run directly: python -m pipeline.test_latency
(or via pytest, which just checks the shape of the returned dict —
it does NOT re-run the live API calls a second time.)
"""

from pipeline.llm_clients import LLMClient
from token_analysis.counter import TokenAnalyzer

SAMPLE_PROMPTS = [
    "Say hello world in one short sentence.",
    "What is the capital of France? Answer in one word.",
    "Summarize the plot of Cinderella in two sentences.",
]

BACKENDS = ["groq"]  # add "openai" back once an OpenAI key with credits is set
MODEL_BY_BACKEND = {"groq": "openai/gpt-oss-20b", "openai": "gpt-4o-mini"}


def run_latency_baseline() -> list[dict]:
    """Runs every sample prompt against every backend once, logs each
    via TokenAnalyzer (same CSV logging path as the rest of the
    pipeline), and returns the analysis dicts for printing."""
    client = LLMClient()
    analyzer = TokenAnalyzer(log_path="evaluation/logs/latency_baseline.csv")
    rows = []

    for backend in BACKENDS:
        for prompt in SAMPLE_PROMPTS:
            result = client.generate(prompt, backend=backend)
            analysis = analyzer.log_run(
                prompt=prompt,
                response=result.response,
                backend=backend,
                model=MODEL_BY_BACKEND[backend],
                latency_ms=result.latency_ms,
            )
            rows.append({"backend": backend, "prompt": prompt, "latency_ms": round(result.latency_ms, 1), **analysis})
    return rows


# --- pytest test (checks shape/contract, not live numbers) ---

def test_generate_returns_response_and_latency():
    """Confirms generate() returns an LLMResponse with the two expected
    fields and that latency_ms is a positive number. Requires valid API
    keys in .env to actually run (it makes one real Groq call)."""
    client = LLMClient()
    result = client.generate("Say hi in one word.", backend="groq")

    assert isinstance(result.response, str)
    assert len(result.response) > 0
    assert result.latency_ms > 0


if __name__ == "__main__":
    results = run_latency_baseline()
    for r in results:
        print(f"[{r['backend']}] {r['latency_ms']} ms, {r['total_tokens']} tokens — {r['prompt']}")
    print("\nLogged to evaluation/logs/latency_baseline.csv")
