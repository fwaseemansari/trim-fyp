"""
Unit test for LLMClient.generate()'s latency measurement.

Runs 3 sample prompts against both backends and writes the results to
docs/latency_baseline.md so there's a dated record of how fast each
backend was during Week 1 (useful later for comparing against the
compression-module's latency once it's inserted into the pipeline).

Run directly: python -m pipeline.test_latency
(or via pytest, which just checks the shape of the returned dict —
it does NOT re-run the live API calls a second time.)
"""

import datetime

from pipeline.llm_clients import LLMClient

SAMPLE_PROMPTS = [
    "Say hello world in one short sentence.",
    "What is the capital of France? Answer in one word.",
    "Summarize the plot of Cinderella in two sentences.",
]

BACKENDS = ["groq"]  # add "openai" back once an OpenAI key with credits is set


def run_latency_baseline() -> list[dict]:
    """Runs every sample prompt against every backend once and returns
    a flat list of result rows."""
    client = LLMClient()
    rows = []

    for backend in BACKENDS:
        for prompt in SAMPLE_PROMPTS:
            result = client.generate(prompt, backend=backend)
            rows.append(
                {
                    "backend": backend,
                    "prompt": prompt,
                    "latency_ms": round(result["latency_ms"], 1),
                    "response_preview": result["response"][:60].replace("\n", " "),
                }
            )
    return rows


def write_latency_report(rows: list[dict], path: str = "docs/latency_baseline.md") -> None:
    today = datetime.date.today().isoformat()
    lines = [f"# Latency Baseline — {today}\n", "| Backend | Prompt | Latency (ms) | Response preview |",
              "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['backend']} | {r['prompt']} | {r['latency_ms']} | {r['response_preview']} |")

    # encoding="utf-8" is explicit here because Windows' default open()
    # encoding is cp1252, which mangles the em dash (—) above into
    # garbled characters when the file is later read back or viewed
    # in a different tool.
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


# --- pytest test (checks shape/contract, not live numbers) ---

def test_generate_returns_response_and_latency():
    """Confirms generate() returns a dict with the two expected keys
    and that latency_ms is a positive number. Requires valid API keys
    in .env to actually run (it makes one real Groq call)."""
    client = LLMClient()
    result = client.generate("Say hi in one word.", backend="groq")

    assert "response" in result
    assert "latency_ms" in result
    assert isinstance(result["response"], str)
    assert result["latency_ms"] > 0


if __name__ == "__main__":
    results = run_latency_baseline()
    write_latency_report(results)
    for r in results:
        print(f"[{r['backend']}] {r['latency_ms']} ms — {r['prompt']}")
    print("\nLogged to docs/latency_baseline.md")
