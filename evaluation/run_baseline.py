"""
Runs the un-compressed, un-managed pipeline over data/squad_sample.json
(from data/prepare_datasets.py) to produce baseline numbers: avg
tokens/query, avg cost, avg latency. Everything else built in Week 2-4
gets compared against these numbers.

Requires data/squad_sample.json to exist first — run
`python -m data.prepare_datasets` before this.

WARNING: makes one real Groq API call per sample (100 by default).
At ~0.5-0.7s each that's roughly 1-2 minutes total on Groq's free
tier — run it and let it finish, don't expect instant output.
"""

import json
import statistics
import time

from pipeline.pipeline import run
from token_analysis.counter import TokenAnalyzer

def run_baseline(sample_path: str = "data/squad_sample.json", out_path: str = "evaluation/baseline_results.json", delay_seconds: float = 1.5) -> None:
    with open(sample_path, encoding="utf-8") as f:
        samples = json.load(f)

    analyzer = TokenAnalyzer(log_path="evaluation/logs/baseline_log.csv")

    tokens, costs, latencies = [], [], []
    for i, sample in enumerate(samples):
        result = run(
            query=sample["question"],
            context=sample["context"],
            compression_enabled=False,  # baseline = no compression, per the plan
            backend="groq",
            analyzer=analyzer,
        )
        tokens.append(result["total_tokens"])
        costs.append(result["cost_usd"])
        latencies.append(result["latency_ms"])
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(samples)} samples done")
        if i < len(samples) - 1:
            time.sleep(delay_seconds)  # pace calls to stay under Groq's free-tier TPM limit

    summary = {
        "n_samples": len(samples),
        "avg_tokens_per_query": round(statistics.mean(tokens), 1),
        "avg_cost_per_query_usd": round(statistics.mean(costs), 6),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "total_cost_usd": round(sum(costs), 4),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBaseline saved to {out_path}:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_baseline()
