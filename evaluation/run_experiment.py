"""
General-purpose experiment runner: loads a dataset sample, runs it
through the pipeline with a chosen config (compression on/off, method,
level), dumps a results CSV.

NOTE (stand-in flag): Easha owns this per the plan (Fri Week 2).
Faiqa's evaluation/run_compression_sweep.py (same day, her task) is a
thin wrapper that calls this 3 times with different compression
levels — built as a wrapper rather than duplicating this loop, so the
sweep and any future one-off experiment both go through the same
tested code path.
"""

import csv
import json
import os
import time 

from pipeline.pipeline import run
from token_analysis.counter import TokenAnalyzer
from evaluation.metrics import full_comparison


def run_experiment(
    sample_path: str = "data/squad_sample.json",
    n_samples: int = 20,
    compression_enabled: bool = True,
    compression_method: str = "extractive",
    compression_level: float = 0.5,
    backend: str = "groq",
    out_path: str = None,
    delay_seconds: float = 1.5
) -> list:
    with open(sample_path, encoding="utf-8") as f:
        samples = json.load(f)[:n_samples]

    analyzer = TokenAnalyzer(log_path=f"evaluation/logs/experiment_{compression_method}_{compression_level}.csv")

    rows = []
    for i, sample in enumerate(samples):
        original_context = sample["context"]

        result = run(
            query=sample["question"],
            context=original_context,
            compression_enabled=compression_enabled,
            compression_method=compression_method,
            compression_level=compression_level,
            backend=backend,
            analyzer=analyzer,
        )

        # Answer-still-present check: crude exact-match string search
        # (per the plan's Tue Week 2 note — "proper QA-accuracy scoring
        # comes Week 3"). Checks the compressed CONTEXT, not the LLM's
        # final answer, since that's what's available at this stage.
        answer_present = sample["answer"].lower() in original_context.lower()

        metrics = {}
        if compression_enabled:
            from prompt_compression.compressor import compress
            from pipeline.llm_clients import LLMClient
            compressed_context = compress(
                original_context, query=sample["question"],
                method=compression_method, level=compression_level,
                llm_client=LLMClient() if compression_method == "llm" else None,
            )
            answer_present = sample["answer"].lower() in compressed_context.lower()

        rows.append({
            "sample_id": sample["id"],
            "question": sample["question"],
            "compression_enabled": compression_enabled,
            "compression_method": compression_method if compression_enabled else "none",
            "compression_level": compression_level if compression_enabled else 1.0,
            "answer_still_present": answer_present,
            "total_tokens": result["total_tokens"],
            "cost_usd": result["cost_usd"],
            "latency_ms": result["latency_ms"],
        })
        
        if i < len(samples) - 1:
            time.sleep(delay_seconds)  # pace calls to stay under Groq's free-tier TPM limit

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} results to {out_path}")

    return rows


if __name__ == "__main__":
    run_experiment(
        n_samples=20,
        compression_enabled=True,
        compression_method="extractive",
        compression_level=0.5,
        out_path="evaluation/results/experiment_extractive_0.5.csv",
    )
