"""
General-purpose experiment runner: loads a dataset sample, runs it
through the pipeline with a chosen config (compression on/off, method,
level), dumps a results CSV.

Week 2 version: records, per sample, the compression metrics of the
CONTEXT (token reduction %, compression ratio, cost reduction %,
information retention) plus the pipeline's own tokens/cost/latency, and
whether the gold answer survived compression. The LLM's raw answer and
the gold answer are saved too, so Week 3 can add EM/F1/ROUGE-L without
re-running any API calls.

Run from the repo root as a module:
    python -m evaluation.run_experiment
"""

import csv
import json
import os
import time

from pipeline.pipeline import run
from prompt_compression.compressor import compress
from token_analysis.counter import TokenAnalyzer, count_tokens
from token_analysis.pricing import estimate_cost
from evaluation.quality_metrics import exact_match, f1_score, contains_answer
from evaluation.metrics import (
    token_reduction_pct,
    compression_ratio,
    cost_reduction_pct,
    information_retention_score,
)

# Which tokenizer/pricing model belongs to which backend.
BACKEND_MODEL = {
    "groq": "openai/gpt-oss-20b",
    "openai": "gpt-4o-mini",
}


def run_experiment(
    sample_path: str = "data/squad_sample.json",
    n_samples: int = 20,
    compression_enabled: bool = True,
    compression_method: str = "extractive",
    compression_level: float = 0.5,
    backend: str = "groq",
    out_path: str = None,
    delay_seconds: float = 1.5,
) -> list:
    with open(sample_path, encoding="utf-8") as f:
        samples = json.load(f)[:n_samples]

    model = BACKEND_MODEL.get(backend, "gpt-4o-mini")
    analyzer = TokenAnalyzer(
        log_path=f"evaluation/logs/experiment_{compression_method}_{compression_level}.csv"
    )

    writer = None
    out_file = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_file = open(out_path, "w", newline="", encoding="utf-8")

    rows = []
    try:
        for i, sample in enumerate(samples):
            original_context = sample["context"]
            gold_answer = sample.get("answer", "")
            # Use every gold answer if the sample file has them, otherwise the single one.
            golds = sample.get("answers") or ([gold_answer] if gold_answer else [])

            result = run(
                query=sample["question"],
                context=original_context,
                compression_enabled=compression_enabled,
                compression_method=compression_method,
                compression_level=compression_level,
                backend=backend,
                analyzer=analyzer,
            )

            # Use the compressed context the pipeline actually sent, if it
            # returns one. Otherwise recompute it (free and identical for the
            # deterministic "extractive" method; "llm" would cost an extra call).
            compressed_context = original_context
            if compression_enabled:
                compressed_context = result.get("compressed_context") or result.get("compressed_text")
                if compressed_context is None:
                    if compression_method == "llm":
                        print(f"[warn] sample {sample.get('id')}: pipeline did not return its compressed "
                              f"context, so the LLM-compressed text is recomputed and may differ.")
                        from pipeline.llm_clients import LLMClient
                        compressed_context = compress(
                            original_context, query=sample["question"], method="llm",
                            level=compression_level, llm_client=LLMClient(), backend=backend,
                        )
                    else:
                        compressed_context = compress(
                            original_context, query=sample["question"],
                            method=compression_method, level=compression_level,
                        )

            # Context-level compression metrics (what the proposal's metrics 1-4 measure).
            orig_tok = count_tokens(original_context, model)
            comp_tok = count_tokens(compressed_context, model)
            orig_cost = estimate_cost(model, orig_tok, 0)
            comp_cost = estimate_cost(model, comp_tok, 0)

            # An empty gold answer means a SQuAD v2 unanswerable question;
            # "" is a substring of everything, so skip the check for those.
            if gold_answer:
                answer_present = gold_answer.lower() in compressed_context.lower()
            else:
                answer_present = None

            llm_response = result.get("response", "") or ""
            # Unanswerable (no gold) questions are not scored: the LLM always answers something.
            scored = bool(golds)

            row = {
                "sample_id": sample["id"],
                "question": sample["question"],
                "compression_enabled": compression_enabled,
                "compression_method": compression_method if compression_enabled else "none",
                "compression_level": compression_level if compression_enabled else 1.0,
                "backend": backend,
                "answer_still_present": answer_present,
                "context_tokens_original": orig_tok,
                "context_tokens_compressed": comp_tok,
                "token_reduction_pct": token_reduction_pct(orig_tok, comp_tok),
                "compression_ratio": compression_ratio(orig_tok, comp_tok),
                "cost_reduction_pct": cost_reduction_pct(orig_cost, comp_cost),
                "information_retention": information_retention_score(original_context, compressed_context),
                "total_tokens": result["total_tokens"],
                "cost_usd": result["cost_usd"],
                "latency_ms": result["latency_ms"],
                "em": exact_match(llm_response, golds) if scored else None,
                "f1": round(f1_score(llm_response, golds), 4) if scored else None,
                "answer_in_response": contains_answer(llm_response, golds) if scored else None,
                "gold_answer": gold_answer,
                "llm_response": llm_response,
            }
            rows.append(row)

            # Write each row as soon as it exists, so a rate-limit crash
            # halfway through does not lose the samples already finished.
            if out_file:
                if writer is None:
                    writer = csv.DictWriter(out_file, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                out_file.flush()

            if i < len(samples) - 1:
                time.sleep(delay_seconds)  # stay under Groq's free-tier TPM limit
    finally:
        if out_file:
            out_file.close()

    if out_path and rows:
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