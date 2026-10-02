"""
CNN/DailyMail summarization experiment: compress each article, ask the LLM
for a short summary, score the summary against the reference highlights
with ROUGE-L. Run as a module:
    python -m evaluation.run_cnn_experiment
"""

import csv
import json
import os
import time

from rouge_score import rouge_scorer

from pipeline.llm_clients import LLMClient
from prompt_compression.compressor import compress
from token_analysis.counter import count_tokens
from token_analysis.pricing import estimate_cost
from evaluation.metrics import (
    token_reduction_pct,
    compression_ratio,
    cost_reduction_pct,
    information_retention_score,
)

BACKEND_MODEL = {"groq": "openai/gpt-oss-20b", "openai": "gpt-4o-mini"}
COMPRESSION_QUERY = "Summarize the main points of this news article."
SUMMARY_PROMPT = (
    "Article:\n{article}\n\n"
    "Summarize the article in 3 short sentences. Reply with only the summary."
)


def run_cnn_experiment(
    sample_path: str = "data/cnn_sample.json",
    n_samples: int = 50,
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
    client = LLMClient()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    writer = None
    out_file = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_file = open(out_path, "w", newline="", encoding="utf-8")

    rows = []
    try:
        for i, sample in enumerate(samples):
            article = sample["article"]
            reference = sample["highlights"]

            compressed = article
            if compression_enabled:
                compressed = compress(
                    article, query=COMPRESSION_QUERY, method=compression_method,
                    level=compression_level, llm_client=client, backend=backend,
                )

            result = client.generate(SUMMARY_PROMPT.format(article=compressed), backend=backend)
            summary = (result.response or "").strip()

            orig_tok = count_tokens(article, model)
            comp_tok = count_tokens(compressed, model)
            score = scorer.score(reference, summary)["rougeL"]

            row = {
                "sample_id": sample["id"],
                "compression_enabled": compression_enabled,
                "compression_method": compression_method if compression_enabled else "none",
                "compression_level": compression_level if compression_enabled else 1.0,
                "backend": backend,
                "context_tokens_original": orig_tok,
                "context_tokens_compressed": comp_tok,
                "token_reduction_pct": token_reduction_pct(orig_tok, comp_tok),
                "compression_ratio": compression_ratio(orig_tok, comp_tok),
                "cost_reduction_pct": cost_reduction_pct(
                    estimate_cost(model, orig_tok, 0), estimate_cost(model, comp_tok, 0)
                ),
                "information_retention": information_retention_score(article, compressed),
                "latency_ms": result.latency_ms,
                "rougeL_f1": round(score.fmeasure, 4),
                "rougeL_precision": round(score.precision, 4),
                "rougeL_recall": round(score.recall, 4),
                "summary": summary,
                "reference": reference,
            }
            rows.append(row)

            if out_file:
                if writer is None:
                    writer = csv.DictWriter(out_file, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                out_file.flush()

            if i < len(samples) - 1:
                time.sleep(delay_seconds)
    finally:
        if out_file:
            out_file.close()

    if out_path and rows:
        print(f"Saved {len(rows)} results to {out_path}")
    return rows


if __name__ == "__main__":
    run_cnn_experiment(
        compression_enabled=False,
        out_path="evaluation/results/cnn_none.csv",
    )