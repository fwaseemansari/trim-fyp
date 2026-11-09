"""CNN/DailyMail summarisation experiment (LLM stage). Replaces evaluation/run_cnn_experiment.py.

Compress each article, ask the LLM for a 3-sentence summary, score it against the
reference highlights with ROUGE-L. Output columns keep the names that
evaluation.summarize_cnn expects (compression_method, compression_level,
token_reduction_pct, compression_ratio, information_retention, rougeL_f1, latency_ms).

    python -m evaluation.run_cnn_experiment        # edit the call at the bottom
    python run_cnn_full_llm.py                     # the planned set of runs

compression_method: "none", "lead", "extractive", or "selective" (uses compression_scorer).
Resumable: reruns skip sample ids already in out_path.
"""

import csv
import json
import os
import time

from rouge_score import rouge_scorer

from evaluation.cnn_common import compress_article, make_compressor
from evaluation.metrics import (
    compression_ratio,
    cost_reduction_pct,
    information_retention_score,
    token_reduction_pct,
)
from pipeline.config import model_for_backend
from pipeline.llm_clients import LLMClient
from token_analysis.counter import count_tokens
from token_analysis.pricing import estimate_cost

SUMMARY_PROMPT = (
    "Article:\n{article}\n\n"
    "Summarize the article in 3 short sentences. Reply with only the summary."
)


def _load_done(out_path):
    if not out_path or not os.path.exists(out_path):
        return set(), None
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        return {r["sample_id"] for r in reader}, fields


def run_cnn_experiment(
    sample_path: str = "data/cnn_subset_1000.json",
    n_samples: int | None = None,
    compression_method: str = "selective",
    compression_scorer: str = "cross_encoder",
    compression_level: float = 0.5,
    backend: str = "groq",
    out_path: str | None = None,
    delay_seconds: float = 1.5,
    compression_window_size: int = 2,
    compression_splitter: str = "regex",
    resume: bool = True,
) -> list:
    with open(sample_path, encoding="utf-8") as f:
        samples = json.load(f)[:n_samples]

    model = model_for_backend(backend)
    client = LLMClient()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    compressed_run = compression_method != "none"
    if compression_method in ("selective", "extractive"):
        compressor = make_compressor(compression_method, scorer=compression_scorer, backend=backend,
                                     window_size=compression_window_size, splitter=compression_splitter)
    else:
        compressor = None
    scorer_label = compression_scorer if compression_method == "selective" else \
        ("tfidf" if compression_method == "extractive" else "none")
    label = f"selective_{compression_scorer}" if compression_method == "selective" else compression_method
    kind = "full" if compression_method == "none" else compression_method

    done, existing_fields = _load_done(out_path) if resume else (set(), None)
    print(f"{len(samples)} samples, {len(done)} done, {len(samples) - len(done)} pending", flush=True)

    writer = None
    out_file = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_file = open(out_path, "a" if done else "w", newline="", encoding="utf-8")
        if done:
            writer = csv.DictWriter(out_file, fieldnames=existing_fields, extrasaction="ignore")

    rows = []
    try:
        for i, sample in enumerate(samples):
            if sample["id"] in done:
                continue
            article, reference = sample["article"], sample["highlights"]
            used_level = compression_level if compressed_run else 1.0
            compressed, orig_tok, comp_tok = compress_article(
                kind, compressor, article, used_level, model, compression_splitter)

            result = client.generate(SUMMARY_PROMPT.format(article=compressed), backend=backend)
            summary = (result.response or "").strip()
            score = rouge.score(reference, summary)["rougeL"]

            row = {
                "sample_id": sample["id"],
                "compression_enabled": compressed_run,
                "compression_method": label if compressed_run else "none",
                "compression_scorer": scorer_label,
                "compression_level": used_level,
                "backend": backend,
                "context_tokens_original": orig_tok,
                "context_tokens_compressed": comp_tok,
                "token_reduction_pct": token_reduction_pct(orig_tok, comp_tok),
                "compression_ratio": compression_ratio(orig_tok, comp_tok),
                "cost_reduction_pct": cost_reduction_pct(
                    estimate_cost(model, orig_tok, 0), estimate_cost(model, comp_tok, 0)),
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
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(samples)}", flush=True)
            if i < len(samples) - 1:
                time.sleep(delay_seconds)
    finally:
        if out_file:
            out_file.close()

    if out_path and rows:
        print(f"Saved {len(rows)} new results to {out_path}")
    return rows


if __name__ == "__main__":
    run_cnn_experiment(
        compression_method="none",
        out_path="evaluation/results/cnn_sub_none.csv",
    )
