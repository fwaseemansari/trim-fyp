"""Reproducible SQuAD experiment runner for compression methods."""

import csv
import json
import os
import time

from pipeline.pipeline import run
from token_analysis.counter import TokenAnalyzer, count_tokens
from token_analysis.pricing import estimate_cost
from evaluation.quality_metrics import exact_match, f1_score, contains_answer
from evaluation.metrics import (
    token_reduction_pct,
    compression_ratio,
    cost_reduction_pct,
    information_retention_score,
)

BACKEND_MODEL = {
    "groq": "openai/gpt-oss-20b",
    "openai": "gpt-4o-mini",
}

def _load_done(out_path):
    """Sample ids already saved in out_path, plus its header (for appending)."""
    if not out_path or not os.path.exists(out_path):
        return set(), None
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        done = {r["sample_id"] for r in reader}
    return done, fieldnames


def run_experiment(
    sample_path: str = "data/squad_300.json",
    n_samples: int = 20,
    compression_enabled: bool = True,
    compression_method: str = "extractive",
    compression_level: float = 0.5,
    backend: str = "groq",
    compression_scorer: str = "tfidf",
    compression_window_size: int = 2,
    out_path: str | None = None,
    delay_seconds: float = 1.5,
    resume: bool = True,
) -> list:
    with open(sample_path, encoding="utf-8") as f:
        samples = json.load(f)[:n_samples]

    model = BACKEND_MODEL.get(backend, "gpt-4o-mini")
    analyzer = TokenAnalyzer(
        log_path=f"evaluation/logs/experiment_{compression_method}_{compression_scorer}_{compression_level}.csv"
    )

    done, existing_fields = _load_done(out_path) if resume else (set(), None)
    pending = [s["id"] for s in samples if s["id"] not in done]
    print(f"DIAG samples={len(samples)} done={len(done)} pending={len(pending)} first={pending[:3]}", flush=True)

    writer = None
    out_file = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_file = open(out_path, "a" if done else "w", newline="", encoding="utf-8")
        if done:
            writer = csv.DictWriter(out_file, fieldnames=existing_fields)
            print(f"Resuming: {len(done)} samples already saved, skipping them.")

    rows = []
    try:
        for i, sample in enumerate(samples):
            if sample["id"] in done:
                continue
            print(f"[{i + 1}/{len(samples)}] running {sample['id']}", flush=True)   # <- add
            original_context = sample["context"]
            gold_answer = sample.get("answer", "")
            golds = sample.get("answers") or ([gold_answer] if gold_answer else [])

            result = run(
                query=sample["question"],
                context=original_context,
                compression_enabled=compression_enabled,
                compression_method=compression_method,
                compression_level=compression_level,
                backend=backend,
                analyzer=analyzer,
                compression_scorer=compression_scorer,
                compression_window_size=compression_window_size,
            )

            # Never recompress: evaluate exactly the text produced by the
            # pipeline for this API call.
            compressed_context = result.get("compressed_context") or original_context
            compression_result = result.get("compression_result")

            orig_tok = count_tokens(original_context, model)
            comp_tok = count_tokens(compressed_context, model)
            orig_cost = estimate_cost(model, orig_tok, 0)
            comp_cost = estimate_cost(model, comp_tok, 0)

            if gold_answer:
                answer_present = gold_answer.lower() in compressed_context.lower()
            else:
                answer_present = None

            llm_response = result.get("response", "") or ""
            scored = bool(golds)

            row = {
                "sample_id": sample["id"],
                "question": sample["question"],
                "compression_enabled": compression_enabled,
                "compression_method": compression_method if compression_enabled else "none",
                "compression_scorer": compression_scorer if compression_enabled else "none",
                "compression_window_size": compression_window_size if compression_enabled else None,
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
                "answer_call_cost_usd": result["cost_usd"],
                "compression_call_cost_usd": result.get("compression_cost_usd", 0.0),
                "end_to_end_cost_usd": result.get("end_to_end_cost_usd", result["cost_usd"]),
                "compression_llm_calls": compression_result.llm_calls if compression_result else 0,
                "compression_input_tokens": compression_result.compression_input_tokens if compression_result else 0,
                "compression_output_tokens": compression_result.compression_output_tokens if compression_result else 0,
                "latency_ms": result["latency_ms"],
                "em": exact_match(llm_response, golds) if scored else None,
                "f1": round(f1_score(llm_response, golds), 4) if scored else None,
                "answer_in_response": contains_answer(llm_response, golds) if scored else None,
                "gold_answer": gold_answer,
                "llm_response": llm_response,
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
    run_experiment(
        n_samples=300,
        compression_enabled=True,
        compression_method="selective",
        compression_scorer="bi_encoder",
        compression_level=0.5,
        out_path="evaluation/results/experiment_selective_bi_encoder_0.5_300.csv",
    )
