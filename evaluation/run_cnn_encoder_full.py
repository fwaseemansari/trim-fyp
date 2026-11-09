"""Encoder-only run over the FULL CNN/DailyMail validation split. No LLM, no API keys.

For every article: full text (ceiling), Lead-N baseline, and each scorer at each
level. Records token reduction and how much of the reference highlights the kept
text still covers (ROUGE-1 / ROUGE-L recall of the highlights against the kept text).

    python -m data.prepare_datasets cnn-full     # once -> data/cnn_full.json
    python -m evaluation.run_cnn_encoder_full [path/to/articles.json]

Resumable: reruns skip (article, level) pairs already in each output CSV.
"""

import csv
import json
import os
import sys
import time

import pandas as pd
from rouge_score import rouge_scorer

from evaluation.cnn_common import compress_article, make_compressor
from pipeline.config import model_for_backend

SAMPLE_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/cnn_full.json"
TAG = os.path.splitext(os.path.basename(SAMPLE_PATH))[0]
OUT_DIR = "evaluation/results"
BACKEND = "groq"                  # decides which tokenizer counts the budget
LEVELS = (0.3, 0.5, 0.7)
SCORERS = ("tfidf", "bi_encoder", "cross_encoder")
# (name, kind, scorer, levels)
VARIANTS = [("full", "full", None, (1.0,)), ("lead", "lead", None, LEVELS)] + \
           [(s, "selective", s, LEVELS) for s in SCORERS]
FIELDS = ["sample_id", "variant", "level", "tokens_original", "tokens_compressed",
          "token_reduction_pct", "rouge1_recall", "rougeL_recall", "compress_ms"]


def _path(name):
    return f"{OUT_DIR}/cnn_encoder_{TAG}_{name}.csv"


def _done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["sample_id"], r["level"]) for r in csv.DictReader(f)}


def run_variant(samples, name, kind, scorer, levels, model, rouge):
    path = _path(name)
    done = _done(path)
    compressor = make_compressor("selective", scorer=scorer, backend=BACKEND) if kind == "selective" else None
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{name}: {len(done)} rows already done", flush=True)
    t0 = time.time()
    with open(path, "a" if done else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for i, s in enumerate(samples, 1):
            for level in levels:
                if (s["id"], str(level)) in done:
                    continue
                t1 = time.perf_counter()
                text, orig, comp = compress_article(kind, compressor, s["article"], level, model)
                ms = (time.perf_counter() - t1) * 1000
                sc = rouge.score(s["highlights"], text)   # target = reference, prediction = kept text
                w.writerow({
                    "sample_id": s["id"], "variant": name, "level": level,
                    "tokens_original": orig, "tokens_compressed": comp,
                    "token_reduction_pct": round((orig - comp) / orig * 100, 2) if orig else 0.0,
                    "rouge1_recall": round(sc["rouge1"].recall, 4),
                    "rougeL_recall": round(sc["rougeL"].recall, 4),
                    "compress_ms": round(ms, 2),
                })
            if i % 200 == 0:
                f.flush()
                print(f"  {name}: {i}/{len(samples)} ({time.time() - t0:.0f}s)", flush=True)


def summarize():
    rows = []
    for name, _, _, levels in VARIANTS:
        if not os.path.exists(_path(name)):
            continue
        df = pd.read_csv(_path(name))
        for level, g in df.groupby("level"):
            rows.append({"variant": name, "level": level, "n": len(g),
                         "token_reduction_%": round(g["token_reduction_pct"].mean(), 1),
                         "rouge1_recall": round(g["rouge1_recall"].mean(), 3),
                         "rougeL_recall": round(g["rougeL_recall"].mean(), 3)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(f"{OUT_DIR}/cnn_encoder_{TAG}_summary.csv", index=False)


if __name__ == "__main__":
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"{len(samples)} articles")
    model = model_for_backend(BACKEND)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    for name, kind, scorer, levels in VARIANTS:
        run_variant(samples, name, kind, scorer, levels, model, rouge)
    summarize()
