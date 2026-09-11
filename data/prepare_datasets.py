"""
Downloads SQuAD v2 and CNN/DailyMail via HuggingFace `datasets`, samples
a small subset of each, and saves to local JSON — so nobody on the team
re-downloads the full datasets repeatedly.

NOTE (stand-in flag): Dania owns this per the plan (Tue task). Run this
once yourself; the output JSONs are what Thu's baseline run and Week 2's
compression experiments read from.

WARNING: first run downloads real data from the HuggingFace Hub —
SQuAD v2 is a few hundred MB, CNN/DailyMail's train split is a few GB
(only used here to sample from, not stored in full). This needs a real
internet connection and will take a few minutes depending on your
connection — kick it off and let it run in the background rather than
expecting it instantly.
"""

import json
import os
import random

from datasets import load_dataset

random.seed(42)  # reproducible sampling


def prepare_squad(n_samples: int = 100, out_path: str = "data/squad_sample.json") -> None:
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    samples = []
    for i in indices:
        row = ds[i]
        # squad_v2 has some unanswerable questions (empty answers list) —
        # skip those for now, since Week 1's baseline assumes an
        # extractable answer exists.
        if not row["answers"]["text"]:
            continue
        samples.append(
            {
                "id": row["id"],
                "question": row["question"],
                "context": row["context"],
                "answer": row["answers"]["text"][0],
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} SQuAD v2 samples to {out_path}")


def prepare_cnn(n_samples: int = 50, out_path: str = "data/cnn_sample.json") -> None:
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    samples = []
    for i in indices:
        row = ds[i]
        samples.append(
            {
                "id": row["id"],
                "article": row["article"],
                "highlights": row["highlights"],
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} CNN/DailyMail samples to {out_path}")


if __name__ == "__main__":
    prepare_squad(n_samples=100)
    prepare_cnn(n_samples=50)
