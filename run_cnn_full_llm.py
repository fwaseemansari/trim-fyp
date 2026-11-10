"""CNN/DailyMail LLM runs on the fixed 1,000-article subset (same file for the whole team).

    python -m data.prepare_datasets cnn-full     # once
    python run_cnn_full_llm.py

Needs GROQ_API_KEYS=key1,key2,... (one line) in .env or the environment.
Runs the 9 experiments below in this order. Resumable: finished experiments are skipped.
If every Groq key hits its daily limit, the script stops; rerun after the reset to resume.
Compare with: python -m evaluation.summarize_cnn <baseline csv> <other csvs...>
"""

import sys
import traceback

from evaluation.run_cnn_experiment import run_cnn_experiment
from pipeline.groq_key_pool import DailyLimitExhausted

R = "evaluation/results"
SAMPLE = "data/cnn_subset_1000.json"
BACKEND = "groq"
DELAY = 3.0

# (name, description, kwargs) in run order
EXPERIMENTS = [
    ("none", "No compression", dict(compression_method="none")),
    ("selective_cross_encoder_0.5", "Selective, 0.5",
     dict(compression_method="selective", compression_scorer="cross_encoder", compression_level=0.5)),
    ("lead_0.5", "Lead, 0.5", dict(compression_method="lead", compression_level=0.5)),
    ("selective_cross_encoder_0.3", "Selective, 0.3",
     dict(compression_method="selective", compression_scorer="cross_encoder", compression_level=0.3)),
    ("lead_0.3", "Lead, 0.3", dict(compression_method="lead", compression_level=0.3)),
    ("selective_cross_encoder_0.7", "Selective, 0.7",
     dict(compression_method="selective", compression_scorer="cross_encoder", compression_level=0.7)),
    ("lead_0.7", "Lead, 0.7", dict(compression_method="lead", compression_level=0.7)),
    ("selective_tfidf_0.5", "TF-IDF",
     dict(compression_method="selective", compression_scorer="tfidf", compression_level=0.5)),
    ("selective_bi_encoder_0.5", "Bi-encoder",
     dict(compression_method="selective", compression_scorer="bi_encoder", compression_level=0.5)),
]


def main():
    total = len(EXPERIMENTS)
    failed = []
    for n, (name, desc, kw) in enumerate(EXPERIMENTS, 1):
        print(f"\n=== Experiment {n}/{total}: {name} ({desc}) ===", flush=True)
        try:
            run_cnn_experiment(sample_path=SAMPLE, n_samples=None, backend=BACKEND, delay_seconds=DELAY,
                               out_path=f"{R}/cnn_sub_{name}.csv", **kw)
        except Exception:  # an unexpected error in one experiment must not stop the overnight run
            traceback.print_exc()
            failed.append(name)
            print(f"[!] Experiment {name} failed; moving on to the next one. "
                  "Rerun later to resume it.", flush=True)
    if failed:
        print(f"\nFinished, but these experiments did not complete: {failed}. Rerun to resume them.", flush=True)
    else:
        print("\nAll experiments finished.", flush=True)


try:
    main()
except DailyLimitExhausted:
    print("\nAll Groq API keys have reached their daily token/request limit. "
          "Progress is saved; rerun after the daily reset to resume.", flush=True)
    sys.exit(1)