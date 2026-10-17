"""Follow-up experiments. Run one at a time; every run is resumable.

    python run_followups.py noise      # baseline run 2 (300 questions)  -> noise floor
    python run_followups.py rewrite    # rewrite, cross-encoder, 0.5 (first 50 questions)
    python run_followups.py nltk       # cross-encoder, 0.5, nltk splitter (300 questions)

Settings match the earlier SQuAD runs (same sample file, backend and window size).
Then rescore each output with evaluation/rescore_results.py before analysing it.
"""

import sys

from evaluation.run_experiment import run_experiment

R = "evaluation/results"
COMMON = dict(sample_path="data/squad_300.json", backend="groq", compression_window_size=2)

RUNS = {
    "noise": dict(n_samples=300, compression_enabled=False,
                  out_path=f"{R}/experiment_baseline_300_run2.csv"),
    "rewrite": dict(n_samples=50, compression_enabled=True, compression_method="rewrite",
                    compression_scorer="cross_encoder", compression_level=0.5,
                    out_path=f"{R}/experiment_rewrite_cross_encoder_0.5_50.csv"),
    "nltk": dict(n_samples=300, compression_enabled=True, compression_method="selective",
                 compression_scorer="cross_encoder", compression_level=0.5, compression_splitter="nltk",
                 out_path=f"{R}/experiment_selective_cross_encoder_0.5_300_nltk.csv"),
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in RUNS:
        raise SystemExit(__doc__)
    run_experiment(**COMMON, **RUNS[sys.argv[1]])