"""Task accuracy with compression vs without (Week 3, metric 5).

Usage (repo root):
  python -m evaluation.compare_accuracy BASELINE.csv TREATMENT1.csv [TREATMENT2.csv ...]

BASELINE is a run_experiment CSV made with compression_enabled=False.
Samples are matched by sample_id, so the comparison is paired.
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["f1", "em", "answer_in_response"]


def compare(baseline_path, treatment_path):
    b = pd.read_csv(baseline_path)
    t = pd.read_csv(treatment_path)
    m = b.merge(t, on="sample_id", suffixes=("_base", "_comp")).dropna(subset=["f1_base", "f1_comp"])
    row = {
        "treatment": f"{t['compression_method'].iloc[0]}@{t['compression_level'].iloc[0]}",
        "n_paired": len(m),
    }
    for c in COLS:
        row[f"{c}_base"] = round(m[f"{c}_base"].mean(), 4)
        row[f"{c}_comp"] = round(m[f"{c}_comp"].mean(), 4)
        row[f"{c}_drop"] = round(row[f"{c}_base"] - row[f"{c}_comp"], 4)
    try:
        from scipy.stats import ttest_rel
        row["f1_paired_p"] = round(float(ttest_rel(m["f1_base"], m["f1_comp"]).pvalue), 4)
    except Exception:
        row["f1_paired_p"] = None
    row["token_reduction_pct"] = round(t["token_reduction_pct"].mean(), 2)
    return row


if __name__ == "__main__":
    base, treatments = sys.argv[1], sys.argv[2:]
    out = pd.DataFrame([compare(base, t) for t in treatments])
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    dest = Path(base).parent / "accuracy_comparison.csv"
    out.to_csv(dest, index=False)
    print(f"Saved {dest}")