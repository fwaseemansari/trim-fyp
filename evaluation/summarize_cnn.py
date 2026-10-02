"""
Summarize CNN/DailyMail runs: mean token reduction, retention and ROUGE-L,
plus a paired t-test of ROUGE-L against the baseline.
Usage:
    python -m evaluation.summarize_cnn evaluation/results/cnn_none.csv evaluation/results/cnn_extractive_0.5.csv
The first file is the baseline.
"""

import sys

import pandas as pd
from scipy import stats


def summarize(paths: list) -> pd.DataFrame:
    base = pd.read_csv(paths[0])
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        row = {
            "method": df["compression_method"].iloc[0],
            "level": df["compression_level"].iloc[0],
            "n": len(df),
            "token_reduction_pct": round(df["token_reduction_pct"].mean(), 1),
            "compression_ratio": round(df["compression_ratio"].mean(), 2),
            "retention": round(df["information_retention"].mean(), 3),
            "rougeL_mean": round(df["rougeL_f1"].mean(), 3),
            "rougeL_std": round(df["rougeL_f1"].std(), 3),
            "latency_ms": round(df["latency_ms"].mean()),
        }
        if p != paths[0]:
            merged = base.merge(df, on="sample_id", suffixes=("_b", "_c"))
            row["rougeL_drop"] = round(merged["rougeL_f1_b"].mean() - merged["rougeL_f1_c"].mean(), 3)
            row["p_value"] = round(stats.ttest_rel(merged["rougeL_f1_b"], merged["rougeL_f1_c"]).pvalue, 3)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = summarize(sys.argv[1:])
    print(out.to_string(index=False))
    out.to_csv("evaluation/results/cnn_summary.csv", index=False)
    out.to_latex("evaluation/results/cnn_summary.tex", index=False)
    print("Saved evaluation/results/cnn_summary.csv")