"""Aggregate per-sample experiment results into the 7-metric summary table + charts.

Input CSV, one row per sample. Required columns:
  method, level, backend, orig_tokens, comp_tokens
Optional columns (used if present):
  orig_cost, comp_cost, orig_latency_ms, comp_latency_ms,
  retention, em, f1, rouge_l

Usage:
  python -m evaluation.summarize_results evaluation/results/a.csv evaluation/results/b.csv
"""
import sys
from pathlib import Path

import pandas as pd

try:
    from evaluation.metrics import token_reduction_pct, compression_ratio, cost_reduction_pct
except ImportError:  # run directly as a script
    from metrics import token_reduction_pct, compression_ratio, cost_reduction_pct

GROUP = ["method", "level", "backend"]
METRICS = ["token_reduction_pct", "compression_ratio", "cost_reduction_pct",
           "latency_ms", "latency_delta_ms", "retention", "em", "f1",
           "answer_in_response", "rouge_l"]
# run_experiment.py column names -> names used here
RENAME = {"compression_method": "method", "compression_level": "level",
          "context_tokens_original": "orig_tokens", "context_tokens_compressed": "comp_tokens",
          "information_retention": "retention"}


def load(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d = d.rename(columns={k: v for k, v in RENAME.items() if v not in d.columns})
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def add_derived(df):
    df = df.copy()
    df["token_reduction_pct"] = [token_reduction_pct(o, c) for o, c in zip(df.orig_tokens, df.comp_tokens)]
    df["compression_ratio"] = [compression_ratio(o, c) for o, c in zip(df.orig_tokens, df.comp_tokens)]
    if {"orig_cost", "comp_cost"} <= set(df.columns):
        df["cost_reduction_pct"] = [cost_reduction_pct(o, c) for o, c in zip(df.orig_cost, df.comp_cost)]
    if {"orig_latency_ms", "comp_latency_ms"} <= set(df.columns):
        df["latency_delta_ms"] = df.comp_latency_ms - df.orig_latency_ms
    return df


def summarize(df):
    df = add_derived(df)
    cols = [m for m in METRICS if m in df.columns]
    agg = df.groupby(GROUP)[cols].agg(["mean", "std"]).round(3)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_samples"] = df.groupby(GROUP).size()
    return agg.reset_index()


def paired_ttest(baseline_scores, treatment_scores):
    """Paired significance check (Phase 2 plan). Returns (t, p)."""
    from scipy.stats import ttest_rel
    t, p = ttest_rel(baseline_scores, treatment_scores)
    return float(t), float(p)


def make_charts(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = add_derived(df)
    outdir.mkdir(parents=True, exist_ok=True)

    if "f1" in df.columns:  # chart 1: trade-off
        g = df.groupby(["method", "level"])[["compression_ratio", "f1"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, part in g.groupby("method"):
            part = part.sort_values("compression_ratio")
            ax.plot(part.compression_ratio, part.f1, marker="o", label=method)
        ax.set_xlabel("Compression ratio"); ax.set_ylabel("Task accuracy (F1)")
        ax.set_title("Compression vs task accuracy"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(outdir / "tradeoff_ratio_vs_f1.png", dpi=200); plt.close(fig)

    g = df.groupby("method")["token_reduction_pct"].mean()  # chart 2
    fig, ax = plt.subplots(figsize=(7, 5))
    g.plot(kind="bar", ax=ax); ax.set_ylabel("Token reduction (%)")
    ax.set_title("Token reduction by method"); fig.tight_layout()
    fig.savefig(outdir / "token_reduction_by_method.png", dpi=200); plt.close(fig)

    if "cost_reduction_pct" in df.columns:  # chart 3
        g = df.groupby("method")["cost_reduction_pct"].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        g.plot(kind="bar", ax=ax); ax.set_ylabel("Cost reduction (%)")
        ax.set_title("Cost savings by method"); fig.tight_layout()
        fig.savefig(outdir / "cost_savings_by_method.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    data = load(paths)
    out = paths[0].parent
    table = summarize(data)
    table.to_csv(out / "summary_all_runs.csv", index=False)
    (out / "summary_all_runs.tex").write_text(table.to_latex(index=False))
    make_charts(data, out / "charts")
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))