"""Measure run-to-run noise from two baseline runs, and compare treatments to it.

The same system, run twice on the same questions, gives slightly different answers
(LLM sampling). Until that noise is measured, "no loss detected" is a statement about
one baseline run only. This script takes two baseline runs and reports:

* the mean paired F1 difference between them (about 0 if there is no drift),
* the standard deviation of the per-question difference and the share of questions
  whose F1 changed,
* the smallest mean F1 change that a paired comparison on this many questions could
  detect (80% power, 5% two-sided): 2.8 * SD / sqrt(n),
* optionally, each treatment's F1 change against run 1, run 2 and their average, and
  whether it is larger than that detectable change.

    python -m evaluation.noise_floor ^
        --run1 evaluation/results/experiment_baseline_300_rescored.csv ^
        --run2 evaluation/results/experiment_baseline_300_run2_rescored.csv ^
        --treatment cross_0.5=evaluation/results/experiment_selective_cross_encoder_0.5_300_rescored.csv

Use rescored files for all runs.
"""

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.compare_runs import bootstrap_ci, load


def noise_floor(run1: pd.DataFrame, run2: pd.DataFrame) -> dict:
    ids = [i for i in run1.index if i in run2.index]
    d = (run2.loc[ids, "f1"] - run1.loc[ids, "f1"]).to_numpy(dtype=float)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    lo, hi = bootstrap_ci(d) if n > 1 else (float("nan"), float("nan"))
    return {
        "n": n,
        "f1_run1": float(run1.loc[ids, "f1"].mean()),
        "f1_run2": float(run2.loc[ids, "f1"].mean()),
        "mean_diff": float(d.mean()),
        "ci_low": lo, "ci_high": hi,
        "sd_per_question": sd,
        "share_changed": float((np.abs(d) > 1e-9).mean()),
        "p_paired_t": float(stats.ttest_rel(run2.loc[ids, "f1"], run1.loc[ids, "f1"]).pvalue) if n > 1 else float("nan"),
        "min_detectable_f1_change": 2.8 * sd / math.sqrt(n) if n > 1 else float("nan"),
    }


def treatment_vs_baselines(run1: pd.DataFrame, run2: pd.DataFrame, treatment: pd.DataFrame, mde: float) -> dict:
    ids = [i for i in treatment.index if i in run1.index and i in run2.index]
    t = treatment.loc[ids, "f1"].to_numpy(dtype=float)
    r1, r2 = run1.loc[ids, "f1"].to_numpy(dtype=float), run2.loc[ids, "f1"].to_numpy(dtype=float)
    avg = (r1 + r2) / 2
    out = {"n": len(ids), "d_vs_run1": float((t - r1).mean()), "d_vs_run2": float((t - r2).mean()),
           "d_vs_avg": float((t - avg).mean())}
    out["larger_than_noise"] = abs(out["d_vs_avg"]) > mde
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run1", required=True)
    ap.add_argument("--run2", required=True)
    ap.add_argument("--treatment", action="append", default=[], metavar="LABEL=CSV")
    args = ap.parse_args(argv)

    r1, r2 = load(args.run1), load(args.run2)
    nf = noise_floor(r1, r2)
    print(f"questions paired: {nf['n']}")
    print(f"mean F1: run1 {nf['f1_run1']:.4f}, run2 {nf['f1_run2']:.4f}")
    print(f"mean paired difference (run2 - run1): {nf['mean_diff']:+.4f} "
          f"(95% CI {nf['ci_low']:+.4f} to {nf['ci_high']:+.4f}, p = {nf['p_paired_t']:.3f})")
    print(f"SD of per-question difference: {nf['sd_per_question']:.4f}; "
          f"questions whose F1 changed: {nf['share_changed']:.1%}")
    print(f"smallest detectable mean F1 change (80% power): {nf['min_detectable_f1_change']:.4f}")
    for item in args.treatment:
        label, _, path = item.partition("=")
        res = treatment_vs_baselines(r1, r2, load(path), nf["min_detectable_f1_change"])
        verdict = "larger than" if res["larger_than_noise"] else "within"
        print(f"{label}: vs run1 {res['d_vs_run1']:+.4f}, vs run2 {res['d_vs_run2']:+.4f}, "
              f"vs average {res['d_vs_avg']:+.4f} -> {verdict} the detectable change")


if __name__ == "__main__":
    main()