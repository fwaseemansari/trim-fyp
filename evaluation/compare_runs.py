"""Compare experiment runs against a reference run on the same questions.

    python -m evaluation.compare_runs ^
        --reference evaluation/results/experiment_baseline_300_rescored.csv ^
        --run rewrite=evaluation/results/experiment_rewrite_cross_encoder_0.5_50_rescored.csv ^
        --run selective=evaluation/results/experiment_selective_cross_encoder_0.5_300_rescored.csv ^
        --n 50

For every run it reports, on the questions it shares with the reference (the first
--n of them, in the reference file's order, when --n is given): mean F1 and EM, the
paired F1 change with a bootstrap 95% CI and a paired t-test p-value, mean context-token
reduction (--ids FILE restricts the comparison to the listed sample ids), and, when the columns exist, compression LLM calls and cost. Use rescored files.

Use it for the rewrite pilot (reference = baseline), for the splitter check
(reference = the regex run, run = the nltk run) and for any other pairwise comparison.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["llm_response"] = df["llm_response"].fillna("") if "llm_response" in df else ""
    return df.drop_duplicates("sample_id").set_index("sample_id")


def bootstrap_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(reference: pd.DataFrame, runs: dict[str, pd.DataFrame], n: int | None = None,
            only_ids: set | None = None) -> pd.DataFrame:
    ids = list(reference.index)
    if only_ids is not None:
        ids = [i for i in ids if i in only_ids]
    if n:
        ids = ids[:n]
    rows = []
    for label, df in {"reference": reference, **runs}.items():
        common = [i for i in ids if i in df.index]
        d, ref = df.loc[common], reference.loc[common]
        row = {"run": label, "n": len(common),
               "f1": round(d["f1"].mean(), 4), "em": round(d["em"].mean(), 4)}
        if label != "reference" and len(common) > 1:
            diff = (d["f1"] - ref["f1"]).to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff)
            row.update({"d_f1": round(diff.mean(), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                        "p_paired_t": round(float(stats.ttest_rel(d["f1"], ref["f1"]).pvalue), 4)})
        for col, name in (("token_reduction_pct", "token_reduction_pct"),
                          ("compression_llm_calls", "compression_calls"),
                          ("answer_call_cost_usd", "answer_cost_usd"),
                          ("compression_call_cost_usd", "compression_cost_usd"),
                          ("end_to_end_cost_usd", "end_to_end_cost_usd")):
            if col in d.columns:
                total = col in ("compression_llm_calls", "answer_call_cost_usd",
                                "compression_call_cost_usd", "end_to_end_cost_usd")
                row[name] = round(float(d[col].sum() if total else d[col].mean()), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", required=True, help="CSV of the reference run (usually the baseline)")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=CSV",
                    help="a run to compare; repeat for several")
    ap.add_argument("--n", type=int, default=None, help="use only the first N reference questions")
    ap.add_argument("--ids", default=None, metavar="FILE",
                    help="text file with one sample id per line; compare only those questions")
    ap.add_argument("--out", default=None, help="optional CSV to save the table")
    args = ap.parse_args(argv)

    runs = {}
    for item in args.run:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--run expects LABEL=CSV, got {item!r}")
        runs[label] = load(path)
    only = None
    if args.ids:
        with open(args.ids, encoding="utf-8") as f:
            only = {line.strip() for line in f if line.strip()}
    table = compare(load(args.reference), runs, args.n, only)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()