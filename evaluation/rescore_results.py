"""Re-score saved experiment CSVs with the current quality_metrics.

Needs no API calls: every results CSV stores the LLM response and the gold
answer, so EM / F1 / answer_in_response can be recomputed offline. Use it after
changing the scoring functions (for example the Unicode normalization fix) so all
runs are scored the same way.

    python -m evaluation.rescore_results evaluation/results/experiment_*_300.csv

For each input X.csv it writes X_rescored.csv (originals are never modified)
with the new em / f1 / answer_in_response and the previous values kept in
em_before / f1_before / answer_in_response_before, then prints the mean change.
Rows without a gold answer stay unscored (empty), as in run_experiment.
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

from evaluation.quality_metrics import contains_answer, exact_match, f1_score


def rescore(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    responses = df["llm_response"].fillna("")
    golds = df["gold_answer"].fillna("")
    scored = golds != ""

    for col in ("em", "f1", "answer_in_response"):
        df[f"{col}_before"] = df[col]

    df["em"] = [exact_match(r, g) if s else None for r, g, s in zip(responses, golds, scored)]
    df["f1"] = [round(f1_score(r, g), 4) if s else None for r, g, s in zip(responses, golds, scored)]
    df["answer_in_response"] = [contains_answer(r, g) if s else None
                                for r, g, s in zip(responses, golds, scored)]
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="results CSVs (globs allowed)")
    args = ap.parse_args(argv)

    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    paths = [p for p in paths if not p.endswith("_rescored.csv")]
    if not paths:
        raise SystemExit("no matching CSV files")

    for p in paths:
        out = rescore(p)
        target = Path(p).with_name(Path(p).stem + "_rescored.csv")
        out.to_csv(target, index=False)
        changed = int((out["f1"].astype(float) != out["f1_before"].astype(float)).sum())
        print(f"{Path(p).name}: F1 {out.f1_before.mean():.4f} -> {out.f1.mean():.4f}   "
              f"EM {out.em_before.mean():.4f} -> {out.em.mean():.4f}   rows changed: {changed}   -> {target.name}")


if __name__ == "__main__":
    main()