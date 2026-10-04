"""Hallucination audit for compressed-context runs (Week 4).

F1 only says an answer was wrong. It cannot say whether the LLM made
something up or the compressor threw the answer away. This tool builds a
sheet of sampled answers to label by hand, then summarises the labels.

Labels (judge each answer against the COMPRESSED context the model saw):

    faithful             the answer is supported by the compressed text
                         (right or wrong).
    unsupported_correct  the answer is right, but the compressed text does
                         not contain it: the model used its own knowledge.
                         F1 counts this as a success, yet compression broke
                         the context.
    hallucinated         the answer is not supported by the compressed text
                         and is wrong.
    unsure               cannot decide; kept and reported, never forced.

Workflow
--------
1. Build a blind sheet (method names are hidden; they live in a key file):

    python -m evaluation.hallucination_audit sample ^
        --results evaluation/results/experiment_baseline_300.csv ^
                  evaluation/results/experiment_selective_cross_encoder_0.5_300.csv ^
                  evaluation/results/experiment_selective_bi_encoder_0.5_300.csv ^
        --samples data/squad_300.json --n 40 ^
        --out evaluation/results/audit_sheet.csv

2. Open audit_sheet.csv, fill in the `label` column (and `notes` if useful).

3. Summarise:

    python -m evaluation.hallucination_audit summarize ^
        --sheet evaluation/results/audit_sheet.csv

Design notes
------------
* Each results CSV is one configuration; the compressor is rebuilt from the
  CSV's own columns, so nothing has to be typed by hand. Selective
  compression is deterministic, so recompression reproduces what the LLM
  saw. The tool checks this by recomputing `answer_still_present` and
  comparing it with the CSV's flag, and reports any mismatch.
* LLM-based compression is NOT deterministic. Those runs can only be
  audited if the CSV has a `compressed_context` column.
* Rows where the answer was lost in compression are over-sampled: they are
  where unsupported answers can occur at all.
* A baseline file (no compression) works too and is a useful control: it
  shows how often the model answers beyond the context with nothing removed.
"""

from __future__ import annotations

import argparse
import json
import math
import string
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

LABELS = ("faithful", "unsupported_correct", "hallucinated", "unsure")

SHEET_COLUMNS = [
    "row_id", "question", "gold_answer", "llm_response",
    "compressed_context", "response_in_context", "label", "notes",
]

CompressFn = Callable[[dict, str, str], str]

_compressor_cache: dict = {}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _truthy(value) -> Optional[bool]:
    """True/False for bool-like cells; None when the cell is missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value).strip().lower() == "true"


def _normalize(text) -> str:
    if not isinstance(text, str):
        return ""
    table = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(table).split())


def response_in_context(response, context) -> bool:
    """Cheap triage hint: is the whole response a substring of the context
    (case and punctuation ignored)? Not a verdict, just a pointer."""
    resp = _normalize(response)
    return bool(resp) and resp in _normalize(context)


def stratified_estimate(strata, z: float = 1.96) -> tuple[float, float, float]:
    """Population rate and approximate 95% CI from a stratified sample.

    strata: iterable of (N_h, n_h, k_h): rows in the stratum overall, rows
    labelled in the audit, rows with the label of interest.

    Answer-lost rows are over-sampled on purpose, so a plain k/n would
    overstate every rate. Each stratum is weighted by its real size instead.
    The interval is a normal approximation with an adjusted variance (so zero
    counts still give a non-zero width): treat it as rough when counts are small.
    """
    strata = [(N, n, k) for N, n, k in strata if N > 0 and n > 0]
    total = sum(N for N, _, _ in strata)
    if total == 0:
        return (0.0, 0.0, 0.0)
    est = sum(N * k / n for N, n, k in strata) / total
    # Variance uses a +0.5 / +1 adjusted proportion so zero counts do not give a
    # zero-width interval (0 events in 30 rows cannot rule out a rate of several percent).
    var = 0.0
    for N, n, k in strata:
        p_adj = (k + 0.5) / (n + 1)
        var += (N / total) ** 2 * p_adj * (1 - p_adj) / (n + 1)
    half = z * math.sqrt(var)
    return (est, max(0.0, est - half), min(1.0, est + half))


def default_compress_fn(config: dict, context: str, query: str) -> str:
    """Rebuild the same compressor the experiment used and apply it.

    Mirrors Pipeline._get_compressor(); adjust the build_compressor call
    here if its signature changes.
    """
    key = (config["method"], config["scorer"], config["window"], config["backend"])
    if key not in _compressor_cache:
        from prompt_compression.compressor import build_compressor
        _compressor_cache[key] = build_compressor(
            config["method"],
            backend=config["backend"],
            scorer=config["scorer"],
            llm_client=None,
            window_size=config["window"],
        )
    return _compressor_cache[key].compress(context, query=query, level=config["level"]).text


def _config_of(df: pd.DataFrame, path: str) -> dict:
    cols = ["compression_method", "compression_scorer", "compression_window_size",
            "compression_level", "backend"]
    unique = df[cols].drop_duplicates()
    if len(unique) != 1:
        raise ValueError(f"{path} mixes several configurations; audit one run per file")
    r = unique.iloc[0]
    window = r["compression_window_size"]
    return {
        "method": str(r["compression_method"]),
        "scorer": str(r["compression_scorer"]),
        "window": None if pd.isna(window) else int(window),
        "level": float(r["compression_level"]),
        "backend": str(r["backend"]),
    }


def _select_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to n rows: answer-lost rows first (at most half of n), then random."""
    present = df["answer_still_present"].map(_truthy)
    lost = df[present == False]  # noqa: E712  (None must not count as lost)
    take_lost = lost.sample(n=min(len(lost), n // 2), random_state=seed)
    rest = df.drop(take_lost.index)
    take_rest = rest.sample(n=min(len(rest), n - len(take_lost)), random_state=seed)
    return pd.concat([take_lost, take_rest])


# --------------------------------------------------------------------------
# sheet building
# --------------------------------------------------------------------------

def build_audit_sheet(results_paths, samples_path, n_per_file: int = 40, seed: int = 42,
                      out_path: str = "evaluation/results/audit_sheet.csv",
                      compress_fn: Optional[CompressFn] = None):
    """Write the blind sheet and its key file; return (sheet_df, key_df)."""
    compress_fn = compress_fn or default_compress_fn
    with open(samples_path, encoding="utf-8") as f:
        samples = {s["id"]: s for s in json.load(f)}

    sheet_rows, key_rows = [], []
    for path in results_paths:
        df = pd.read_csv(path)
        cfg = _config_of(df, str(path))
        has_ctx_col = "compressed_context" in df.columns
        skipped = 0
        first_key = len(key_rows)
        all_flags = df["answer_still_present"].map(_truthy)
        pop = {"lost": int((all_flags == False).sum()),            # noqa: E712
               "kept": int((all_flags != False).sum())}
        for _, r in _select_rows(df, n_per_file, seed).iterrows():
            sample = samples.get(r["sample_id"])
            if sample is None:
                skipped += 1
                continue
            original = sample["context"]
            saved = r["compressed_context"] if has_ctx_col else None
            if isinstance(saved, str) and saved.strip():
                compressed = saved
            elif cfg["method"] == "none":
                compressed = original
            elif "llm" in cfg["method"].lower():
                raise NotImplementedError(
                    f"{path}: LLM compression is not deterministic. Re-run it with a "
                    "'compressed_context' column in the output CSV to audit it.")
            else:
                compressed = compress_fn(cfg, original, r["question"])

            flag = _truthy(r.get("answer_still_present"))
            gold = r.get("gold_answer")
            recomputed = gold.lower() in compressed.lower() if isinstance(gold, str) and gold else None
            match = None if flag is None or recomputed is None else (flag == recomputed)

            sheet_rows.append({
                "question": r["question"],
                "gold_answer": gold,
                "llm_response": r.get("llm_response"),
                "compressed_context": compressed,
                "response_in_context": response_in_context(r.get("llm_response"), compressed),
                "label": "", "notes": "",
            })
            key_rows.append({
                "results_file": Path(path).name, "sample_id": r["sample_id"],
                "method": cfg["method"], "scorer": cfg["scorer"], "level": cfg["level"],
                "answer_still_present": flag, "f1": r.get("f1"),
                "recompress_matches_flag": match,
                "stratum": "lost" if flag is False else "kept",
            })
        for stratum in ("lost", "kept"):
            rows = [k for k in key_rows[first_key:] if k["stratum"] == stratum]
            for k in rows:
                k["stratum_population"] = pop[stratum]
                k["stratum_sampled"] = len(rows)
        if skipped:
            print(f"warning: {skipped} sample ids from {path} not found in {samples_path}")

    order = pd.Series(range(len(sheet_rows))).sample(frac=1, random_state=seed).tolist()
    sheet = pd.DataFrame([sheet_rows[i] for i in order])
    key = pd.DataFrame([key_rows[i] for i in order])
    sheet.insert(0, "row_id", range(1, len(sheet) + 1))
    key.insert(0, "row_id", range(1, len(key) + 1))
    sheet = sheet[SHEET_COLUMNS]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    key_path = out.with_name(out.stem + "_key.csv")
    sheet.to_csv(out, index=False)
    key.to_csv(key_path, index=False)

    checked = key["recompress_matches_flag"].dropna()
    if len(checked):
        print(f"recompression reproduced answer_still_present for "
              f"{int(checked.sum())}/{len(checked)} rows")
        if not checked.all():
            print("warning: some rows do not match the CSV flag; the compressor settings "
                  "may differ from the experiment's. Check before labelling.")
    print(f"wrote {len(sheet)} rows to {out} (key: {key_path})")
    return sheet, key


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def summarize_audit(sheet_path: str, key_path: Optional[str] = None,
                    out_path: Optional[str] = None) -> pd.DataFrame:
    """Per-configuration label counts and population rates (stratified, 95% CI).

    n and the label counts are raw audited rows; the rates are weighted back
    to the full run because answer-lost rows were over-sampled."""
    sheet_path = Path(sheet_path)
    key_path = Path(key_path) if key_path else sheet_path.with_name(sheet_path.stem + "_key.csv")
    sheet = pd.read_csv(sheet_path)
    key = pd.read_csv(key_path)

    labels = sheet["label"].fillna("").astype(str).str.strip().str.lower()
    unlabeled = int((labels == "").sum())
    bad = sheet.loc[(labels != "") & (~labels.isin(LABELS)), "row_id"].tolist()
    if bad:
        raise ValueError(f"unknown labels on rows {bad}; allowed: {', '.join(LABELS)}")
    if unlabeled:
        print(f"warning: {unlabeled} unlabeled rows excluded")

    merged = key.merge(sheet[["row_id"]].assign(label=labels), on="row_id")
    merged = merged[merged["label"] != ""]

    records = []
    for (method, scorer, level), g in merged.groupby(["method", "scorer", "level"]):
        counts = {lab: int((g["label"] == lab).sum()) for lab in LABELS}

        def strata_for(is_target):
            out = []
            for _, h in g.groupby("stratum"):
                out.append((h["stratum_population"].iloc[0], len(h), int(is_target(h["label"]).sum())))
            return out

        h_est, h_lo, h_hi = stratified_estimate(strata_for(lambda l: l == "hallucinated"))
        u_est, u_lo, u_hi = stratified_estimate(
            strata_for(lambda l: l.isin(["hallucinated", "unsupported_correct"])))
        records.append({
            "method": method, "scorer": scorer, "level": level, "n": len(g), **counts,
            "hallucinated_rate": round(h_est, 3),
            "hallucinated_ci": f"[{h_lo:.3f}, {h_hi:.3f}]",
            "unsupported_rate": round(u_est, 3),
            "unsupported_ci": f"[{u_lo:.3f}, {u_hi:.3f}]",
        })
    result = pd.DataFrame(records)
    if out_path:
        result.to_csv(out_path, index=False)
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="build the blind audit sheet")
    s.add_argument("--results", nargs="+", required=True)
    s.add_argument("--samples", default="data/squad_300.json")
    s.add_argument("--n", type=int, default=40, help="rows per results file")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="evaluation/results/audit_sheet.csv")

    m = sub.add_parser("summarize", help="summarise the labelled sheet")
    m.add_argument("--sheet", default="evaluation/results/audit_sheet.csv")
    m.add_argument("--key", default=None)
    m.add_argument("--out", default="evaluation/results/audit_summary.csv")

    args = parser.parse_args(argv)
    if args.cmd == "sample":
        build_audit_sheet(args.results, args.samples, args.n, args.seed, args.out)
    else:
        result = summarize_audit(args.sheet, args.key, args.out)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(result.to_string(index=False))


if __name__ == "__main__":
    main()