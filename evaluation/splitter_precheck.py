"""Splitter pre-check: how much does regex vs nltk change what the compressor keeps?

No API calls. For every question it compresses the context twice (regex and nltk
sentence splitting, same scorer and level) and reports:

* how many contexts the two splitters split differently,
* how many questions end up with a different compressed context,
* mean token reduction and the share of questions whose gold answer is still in the
  compressed text, for each splitter, and how many questions keep it under only one.

Only questions whose compressed context differs can get a different LLM answer from the
splitter. The ids of those questions are saved so the later comparison can also be run on
just them (compare_runs --ids), where the effect is not diluted by identical inputs.

    python -m evaluation.splitter_precheck --scorer cross_encoder --level 0.5
"""

import argparse
import json

import pandas as pd

from prompt_compression.compressor import build_compressor
from prompt_compression.segmenters import split_sentences


def compare_splitters(samples: list[dict], scorer: str = "cross_encoder", level: float = 0.5,
                      window_size: int = 2, backend: str = "groq") -> pd.DataFrame:
    comps = {sp: build_compressor("selective", backend=backend, scorer=scorer,
                                  window_size=window_size, splitter=sp)
             for sp in ("regex", "nltk")}
    rows = []
    for s in samples:
        res = {sp: c.compress(s["context"], s["question"], level) for sp, c in comps.items()}
        gold = (s.get("answer") or "").lower()
        rows.append({
            "sample_id": s["id"],
            "split_differs": split_sentences(s["context"], "regex") != split_sentences(s["context"], "nltk"),
            "text_differs": res["regex"].text != res["nltk"].text,
            "reduction_regex": res["regex"].token_reduction_pct,
            "reduction_nltk": res["nltk"].token_reduction_pct,
            "answer_kept_regex": (gold in res["regex"].text.lower()) if gold else None,
            "answer_kept_nltk": (gold in res["nltk"].text.lower()) if gold else None,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    has_gold = df["answer_kept_regex"].notna()
    kr, kn = df.loc[has_gold, "answer_kept_regex"].astype(bool), df.loc[has_gold, "answer_kept_nltk"].astype(bool)
    return {
        "questions": len(df),
        "contexts_split_differently": int(df["split_differs"].sum()),
        "questions_with_different_compressed_text": int(df["text_differs"].sum()),
        "mean_reduction_regex": round(float(df["reduction_regex"].mean()), 2),
        "mean_reduction_nltk": round(float(df["reduction_nltk"].mean()), 2),
        "answer_kept_regex": round(float(kr.mean()), 4),
        "answer_kept_nltk": round(float(kn.mean()), 4),
        "answer_kept_only_by_regex": int((kr & ~kn).sum()),
        "answer_kept_only_by_nltk": int((~kr & kn).sum()),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", default="data/squad_300.json")
    ap.add_argument("--scorer", default="cross_encoder", choices=["tfidf", "bi_encoder", "cross_encoder"])
    ap.add_argument("--level", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out-ids", default="evaluation/results/splitter_diff_ids.txt")
    args = ap.parse_args(argv)

    with open(args.samples, encoding="utf-8") as f:
        samples = json.load(f)[: args.n]
    df = compare_splitters(samples, args.scorer, args.level, args.window)
    for key, value in summarize(df).items():
        print(f"{key}: {value}")
    ids = df.loc[df["text_differs"], "sample_id"].tolist()
    with open(args.out_ids, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + ("\n" if ids else ""))
    print(f"saved {len(ids)} ids with a different compressed text to {args.out_ids}")


if __name__ == "__main__":
    main()