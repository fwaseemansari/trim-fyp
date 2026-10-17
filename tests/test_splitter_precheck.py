"""splitter_precheck on synthetic samples (TF-IDF scorer, stand-in token counter; no models, no network)."""

import pytest

from evaluation.splitter_precheck import compare_splitters, summarize
from prompt_compression import compressor as comp_mod

SAMPLES = [
    {"id": "a", "question": "Who said inequality is rising?",
     "context": ("The 2013 Economics Nobel prize winner Robert J. Shiller said inequality is rising. "
                 "Prices went up in 2014. Wages did not follow. Critics disagreed with the analysis. "
                 "The debate continues today."), "answer": "Shiller"},
    {"id": "b", "question": "What happened to prices?",
     "context": "Prices went up in 2014. Wages did not follow. Critics disagreed. The debate goes on. Nothing else.",
     "answer": "went up"},
]


def test_precheck_counts_differences_and_flags_identical_contexts(monkeypatch):
    monkeypatch.setattr(comp_mod, "count_tokens", lambda text, model: len(text.split()))
    df = compare_splitters(SAMPLES, scorer="tfidf", level=0.5)
    out = summarize(df)
    assert out["questions"] == 2
    assert out["contexts_split_differently"] == 1                      # only sample "a" has an initial
    assert not df.loc[df.sample_id == "b", "text_differs"].iloc[0]      # identical input, identical output
    assert out["answer_kept_regex"] <= 1.0 and out["answer_kept_nltk"] <= 1.0