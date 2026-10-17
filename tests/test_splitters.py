"""Sentence splitter option (regex vs nltk) and its plumbing through the compressor."""

import pytest

from pipeline import pipeline as pipe
from pipeline.config import PipelineConfig
from prompt_compression import compressor as comp_mod
from prompt_compression.compressor import SelectiveCompressor, build_compressor
from prompt_compression.segmenters import (
    OverlappingWindowSegmenter, SentenceSegmenter, split_sentences,
)

TEXT = ("The 2013 Economics Nobel prize winner Robert J. Shiller said inequality is rising. "
        "Prices went up in 2014. Wages did not follow. Critics disagreed with the analysis. "
        "The debate continues today.")


def test_regex_is_still_the_default():
    assert split_sentences(TEXT) == split_sentences(TEXT, "regex")


def test_regex_oversplits_an_initial_and_nltk_does_not():
    regex, nltk_s = split_sentences(TEXT, "regex"), split_sentences(TEXT, "nltk")
    assert len(regex) == len(nltk_s) + 1
    assert any(s.endswith("Robert J.") for s in regex)
    assert not any(s.endswith("Robert J.") for s in nltk_s)


def test_unknown_splitter_is_rejected_everywhere():
    with pytest.raises(ValueError):
        split_sentences("A b.", "spacy")
    with pytest.raises(ValueError):
        SentenceSegmenter(splitter="spacy")
    with pytest.raises(ValueError):
        OverlappingWindowSegmenter(window_size=2, splitter="spacy")
    with pytest.raises(ValueError):
        build_compressor("selective", scorer="tfidf", splitter="spacy")


def test_segmenters_use_their_splitter():
    n_regex = len(OverlappingWindowSegmenter(2, "regex").segment(TEXT))
    n_nltk = len(OverlappingWindowSegmenter(2, "nltk").segment(TEXT))
    assert n_regex == n_nltk + 1
    assert len(SentenceSegmenter("nltk").segment(TEXT)) == len(split_sentences(TEXT, "nltk"))


def test_build_compressor_passes_the_splitter_through():
    assert build_compressor("selective", scorer="tfidf", splitter="nltk").segmenter.splitter == "nltk"
    assert build_compressor("extractive", splitter="nltk").segmenter.splitter == "nltk"
    assert build_compressor("selective", scorer="tfidf").segmenter.splitter == "regex"


class _PositionScorer:
    name = "position"

    def score(self, segments, query):
        return [float(i == 1) for i, _ in enumerate(segments)]


@pytest.mark.parametrize("splitter", ["regex", "nltk"])
def test_compressor_indices_stay_consistent_with_the_splitter(monkeypatch, splitter):
    monkeypatch.setattr(comp_mod, "count_tokens", lambda text, model: len(text.split()))
    c = SelectiveCompressor(OverlappingWindowSegmenter(2, splitter), _PositionScorer(), "gpt-4o-mini")
    result = c.compress(TEXT, "q", 0.5)
    sentences = split_sentences(TEXT, splitter)
    assert result.kept_indices == sorted(result.kept_indices)
    assert max(result.kept_indices) < len(sentences)
    assert result.text == " ".join(sentences[i] for i in result.kept_indices)


def test_pipeline_config_splitter_reaches_build_compressor(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipe, "build_compressor", lambda *a, **k: seen.update(k) or object())
    p = pipe.Pipeline(llm=object(), analyzer=object(),
                      config=PipelineConfig(compression_enabled=True, compression_splitter="nltk"))
    p._get_compressor()
    assert seen["splitter"] == "nltk" and seen["window_size"] == 2


def test_run_passes_the_splitter_into_the_config(monkeypatch):
    captured = {}

    class FakePipeline:
        def __init__(self, llm=None, analyzer=None, config=None, **kw):
            captured["config"] = config

        def process(self, **kw):
            return {}

    monkeypatch.setattr(pipe, "Pipeline", FakePipeline)
    pipe.run("q", "c", compression_enabled=True, compression_splitter="nltk")
    assert captured["config"].compression_splitter == "nltk"