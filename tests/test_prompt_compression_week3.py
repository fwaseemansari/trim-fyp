from dataclasses import dataclass

from prompt_compression.segmenters import OverlappingWindowSegmenter
from prompt_compression.scorers import Scorer
from prompt_compression.compressor import (
    SelectiveCompressor,
    RewriteCompressor,
)
from prompt_compression.results import CompressionResult


class FixedScorer(Scorer):
    """Deterministic scorer for testing.

    The third window receives the highest score so that for:

        A B
        B C
        C D
        D E

    the expected selected content is C D.
    """

    name = "fixed"

    def score(self, segments, query):
        scores = []

        for segment in segments:
            if segment.text.startswith("C. D."):
                scores.append(10.0)
            elif segment.text.startswith("D. E."):
                scores.append(5.0)
            elif segment.text.startswith("B. C."):
                scores.append(2.0)
            else:
                scores.append(1.0)

        return scores


@dataclass
class FakeResponse:
    response: str
    latency_ms: float = 0.0


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, backend="groq"):
        self.calls += 1
        return FakeResponse("A compact faithful fact.")


def test_overlapping_windows_are_two_sentences_and_overlap():
    segments = OverlappingWindowSegmenter(2).segment(
        "A. B. C. D."
    )

    assert [
        segment.sentence_indices
        for segment in segments
    ] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_selective_compressor_returns_original_sentence_order():
    text = "A. B. C. D. E."

    result = SelectiveCompressor(
        OverlappingWindowSegmenter(2),
        FixedScorer(),
        model="gpt-4o-mini",
    ).compress(
        text,
        query="C",
        level=0.5,
    )

    assert isinstance(result, CompressionResult)

    # The scorer intentionally makes the C-D window the highest-scoring
    # window. The compressor should return the selected sentences in their
    # original document order.
    assert result.text == "C. D."

    assert result.kept_indices == [2, 3]

    assert result.compressed_tokens <= result.original_tokens


def test_compressor_never_returns_empty_text():
    result = SelectiveCompressor(
        OverlappingWindowSegmenter(2),
        FixedScorer(),
        model="gpt-4o-mini",
    ).compress(
        "One. Two. Three.",
        query="anything",
        level=0.01,
    )

    assert result.text
    assert result.compressed_tokens > 0


def test_rewrite_compressor_tracks_llm_calls_and_cost_tokens():
    llm = FakeLLM()

    result = RewriteCompressor(
        OverlappingWindowSegmenter(2),
        FixedScorer(),
        llm,
        "groq",
        "gpt-4o-mini",
    ).compress(
        "A. B. C. D. E. F.",
        query="D",
        level=0.5,
    )

    assert isinstance(result, CompressionResult)

    assert result.method == "rewrite"

    assert result.llm_calls == llm.calls

    assert result.llm_calls > 0

    assert result.compression_input_tokens >= 0

    assert result.compression_output_tokens >= 0