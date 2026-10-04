"""Segmentation strategies used by selective compressors."""

from dataclasses import dataclass
import re


def split_sentences(text: str) -> list[str]:
    """Split text into sentences deterministically.

    The compression architecture relies on stable sentence indices, so
    we intentionally use a lightweight regex splitter rather than an
    external NLP sentence tokenizer.

    This handles normal prose such as:
        "The patient arrived. The doctor examined him."

    and short test sentences such as:
        "A. B. C. D."

    Limitation: abbreviations followed by a capitalised word ("Dr. Smith",
    "St. Louis", "U.S. Army") are split as if they ended a sentence. This only
    changes where segment boundaries fall; no text is lost.
    """

    text = (text or "").strip()

    if not text:
        return []

    # Split after ., !, or ? when followed by whitespace and the
    # beginning of the next sentence.
    parts = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\'])',
        text,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


@dataclass(frozen=True)
class Segment:
    text: str
    sentence_indices: tuple[int, ...]


class SentenceSegmenter:
    """One sentence per segment."""

    def segment(self, text: str) -> list[Segment]:
        sentences = split_sentences(text)

        return [
            Segment(
                sentence,
                (i,),
            )
            for i, sentence in enumerate(sentences)
        ]


@dataclass
class OverlappingWindowSegmenter:
    """Build overlapping windows of 2 or 3 sentences, sliding one sentence at a time.

    A text with N sentences gives N - window_size + 1 windows, so neighbouring
    windows share sentences. A text with window_size sentences or fewer becomes
    a single segment, which the compressor treats as "nothing to choose between".
    """

    window_size: int = 2

    def __post_init__(self):
        if self.window_size not in (2, 3):
            raise ValueError("window_size must be 2 or 3")

    def segment(self, text: str) -> list[Segment]:
        sentences = split_sentences(text)

        if not sentences:
            return []

        # If there are fewer sentences than the requested window size,
        # keep the complete document as one segment.
        if len(sentences) <= self.window_size:
            return [
                Segment(
                    text=" ".join(sentences),
                    sentence_indices=tuple(range(len(sentences))),
                )
            ]

        segments = []

        for start in range(
            len(sentences) - self.window_size + 1
        ):
            indices = tuple(
                range(
                    start,
                    start + self.window_size,
                )
            )

            segments.append(
                Segment(
                    text=" ".join(
                        sentences[i]
                        for i in indices
                    ),
                    sentence_indices=indices,
                )
            )

        return segments