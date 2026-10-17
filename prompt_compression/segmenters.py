"""Segmentation strategies used by selective compressors.

Two sentence splitters are available: "regex" (the default; lightweight and
deterministic) and "nltk" (Punkt, which knows common abbreviations). On the 265
distinct SQuAD contexts in squad_300.json they disagree on 64 (24%): regex
over-splits after initials and abbreviations in 21, nltk splits more often in
42. The choice is an experiment parameter, not a fixed assumption.
"""

from dataclasses import dataclass
import re

SPLITTERS = ("regex", "nltk")


def _split_regex(text: str) -> list[str]:
    """Regex splitter: a sentence ends at . ! or ? followed by whitespace and a
    capital letter, digit or quote.

    The compression architecture relies on stable sentence indices, so the
    default is this lightweight regex splitter rather than an external NLP
    sentence tokenizer.

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


def _split_nltk(text: str) -> list[str]:
    """Punkt splitter from nltk. Downloads the Punkt data on first use."""
    text = (text or "").strip()
    if not text:
        return []
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def split_sentences(text: str, splitter: str = "regex") -> list[str]:
    """Split ``text`` into sentences with the chosen splitter ("regex" or "nltk")."""
    if splitter == "regex":
        return _split_regex(text)
    if splitter == "nltk":
        return _split_nltk(text)
    raise ValueError(f"Unknown splitter: {splitter!r}; choose from {SPLITTERS}")


@dataclass(frozen=True)
class Segment:
    text: str
    sentence_indices: tuple[int, ...]


class SentenceSegmenter:
    """One sentence per segment."""

    def __init__(self, splitter: str = "regex"):
        split_sentences("", splitter)  # validates the name
        self.splitter = splitter

    def segment(self, text: str) -> list[Segment]:
        sentences = split_sentences(text, self.splitter)

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
    splitter: str = "regex"

    def __post_init__(self):
        if self.window_size not in (2, 3):
            raise ValueError("window_size must be 2 or 3")
        split_sentences("", self.splitter)  # validates the name

    def segment(self, text: str) -> list[Segment]:
        sentences = split_sentences(text, self.splitter)

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