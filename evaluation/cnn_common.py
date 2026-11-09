"""Shared helpers for the CNN/DailyMail full-dataset scripts.

Put this file in evaluation/.
"""

from prompt_compression.compressor import build_compressor
from prompt_compression.segmenters import split_sentences
from token_analysis.counter import count_tokens

# CNN has no question, so every scorer ranks sentence windows against this fixed query.
COMPRESSION_QUERY = "Summarize the main points of this news article."


class CachedScorer:
    """Remembers the last scoring call, so the three levels of one article score it once.

    Levels only change which windows are kept, not the scores. Run the levels for an
    article back to back and the encoder (the slow part) runs once instead of three times.
    """

    def __init__(self, scorer):
        self._scorer = scorer
        self.name = scorer.name
        self._key = None
        self._val = None

    def score(self, segments, query):
        key = (tuple(s.text for s in segments), query)
        if key != self._key:
            self._val = self._scorer.score(segments, query)
            self._key = key
        return self._val


def make_compressor(method, scorer="tfidf", backend="groq", window_size=2, splitter="regex"):
    comp = build_compressor(method, backend=backend, scorer=scorer,
                            window_size=window_size, splitter=splitter)
    comp.scorer = CachedScorer(comp.scorer)
    return comp


def lead_compress(text, level, model, splitter="regex"):
    """Lead-N baseline: the first sentences that fit the same token budget.

    Same budget rule as SelectiveCompressor (int(original * level); the first
    sentence is always kept). Returns (text, original_tokens, compressed_tokens).
    """
    original = count_tokens(text, model)
    sentences = split_sentences(text, splitter)
    if not sentences or original == 0:
        return text, original, original
    budget = max(1, int(original * level))
    kept = []
    for s in sentences:
        if kept and count_tokens(" ".join(kept + [s]), model) > budget:
            break
        kept.append(s)
    out = " ".join(kept)
    return out, original, count_tokens(out, model)


def compress_article(kind, compressor, text, level, model, splitter="regex"):
    """kind: "full" (no compression), "lead", or "selective"/"extractive" (needs compressor).

    Returns (compressed_text, original_tokens, compressed_tokens).
    """
    if kind == "full":
        n = count_tokens(text, model)
        return text, n, n
    if kind == "lead":
        return lead_compress(text, level, model, splitter)
    res = compressor.compress(text, query=COMPRESSION_QUERY, level=level)
    return res.text, res.original_tokens, res.compressed_tokens
