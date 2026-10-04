"""Prompt Compression Module: composable, budget-aware context compression.

Architecture
------------
A *selective* compressor is built from three small parts:

    Segmenter  ->  Scorer  ->  greedy budget selection

* Segmenter (segmenters.py) splits the text into scorable segments: single
  sentences, or overlapping windows of 2-3 sentences.
* Scorer (scorers.py) gives each segment a relevance score against the query:
  TF-IDF, bi-encoder (embeddings) or cross-encoder.
* Selection keeps the highest-scoring segments while the union of their
  sentences fits a token budget, then restores document order.

Every compressor returns a typed CompressionResult (results.py): the text,
token counts, which sentences were kept, the scores and the cost of any LLM
calls, so each experiment row can be traced back to what the model saw.

Methods exposed by build_compressor()
-------------------------------------
extractive          sentence-level TF-IDF baseline (Week 2).
selective / drop    windowed segments + the chosen scorer; drops the rest.
rewrite             selective, then an LLM rewrites dropped windows into short
                    statements that fit any leftover budget.

``compress()`` keeps the Week 2 string-in / string-out API for compatibility.

Meaning of ``level``
--------------------
``level`` is the fraction of the ORIGINAL token count the output may use (a
token budget), not a fraction of sentences. It is a target, not a guarantee:
the best-scoring window is always kept even when it exceeds the budget, and
windows are whole units, so budget is often left unused. Report the achieved
reduction (CompressionResult.token_reduction_pct), not the nominal level. On
the 300-sample SQuAD runs, level 0.3 kept about 46% of tokens, level 0.5 about
52% and level 0.7 about 64%.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pipeline import config
from token_analysis.counter import count_tokens
from token_analysis.pricing import estimate_cost
from .results import CompressionResult
from .segmenters import SentenceSegmenter, OverlappingWindowSegmenter, split_sentences
from .scorers import TfidfScorer, BiEncoderScorer, CrossEncoderScorer
from .selectors import BudgetSelector


BACKEND_MODEL = {
    "groq": config.GROQ_MODEL,
    "openai": config.OPENAI_MODEL,
}


class Compressor(ABC):
    """Interface: turn ``text`` plus a ``query`` into a CompressionResult that
    uses roughly ``level`` of the original tokens."""

    @abstractmethod
    def compress(self, text: str, query: str = "", level: float = 0.5) -> CompressionResult:
        """Compress ``text`` for ``query``; ``level`` is in (0, 1]."""
        raise NotImplementedError


class SelectiveCompressor(Compressor):
    """Score segments, then greedily keep the best ones under a token budget.

    Segments are taken in descending score order (ties go to the earlier
    segment). A segment is added if the union of all kept sentences still fits
    the budget (int(original_tokens * level)). The first, best segment is kept
    even if it alone exceeds the budget, so the output is never empty. Kept
    sentences are joined in their original order.

    ``model`` is the tokenizer name used for all token counts, so budgets match
    what the target LLM is billed for.
    """

    def __init__(self, segmenter, scorer, model: str = "gpt-4o-mini"):
        self.segmenter = segmenter
        self.scorer = scorer
        self.model = model
        self.selector = BudgetSelector(model)

    def compress(self, text: str, query: str = "", level: float = 0.5) -> CompressionResult:
        """Compress ``text`` for ``query`` to roughly ``level`` of its tokens.

        Returns the text unchanged when it is empty or has at most one segment
        (with 2-sentence windows: a context of two sentences or fewer), because
        there is nothing to choose between.

        Two index spaces appear in the result: ``kept_indices`` are SENTENCE
        positions in the original text, while ``scores`` holds one value per
        SEGMENT (window), in segment order.
        """
        level = _validate_level(level)
        original_tokens = count_tokens(text, self.model)
        if not text.strip() or original_tokens == 0:
            return CompressionResult(text=text, original_tokens=original_tokens,
                                    compressed_tokens=original_tokens,
                                    method="selective", scorer=self.scorer.name)

        segments = self.segmenter.segment(text)
        if len(segments) <= 1:
            return CompressionResult(text=text, original_tokens=original_tokens,
                                    compressed_tokens=original_tokens,
                                    kept_indices=list(segments[0].sentence_indices) if segments else [],
                                    scores=[1.0] if segments else [],
                                    method="selective", scorer=self.scorer.name,
                                    segment_count=len(segments))

        scores = self.scorer.score(segments, query)
        budget = max(1, int(original_tokens * level))
        sentences = split_sentences(text)
        ranked = sorted(range(len(segments)), key=lambda i: (scores[i], -i), reverse=True)

        kept_sentence_indices = set()
        selected_segments = []
        for segment_id in ranked:
            candidate = kept_sentence_indices | set(segments[segment_id].sentence_indices)
            candidate_text = " ".join(sentences[i] for i in sorted(candidate))
            candidate_tokens = count_tokens(candidate_text, self.model)
            if not selected_segments and candidate_tokens > budget:
                # Always keep at least one semantically selected window.
                selected_segments.append(segment_id)
                kept_sentence_indices = candidate
                break
            if candidate_tokens <= budget:
                selected_segments.append(segment_id)
                kept_sentence_indices = candidate

        if not selected_segments:
            # Defensive only: the loop above always selects the first-ranked segment.
            selected_segments = [ranked[-1]]
            kept_sentence_indices = set(segments[selected_segments[0]].sentence_indices)

        kept_sentence_indices = sorted(kept_sentence_indices)
        compressed = " ".join(sentences[i] for i in kept_sentence_indices)
        compressed_tokens = count_tokens(compressed, self.model)

        return CompressionResult(
            text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            kept_indices=kept_sentence_indices,
            scores=scores,
            method="selective",
            scorer=self.scorer.name,
            segment_count=len(segments),
            metadata={"target_level": level, "budget_tokens": budget,
                      "selected_segments": selected_segments},
        )


class RewriteCompressor(SelectiveCompressor):
    """Selective compression with LLM rewrites for dropped chunks.

    The selective compressor first chooses the retained sentence union.
    Dropped chunks are then rewritten by the LLM. Rewritten chunks are
    included only when they fit the context token budget.

    Importantly, LLM calls are still made and counted even when a generated
    rewrite cannot fit the final context budget. This is necessary because
    the rewrite call itself is part of the end-to-end compression cost.

    Known limitations (documented, not yet fixed):
    * Windows overlap, so a "dropped" window can share sentences with kept
      text; its rewrite may repeat facts that are already kept.
    * Rewrites are appended AFTER the kept text, not interleaved in document
      order.
    * One LLM call is made per dropped window, so cost grows with document
      length. Every call is counted in compression_input/output_tokens.
    """

    def __init__(self, segmenter, scorer, llm_client, backend: str, model: str):
        super().__init__(segmenter, scorer, model)
        self.llm_client = llm_client
        self.backend = backend

    def compress(
        self,
        text: str,
        query: str = "",
        level: float = 0.5,
    ) -> CompressionResult:

        base = super().compress(text, query, level)

        if base.original_tokens == 0:
            return base

        if base.segment_count <= 1:
            return CompressionResult(
                text=base.text,
                original_tokens=base.original_tokens,
                compressed_tokens=base.compressed_tokens,
                kept_indices=base.kept_indices,
                scores=base.scores,
                method="rewrite",
                scorer=base.scorer,
                segment_count=base.segment_count,
                llm_calls=0,
                compression_input_tokens=0,
                compression_output_tokens=0,
                metadata={
                    **base.metadata,
                    "rewritten_segments": [],
                },
            )

        segments = self.segmenter.segment(text)
        scores = base.scores

        budget = base.metadata.get(
            "budget_tokens",
            max(1, int(base.original_tokens * level)),
        )

        selected_segment_ids = set(
            base.metadata.get("selected_segments", [])
        )

        dropped_ids = [
            i
            for i in range(len(segments))
            if i not in selected_segment_ids
        ]

        # Nothing was dropped, so there is nothing to rewrite.
        if not dropped_ids:
            return CompressionResult(
                text=base.text,
                original_tokens=base.original_tokens,
                compressed_tokens=base.compressed_tokens,
                kept_indices=base.kept_indices,
                scores=base.scores,
                method="rewrite",
                scorer=base.scorer,
                segment_count=base.segment_count,
                llm_calls=0,
                compression_input_tokens=0,
                compression_output_tokens=0,
                metadata={
                    **base.metadata,
                    "rewritten_segments": [],
                },
            )

        used_tokens = count_tokens(base.text, self.model)

        input_tokens = 0
        output_tokens = 0
        llm_calls = 0

        kept_rewrites = []

        # Rewrite higher-scoring dropped chunks first.
        ranked_dropped_ids = sorted(
            dropped_ids,
            key=lambda x: (scores[x], -x),
            reverse=True,
        )

        for segment_id in ranked_dropped_ids:

            chunk = segments[segment_id].text

            prompt = (
                "Rewrite this context chunk into the shortest faithful "
                "statement(s) that preserve facts potentially needed to "
                "answer the question. Do not add information or speculate. "
                "Return only the rewritten text.\n\n"
                f"Question: {query}\n\n"
                f"Chunk:\n{chunk}"
            )

            # Make the actual LLM compression call.
            response_obj = self.llm_client.generate(
                prompt,
                backend=self.backend,
            )

            # Count the actual call, not the number of chunks we intended
            # to process.
            llm_calls += 1

            response = (
                getattr(response_obj, "response", "")
                or ""
            ).strip()

            prompt_token_count = count_tokens(
                prompt,
                self.model,
            )

            response_token_count = count_tokens(
                response,
                self.model,
            )

            input_tokens += prompt_token_count
            output_tokens += response_token_count

            if not response:
                continue

            # The final compressed context must still respect the requested
            # context budget. The rewrite call itself is counted separately
            # above as compression cost.
            if used_tokens + response_token_count <= budget:
                kept_rewrites.append(
                    (
                        segment_id,
                        response,
                        scores[segment_id],
                    )
                )

                used_tokens += response_token_count

        # Kept text first, then the rewrites ordered by segment index
        # (not interleaved with the kept sentences).
        pieces = [base.text]

        for _, response, _ in sorted(
            kept_rewrites,
            key=lambda item: item[0],
        ):
            pieces.append(response)

        final_text = " ".join(
            piece
            for piece in pieces
            if piece.strip()
        )

        final_tokens = count_tokens(
            final_text,
            self.model,
        )

        return CompressionResult(
            text=final_text,
            original_tokens=base.original_tokens,
            compressed_tokens=final_tokens,
            kept_indices=base.kept_indices,
            scores=base.scores,
            method="rewrite",
            scorer=base.scorer,
            segment_count=base.segment_count,
            llm_calls=llm_calls,
            compression_input_tokens=input_tokens,
            compression_output_tokens=output_tokens,
            metadata={
                **base.metadata,
                "rewritten_segments": [
                    segment_id
                    for segment_id, _, _ in kept_rewrites
                ],
            },
        )


def _validate_level(level: float) -> float:
    """Return ``level`` as a float; raise ValueError unless 0 < level <= 1."""
    if not 0 < level <= 1:
        raise ValueError("level must be in the range (0, 1]")
    return float(level)


def build_compressor(method: str, *, backend: str = "groq", scorer: str = "bi_encoder",
                     level: float = 0.5, llm_client=None, window_size: int = 2):
    """Factory used by the pipeline and experiment scripts.

    method: "extractive" (sentence-level TF-IDF baseline), "selective" or
        "drop" (windowed segments + ``scorer``), or "rewrite" (selective plus
        LLM rewrites of dropped windows; needs ``llm_client``).
    backend: "groq" or "openai"; decides which tokenizer counts the budget.
    scorer: "tfidf", "bi_encoder" or "cross_encoder" (ignored by "extractive").
    window_size: sentences per window, 2 or 3.
    level: accepted for API compatibility but unused here; pass the level to
        ``compress()``.

    Raises ValueError for an unknown method or scorer. Whole-text LLM
    condensing is not a builder method; it exists only in the legacy
    ``compress(method="llm")``.
    """
    model = BACKEND_MODEL[backend]
    segmenter = OverlappingWindowSegmenter(window_size=window_size)

    scorer_map = {
        "tfidf": TfidfScorer,
        "bi_encoder": BiEncoderScorer,
        "cross_encoder": CrossEncoderScorer,
    }
    if scorer not in scorer_map:
        raise ValueError(f"Unknown scorer: {scorer}")
    scorer_obj = scorer_map[scorer]()

    if method in {"extractive", "selective", "drop"}:
        # Week 2's extractive baseline remains sentence-level TF-IDF when
        # explicitly requested; Week 3 selective variants use windows.
        if method == "extractive":
            return SelectiveCompressor(SentenceSegmenter(), TfidfScorer(), model)
        return SelectiveCompressor(segmenter, scorer_obj, model)

    if method == "rewrite":
        if llm_client is None:
            raise ValueError("method='rewrite' requires llm_client")
        return RewriteCompressor(segmenter, scorer_obj, llm_client, backend, model)

    raise ValueError(f"Unknown compression method: {method}")


def compress(text: str, query: str = "", method: str = "extractive", level: float = 0.5,
             llm_client=None, backend: str = "groq", scorer: str = "tfidf") -> str:
    """Backward-compatible Week 2 API: returns only the compressed string.

    method="llm" asks the LLM to condense the whole text to about ``level`` of
    its length (and falls back to the original text if the reply is empty).
    Other methods go through build_compressor(). New code should call
    ``build_compressor(...).compress(...)`` to get the full CompressionResult.
    """
    if method == "llm":
        if llm_client is None:
            raise ValueError("method='llm' requires llm_client")
        target_pct = round(_validate_level(level) * 100)
        prompt = (
            f"Condense the following text to roughly {target_pct}% of its original length, "
            "preserving all facts needed to answer the question. Return only the condensed text.\n\n"
            f"Question: {query}\n\nText:\n{text}"
        )
        response = llm_client.generate(prompt, backend=backend).response.strip()
        return response or text

    return build_compressor(method, backend=backend, scorer=scorer, llm_client=llm_client).compress(
        text, query, level
    ).text