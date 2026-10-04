"""Typed results returned by the Prompt Compression module."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompressionResult:
    """Complete, traceable result of one compression operation.

    text: the compressed context actually sent to the LLM.
    original_tokens / compressed_tokens: counted with the target model's tokenizer.
    kept_indices: SENTENCE positions kept from the original text.
    scores: one relevance score per SEGMENT (window), in segment order; not
        comparable across different scorers.
    method / scorer: how the result was produced.
    segment_count: number of segments the text was split into.
    llm_calls, compression_input_tokens, compression_output_tokens: cost of any
        LLM calls made by the compressor itself (0 for local selective methods).
    metadata: extras such as target_level, budget_tokens, selected_segments.
    """

    text: str
    original_tokens: int
    compressed_tokens: int
    kept_indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    method: str = "extractive"
    scorer: str = "tfidf"
    segment_count: int = 0
    llm_calls: int = 0
    compression_input_tokens: int = 0
    compression_output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round((self.original_tokens - self.compressed_tokens) / self.original_tokens * 100, 2)