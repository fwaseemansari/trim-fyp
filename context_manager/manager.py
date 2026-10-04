"""
Adaptive Context Manager — Strategy pattern (revised from the earlier
single-class-with-strategy-switch design, for consistency with the
LLMClient refactor).

Inheritance structure — deliberately NOT three equal siblings:

    ContextStrategy (ABC)
    ├── SlidingWindowStrategy
    └── RelevanceAwareStrategy
            └── MemorySummarizationStrategy

sliding_window and relevance_aware are genuinely independent — they
decide what to KEEP using completely different logic (recency vs.
TF-IDF scoring), so they're siblings. memory_summarization is NOT
independent of relevance_aware: it uses the exact same TF-IDF scoring
to decide what to keep, and only differs in what happens to the turns
that get DROPPED (discarded outright vs. condensed into a 1-sentence
LLM summary). Making it a third sibling would mean copy-pasting the
entire TF-IDF scoring block into a third class — a real step backward,
since a bug fix there would then need to happen twice. Instead
MemorySummarizationStrategy subclasses RelevanceAwareStrategy and
overrides exactly one hook method (_handle_dropped) — the one point
where the two behaviors actually differ.

ContextManager below is a thin factory (using __new__ to return the
right subclass instance) so existing call sites elsewhere
(pipeline.run_conversation, context_manager/test_manager.py) don't
need to change: ContextManager(strategy="sliding_window", max_turns=6)
still works exactly as before, now backed by real subclasses instead
of an if/elif strategy switch inside one class.
"""

from abc import ABC, abstractmethod

from token_analysis.counter import count_tokens


class ContextStrategy(ABC):
    """Shared interface every strategy implements: add_turn() to feed
    in a new turn, get_context()/total_tokens() to read out the
    current context. get_context()/total_tokens() have a shared default
    implementation here since both concrete strategies below use the
    same "join kept turns" logic — only add_turn() must be overridden
    per strategy, since that's where the actual decision logic differs."""

    def __init__(self):
        self.history: list[dict] = []  # turns currently kept in context

    @abstractmethod
    def add_turn(self, turn: dict, query: str = "") -> None:
        """Adds one turn (expects at least a 'text' key — matches
        LoCoMo's {"speaker", "dia_id", "text"} shape) and applies this
        strategy's keep/drop decision. `query` is the current question
        being asked — ignored by SlidingWindowStrategy, required for
        the TF-IDF-based strategies to score relevance against."""
        raise NotImplementedError

    def get_context(self) -> str:
        """Returns the current context as a single string: kept turns
        in chronological order, ready to prepend to a prompt sent to
        the LLM. Subclasses with extra state to append (memory
        summaries) override this and extend the base result."""
        parts = [f"{t.get('speaker', '')}: {t['text']}" for t in self.history]
        return "\n".join(parts)

    def total_tokens(self) -> int:
        return count_tokens(self.get_context())


class SlidingWindowStrategy(ContextStrategy):
    """Simplest strategy, and the Week 1 fallback/control condition:
    keep only the last N turns, drop everything older with no record
    kept at all."""

    def __init__(self, max_turns: int = 6):
        super().__init__()
        self.max_turns = max_turns

    def add_turn(self, turn: dict, query: str = "") -> None:
        self.history.append(turn)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]


class RelevanceAwareStrategy(ContextStrategy):
    """Scores every turn's relevance to `query` via TF-IDF cosine
    similarity (reuses the same approach as the Prompt Compression
    Module for consistency — see prompt_compression/compressor.py),
    keeps turns in relevance order until max_tokens is spent.

    Turns that get dropped are discarded outright by default —
    MemorySummarizationStrategy below overrides _handle_dropped() to
    condense them into a memory summary instead, without needing to
    duplicate any of the scoring/ranking logic here.
    """

    def __init__(self, max_tokens: int = 1000):
        super().__init__()
        self.max_tokens = max_tokens

    def add_turn(self, turn: dict, query: str = "") -> None:
        self.history.append(turn)
        if not query or len(self.history) <= 1:
            return  # nothing to score against yet

        scores = self._score_turns(query)
        ranked = sorted(zip(self.history, scores), key=lambda pair: pair[1], reverse=True)

        kept, dropped = [], []
        token_budget = self.max_tokens
        for turn_, score in ranked:
            t_tokens = count_tokens(turn_["text"])
            if t_tokens <= token_budget:
                kept.append(turn_)
                token_budget -= t_tokens
            else:
                dropped.append(turn_)

        # Restore original chronological order among kept turns —
        # relevance decides WHAT stays, not the order it's read back in.
        kept_set = {id(t) for t in kept}
        self.history = [t for t in self.history if id(t) in kept_set]

        if dropped:
            self._handle_dropped(dropped)

    def _handle_dropped(self, dropped: list[dict]) -> None:
        """Hook method: base RelevanceAwareStrategy discards dropped
        turns with no trace. MemorySummarizationStrategy overrides this
        one method to condense them instead — the only point where the
        two strategies' behavior actually differs."""
        pass

    def _score_turns(self, query: str) -> list[float]:
        """Hook method: pure TF-IDF cosine similarity to `query`.
        HybridScoringStrategy overrides this one method to blend in a
        recency term — every other part of add_turn (ranking, token
        budget, dropped-turn handling) is reused unchanged."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [t["text"] for t in self.history]
        vectorizer = TfidfVectorizer().fit(texts + [query])
        turn_vectors = vectorizer.transform(texts)
        query_vector = vectorizer.transform([query])
        return list(cosine_similarity(turn_vectors, query_vector).flatten())


class MemorySummarizationStrategy(RelevanceAwareStrategy):
    """Same TF-IDF relevance scoring as RelevanceAwareStrategy (inherited,
    not duplicated) — but turns that get dropped are condensed into a
    1-sentence LLM summary and kept in self.memory instead of being
    discarded outright."""

    def __init__(self, max_tokens: int = 1000, llm_client=None):
        super().__init__(max_tokens=max_tokens)
        if llm_client is None:
            raise ValueError("MemorySummarizationStrategy requires an llm_client")
        self.llm_client = llm_client
        self.memory: list[str] = []  # 1-sentence summaries of dropped turns

    def _handle_dropped(self, dropped: list[dict]) -> None:
        for turn in dropped:
            summary_prompt = (
                f"Summarize this in one short sentence, preserving any facts, "
                f"names, or dates: \"{turn['text']}\""
            )
            result = self.llm_client.generate(summary_prompt, backend="groq")
            self.memory.append(result.response)

    def get_context(self) -> str:
        base = super().get_context()
        if self.memory:
            base += "\n(Earlier context, summarized): " + " ".join(self.memory)
        return base


class HybridScoringStrategy(RelevanceAwareStrategy):
    """Week 4 depth upgrade: combines recency with TF-IDF relevance into
    one weighted score, instead of ranking by relevance alone.

    Pure relevance-aware selection has a known failure mode: a highly
    relevant turn from early in the conversation can permanently outrank
    everything since, starving out genuinely recent context the model
    may also need (e.g. "what did we just agree on?" style follow-ups).
    Blending in a recency term (normalized turn position, most recent
    turn = 1.0, oldest = close to 0.0) guards against that, at the cost
    of sometimes keeping a less-relevant-but-recent turn over a
    more-relevant-but-old one.

    Subclasses RelevanceAwareStrategy and overrides only _score_turns —
    every other part (ranking, token budget, dropped-turn handling) is
    inherited unchanged, same hook-method pattern as
    MemorySummarizationStrategy above.
    """

    def __init__(self, max_tokens: int = 1000, recency_weight: float = 0.3):
        super().__init__(max_tokens=max_tokens)
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError("recency_weight must be between 0.0 and 1.0")
        self.recency_weight = recency_weight

    def _score_turns(self, query: str) -> list[float]:
        relevance_scores = super()._score_turns(query)
        n = len(self.history)
        # normalized position: oldest turn -> ~0, most recent turn -> 1.0
        recency_scores = [(i + 1) / n for i in range(n)]
        w = self.recency_weight
        return [
            (1 - w) * rel + w * rec
            for rel, rec in zip(relevance_scores, recency_scores)
        ]


class ContextManager:
    """Factory: returns the concrete strategy instance matching
    `strategy`, so existing call sites don't need to change —
    ContextManager(strategy="sliding_window", max_turns=6) still works
    exactly as before, now backed by real subclasses instead of an
    if/elif switch inside one class.

    Implemented via __new__ returning an instance of a DIFFERENT class
    than ContextManager itself — Python skips calling __init__ again in
    that case, so the returned object is a plain SlidingWindowStrategy
    (etc.) instance, not wrapped in anything extra.
    """

    def __new__(
        cls,
        strategy: str = "sliding_window",
        max_turns: int = 6,
        max_tokens: int = 1000,
        llm_client=None,
        recency_weight: float = 0.3,
    ):
        if strategy == "sliding_window":
            return SlidingWindowStrategy(max_turns=max_turns)
        elif strategy == "relevance_aware":
            return RelevanceAwareStrategy(max_tokens=max_tokens)
        elif strategy == "memory_summarization":
            return MemorySummarizationStrategy(max_tokens=max_tokens, llm_client=llm_client)
        elif strategy == "hybrid_scoring":
            return HybridScoringStrategy(max_tokens=max_tokens, recency_weight=recency_weight)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")


if __name__ == "__main__":
    # Quick manual check with the sliding-window strategy on 8 fake turns.
    cm = ContextManager(strategy="sliding_window", max_turns=3)
    for i in range(8):
        cm.add_turn({"speaker": "A", "text": f"This is turn number {i}."})
    print("Kept turns after sliding window (max_turns=3):")
    print(cm.get_context())
