"""
Adaptive Context Manager: decides which past conversation turns stay in
context as a multi-turn session grows.

NOTE (stand-in flag): Dania owns this per the plan (Week 1 Thu/Fri +
Week 2 Mon/Tue tasks). Built as one class with a `strategy` switch
instead of three separate classes — see the architecture note below.

DECISION WORTH FLAGGING: the plan describes sliding-window, relevance-
aware, and memory-summarization as three successive versions built on
different days. Building them as three separate classes would mean
duplicating the "add a turn, enforce a token budget" bookkeeping three
times. Instead this is ONE class with a `strategy` parameter — sliding-
window and relevance-aware differ only in which turns get kept;
memory-summarization differs only in what happens to the turns that
DON'T get kept (discard vs. LLM-summarize). Alternative considered:
Strategy-pattern subclasses (SlidingWindowManager, RelevanceAwareManager,
...) — more "textbook," but overkill for 3 closely related variants that
share this much logic; would be worth it if a 4th, structurally
different strategy shows up later.
"""

from token_analysis.counter import count_tokens


class ContextManager:
    def __init__(
        self,
        strategy: str = "sliding_window",
        max_turns: int = 6,
        max_tokens: int = 1000,
        llm_client=None,
    ):
        """
        strategy: "sliding_window" | "relevance_aware" | "memory_summarization"
        max_turns: for sliding_window — keep the last N turns.
        max_tokens: for relevance_aware / memory_summarization — keep
            turns (by relevance) until this token budget is used up.
        llm_client: required only for "memory_summarization" (needs to
            call an LLM to summarize dropped turns). Pass an LLMClient
            instance from pipeline.llm_clients.
        """
        if strategy not in ("sliding_window", "relevance_aware", "memory_summarization"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        if strategy == "memory_summarization" and llm_client is None:
            raise ValueError("memory_summarization strategy requires an llm_client")

        self.strategy = strategy
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.llm_client = llm_client

        self.history: list[dict] = []   # turns currently kept in context
        self.memory: list[str] = []     # 1-sentence summaries of dropped turns (memory_summarization only)

    def add_turn(self, turn: dict, query: str = "") -> None:
        """Adds one turn (expects at least a 'text' key — matches
        LoCoMo's {"speaker", "dia_id", "text"} shape) and re-applies the
        active strategy. `query` is the current question being asked —
        only used by relevance_aware/memory_summarization to score
        relevance against."""
        self.history.append(turn)

        if self.strategy == "sliding_window":
            self._apply_sliding_window()
        else:
            self._apply_relevance_aware(query)

    def _apply_sliding_window(self) -> None:
        """Simplest strategy, and the Week 1 fallback/control condition:
        keep only the last N turns, drop everything older with no
        record kept at all."""
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def _apply_relevance_aware(self, query: str) -> None:
        """Scores every turn's relevance to `query` via TF-IDF cosine
        similarity (reuses the same approach as the Prompt Compression
        Module for consistency — see prompt_compression/compressor.py),
        keeps turns in relevance order until max_tokens is spent.

        If strategy == 'memory_summarization', turns that get dropped
        are condensed into a 1-sentence LLM summary and appended to
        self.memory instead of being discarded outright.
        """
        if not query or len(self.history) <= 1:
            return  # nothing to score against yet

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [t["text"] for t in self.history]
        vectorizer = TfidfVectorizer().fit(texts + [query])
        turn_vectors = vectorizer.transform(texts)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(turn_vectors, query_vector).flatten()

        ranked = sorted(zip(self.history, scores), key=lambda pair: pair[1], reverse=True)

        kept, dropped = [], []
        token_budget = self.max_tokens
        for turn, score in ranked:
            t_tokens = count_tokens(turn["text"])
            if t_tokens <= token_budget:
                kept.append(turn)
                token_budget -= t_tokens
            else:
                dropped.append(turn)

        # Restore original chronological order among kept turns —
        # relevance decides WHAT stays, not the order it's read back in.
        kept_set = {id(t) for t in kept}
        self.history = [t for t in self.history if id(t) in kept_set]

        if self.strategy == "memory_summarization" and dropped:
            for turn in dropped:
                summary_prompt = (
                    f"Summarize this in one short sentence, preserving any facts, "
                    f"names, or dates: \"{turn['text']}\""
                )
                result = self.llm_client.generate(summary_prompt, backend="groq")
                self.memory.append(result["response"])

    def get_context(self) -> str:
        """Returns the current context as a single string: kept turns
        (chronological) + any memory summaries, ready to prepend to a
        query sent to the LLM."""
        parts = [f"{t.get('speaker', '')}: {t['text']}" for t in self.history]
        if self.memory:
            parts.append("(Earlier context, summarized): " + " ".join(self.memory))
        return "\n".join(parts)

    def total_tokens(self) -> int:
        return count_tokens(self.get_context())


if __name__ == "__main__":
    # Quick manual check with the sliding-window strategy on 8 fake turns.
    cm = ContextManager(strategy="sliding_window", max_turns=3)
    for i in range(8):
        cm.add_turn({"speaker": "A", "text": f"This is turn number {i}."})
    print("Kept turns after sliding window (max_turns=3):")
    print(cm.get_context())
