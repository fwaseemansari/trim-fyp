"""
Pipeline orchestration — the entry point that wires together every
other module. Two entry points:

  run(query, ...)              — single-turn (SQuAD-style: one question,
                                  optional context passage).
  run_conversation(session,...) — multi-turn (LoCoMo-style: a list of
                                  turns + a question at the end, routed
                                  through ContextManager).

DECISION WORTH FLAGGING — one pipeline.py with two entry points vs. two
separate pipeline files: the plan introduces run() in Week 1 (Faiqa) and
run_conversation() in Week 2 (Dania), on different days, by different
people. Kept them in the same file because they share the exact same
tail end (LLMClient.generate -> TokenAnalyzer.log_run) — splitting them
would mean copy-pasting that logging logic twice and the two ever
silently drifting apart in what they log. Alternative: a shared
_log_and_return() helper in a third file if this file starts feeling
crowded once Week 3's full integration (TokenAnalyzer -> Compressor ->
ContextManager -> LLM -> Response Evaluation, all in one call) lands.
"""

from pipeline.llm_clients import LLMClient
from token_analysis.counter import TokenAnalyzer
from prompt_compression.compressor import compress
from context_manager.manager import ContextManager


def run(
    query: str,
    context: str = "",
    compression_enabled: bool = False,
    compression_method: str = "extractive",
    compression_level: float = 0.5,
    backend: str = "groq",
    llm_client: LLMClient = None,
    analyzer: TokenAnalyzer = None,
) -> dict:
    """Single-turn pipeline: query (+ optional context passage) -> LLM -> logged metrics.

    compression_enabled: when True, `context` is compressed before being
    sent to the LLM — this flag is what lets you run the SAME query with
    and without compression for a controlled comparison (Week 2 Wed task).
    When False, the pipeline behaves exactly like Week 1's plain
    skeleton: query -> LLM -> log, no compression stage at all.
    """
    llm_client = llm_client or LLMClient()
    analyzer = analyzer or TokenAnalyzer()

    if context and compression_enabled:
        context = compress(context, query=query, method=compression_method, level=compression_level)

    full_prompt = f"Context: {context}\n\nQuestion: {query}" if context else query

    result = llm_client.generate(full_prompt, backend=backend)
    model = "openai/gpt-oss-20b" if backend == "groq" else "gpt-4o-mini"

    analysis = analyzer.log_run(
        prompt=full_prompt,
        response=result["response"],
        backend=backend,
        model=model,
        latency_ms=result["latency_ms"],
    )

    return {
        "query": query,
        "response": result["response"],
        "latency_ms": result["latency_ms"],
        **analysis,
    }


def run_conversation(
    session: list,
    query: str,
    context_strategy: str = "sliding_window",
    max_turns: int = 6,
    max_tokens: int = 1000,
    backend: str = "groq",
    llm_client: LLMClient = None,
    analyzer: TokenAnalyzer = None,
) -> dict:
    """Multi-turn pipeline: loops a LoCoMo-style session (list of
    {"speaker", "text"} turns) through ContextManager one turn at a
    time, logging tokens per turn via TokenAnalyzer, then answers
    `query` using whatever context survived the strategy.

    Returns the final answer plus a `per_turn_tokens` list so you can
    see how context size evolved turn-by-turn (this is the data Week 2
    Sat's demo and Week 3's "where does sliding-window start failing"
    analysis both need).
    """
    llm_client = llm_client or LLMClient()
    analyzer = analyzer or TokenAnalyzer()
    model = "openai/gpt-oss-20b" if backend == "groq" else "gpt-4o-mini"

    cm_kwargs = {"strategy": context_strategy, "max_turns": max_turns, "max_tokens": max_tokens}
    if context_strategy == "memory_summarization":
        cm_kwargs["llm_client"] = llm_client
    cm = ContextManager(**cm_kwargs)

    per_turn_tokens = []
    for turn in session:
        cm.add_turn(turn, query=query)
        per_turn_tokens.append(cm.total_tokens())

    full_prompt = f"Conversation so far:\n{cm.get_context()}\n\nQuestion: {query}"
    result = llm_client.generate(full_prompt, backend=backend)

    analysis = analyzer.log_run(
        prompt=full_prompt,
        response=result["response"],
        backend=backend,
        model=model,
        latency_ms=result["latency_ms"],
    )

    return {
        "query": query,
        "response": result["response"],
        "latency_ms": result["latency_ms"],
        "per_turn_tokens": per_turn_tokens,
        "context_strategy": context_strategy,
        **analysis,
    }


if __name__ == "__main__":
    # Manual smoke test, single-turn, no compression.
    out = run("What is the capital of France?")
    print(out)
