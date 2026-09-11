"""
Evaluation metrics — 4 of the proposal's 7 target metrics live here.
The other 3 (task accuracy, response quality/ROUGE-L, latency) come in
Week 3 per the plan; latency is technically already covered since
Week 1 (LLMClient.generate() returns it on every call).

NOTE (stand-in flag): Easha owns this per the plan (Wed + Fri Week 2
tasks, merged into one file here rather than built across two days).
"""

from token_analysis.counter import TokenAnalyzer


def token_reduction_pct(original_tokens: int, compressed_tokens: int) -> float:
    """% fewer tokens after compression. Positive = reduction, negative
    = compression made it LONGER (can happen with LLM-based compression
    on already-short text — worth flagging if you see this in results)."""
    if original_tokens == 0:
        return 0.0
    return round((original_tokens - compressed_tokens) / original_tokens * 100, 2)


def compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    """original / compressed. A ratio of 2.0 means the compressed
    version is half the size. Guards against divide-by-zero if
    compression somehow produced empty output."""
    if compressed_tokens == 0:
        return float("inf")
    return round(original_tokens / compressed_tokens, 3)


def cost_reduction_pct(original_cost: float, compressed_cost: float) -> float:
    if original_cost == 0:
        return 0.0
    return round((original_cost - compressed_cost) / original_cost * 100, 2)


_retention_model = None  # lazy-loaded — see information_retention_score()


def information_retention_score(original_text: str, compressed_text: str) -> float:
    """Cosine similarity between sentence-transformer embeddings of the
    original and compressed text. ~1.0 = meaning fully preserved,
    dropping toward 0 as compression discards more content.

    NOTE: first call downloads the 'all-MiniLM-L6-v2' model (~90 MB)
    from the HuggingFace Hub — needs a real internet connection the
    first time this runs, and takes a few seconds. Cached locally after
    that. Model is loaded lazily (only on first call to this function)
    so importing this module doesn't force that download just to use
    the other 3 metrics above.
    """
    global _retention_model
    if _retention_model is None:
        from sentence_transformers import SentenceTransformer
        _retention_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = _retention_model.encode([original_text, compressed_text])

    from sklearn.metrics.pairwise import cosine_similarity
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(score), 4)


def full_comparison(original_text: str, compressed_text: str, original_cost: float, compressed_cost: float) -> dict:
    """Convenience wrapper computing all 4 metrics at once for one
    original/compressed pair — what evaluation/run_experiment.py and
    evaluation/run_compression_sweep.py call per sample."""
    from token_analysis.counter import count_tokens

    orig_tokens = count_tokens(original_text)
    comp_tokens = count_tokens(compressed_text)

    return {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "token_reduction_pct": token_reduction_pct(orig_tokens, comp_tokens),
        "compression_ratio": compression_ratio(orig_tokens, comp_tokens),
        "cost_reduction_pct": cost_reduction_pct(original_cost, compressed_cost),
        "information_retention": information_retention_score(original_text, compressed_text),
    }


if __name__ == "__main__":
    original = (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was designed by Gustave Eiffel and completed in 1889. "
        "It stands 330 metres tall."
    )
    compressed = "The Eiffel Tower in Paris stands 330 metres tall, completed in 1889."
    print(full_comparison(original, compressed, original_cost=0.0002, compressed_cost=0.0001))
