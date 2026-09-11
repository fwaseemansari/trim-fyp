"""
Prompt Compression Module — Faiqa's core deliverable.

Two methods behind one interface: compress(text, query, method, level).

DECISION WORTH FLAGGING — nltk vs. regex sentence splitting: the plan
suggests nltk.sent_tokenize() OR a simple regex split. nltk's tokenizer
needs a one-time `nltk.download('punkt')` (network call to nltk's
servers, plus a local data file) — a regex split has zero setup cost
and is "good enough" for English prose. Went with regex here to avoid
adding a download step to your setup today; swap in nltk if compression
quality on messier text (LoCoMo dialogue especially) turns out to need
better sentence boundaries than the regex catches.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _split_sentences(text: str) -> list[str]:
    """Regex-based sentence split — see module docstring for why this
    was chosen over nltk.sent_tokenize(). Splits on '.', '!', '?'
    followed by whitespace, while trying not to break on common
    abbreviations (Mr., Dr., etc.) or decimal numbers."""
    text = text.strip()
    if not text:
        return []
    # Negative lookbehind for a handful of common abbreviations, and for
    # a digit (to avoid splitting "3.14" or "No. 5" mid-number).
    pattern = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bJr)(?<!\bSt)(?<!\d)[.!?]+\s+"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _extractive_compress(text: str, query: str, keep_ratio: float) -> str:
    """Baseline 1: split into sentences, rank by TF-IDF relevance to
    the query, keep the top keep_ratio fraction."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text  # nothing meaningful to compress

    vectorizer = TfidfVectorizer().fit(sentences + [query])
    sentence_vectors = vectorizer.transform(sentences)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(sentence_vectors, query_vector).flatten()

    n_keep = max(1, round(len(sentences) * keep_ratio))
    # Rank by score, but keep the ORIGINAL order among the kept
    # sentences — an extractive summary that scrambles sentence order
    # reads worse and can break factual continuity (e.g. pronoun
    # references to something described 2 sentences earlier).
    top_indices = sorted(
        sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:n_keep]
    )
    return " ".join(sentences[i] for i in top_indices)


def _llm_compress(text: str, query: str, keep_ratio: float, llm_client) -> str:
    """Baseline 2: ask the LLM itself to condense the text, preserving
    facts needed to answer questions about it. Uses the cheap backend
    per the plan (gpt-4o-mini) — falls back to Groq if no working
    OpenAI key is configured, so this stays runnable without OpenAI
    credits; swap back once OpenAI is funded, since gpt-4o-mini is the
    plan's specified backend for this method."""
    target_pct = round(keep_ratio * 100)
    prompt = (
        f"Condense the following text to roughly {target_pct}% of its "
        f"original length, in the fewest tokens possible, while "
        f"preserving all facts needed to answer questions about it "
        f"(especially anything relevant to: \"{query}\").\n\n"
        f"Text:\n{text}\n\nCondensed version:"
    )
    backend = "openai" if llm_client.has_openai else "groq"
    result = llm_client.generate(prompt, backend=backend)
    return result["response"]


def compress(text: str, query: str = "", method: str = "extractive", level: float = 0.5, llm_client=None) -> str:
    """
    text: the passage/context to compress.
    query: the question being asked about `text` — both methods use
        this to decide what's worth keeping.
    method: "extractive" | "llm"
    level: fraction of original content to target keeping (e.g. 0.5 =
        keep ~50%). Matches the plan's "30%/50%/70%" sweep language.
    llm_client: required only for method="llm".
    """
    if method == "extractive":
        return _extractive_compress(text, query, level)
    elif method == "llm":
        if llm_client is None:
            raise ValueError("method='llm' requires an llm_client")
        return _llm_compress(text, query, level, llm_client)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'extractive' or 'llm'.")


if __name__ == "__main__":
    sample_text = (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was designed by Gustave Eiffel and completed in 1889. "
        "It stands 330 metres tall and was the tallest man-made structure "
        "in the world for 41 years. Millions of tourists visit it every year. "
        "The tower is named after the engineer whose company designed it."
    )
    sample_query = "How tall is the Eiffel Tower?"

    print("--- Original ---")
    print(sample_text)
    print("\n--- Extractive compression (level=0.4) ---")
    print(compress(sample_text, sample_query, method="extractive", level=0.4))
