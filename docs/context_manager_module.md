# Adaptive Context Manager — Design Doc

**Status:** stand-in draft — Dania to review/own once she starts.

## Purpose
Decides which past conversation turns stay in the LLM's context window
as a multi-turn session (LoCoMo-style) grows past what can reasonably
be sent in full. Three strategies, same interface, so they can be
swapped and compared under identical conditions.

## Data flow
```
incoming turn -> add_turn(turn, query)
                    |
                    v
         [strategy == sliding_window?] --yes--> keep last N turns, drop rest (no record kept)
                    | no
                    v
         score every turn's TF-IDF relevance to `query`
                    |
                    v
         keep turns (highest relevance first) until max_tokens spent
                    |
                    v
         [strategy == memory_summarization?] --yes--> dropped turns -> 1-sentence LLM summary -> self.memory
                    | no
                    v
         dropped turns discarded outright
```

## Inputs
- `turn: dict` — `{"speaker": ..., "text": ...}` (matches LoCoMo's
  turn shape directly, so sessions load in with no reformatting).
- `query: str` — the question the context is being assembled to help
  answer. Required for `relevance_aware`/`memory_summarization`; unused
  by `sliding_window`.

## Outputs
- `get_context() -> str` — the assembled context string, ready to
  prepend to a prompt.
- `total_tokens() -> int` — current context size, via the shared
  `token_analysis.counter.count_tokens()` (same counting logic the rest
  of the pipeline uses, so numbers are comparable).

## Strategy comparison

| Strategy | What decides what stays | Handles dropped turns |
|---|---|---|
| `sliding_window` | Recency only (last N turns) | Discarded, no trace |
| `relevance_aware` | TF-IDF similarity to current query | Discarded, no trace |
| `memory_summarization` | TF-IDF similarity to current query | Condensed to 1 sentence, kept in `self.memory` |

## Known limitation (flag for report's Limitations section)
`relevance_aware`/`memory_summarization` re-score ALL turns seen so far
against the query every time a new turn is added — this is O(n) work
per turn, so O(n²) over a full session. Fine at LoCoMo's scale (session
turn counts stay well under a few hundred), would need a smarter
incremental-scoring approach at much larger scale.
