# Token Analysis Module — Design Doc

**Status:** stand-in draft, built as part of the pipeline scaffold — Easha to review/own and adjust once she starts.

## Purpose
Single source of truth for token counts, cost, and latency across every
run of the pipeline, regardless of which backend, compression method,
or context strategy is active. Every other module (Compression, Context
Manager) is evaluated by comparing its `TokenAnalyzer` numbers against
the Week 1 baseline.

## Inputs
- `prompt: str` — the full text sent to the LLM (after any compression/
  context-management has already been applied).
- `response: str` — the text the LLM returned.
- `backend: str` — `"groq"` or `"openai"`.
- `model: str` — the specific model string used (for cost lookup).
- `latency_ms: float` — wall-clock time for the API call, from
  `LLMClient.generate()`.

## Outputs
- `analyze()` returns a dict: `input_tokens`, `output_tokens`,
  `total_tokens`, `cost_usd`.
- `log_run()` returns the same dict, and additionally appends one row
  to `evaluation/logs/<name>.csv` with a timestamp and prompt length —
  this CSV is the persistent record every experiment script reads back.
- `compare()` returns `token_reduction_pct`, `cost_reduction_pct`,
  `latency_delta_ms` between two `analyze()`/`log_run()` result dicts.

## Class diagram (sketch)

```
+---------------------------+
|      TokenAnalyzer        |
+---------------------------+
| - log_path: str           |
+---------------------------+
| + analyze(prompt,         |
|     response, model)      |
|     -> dict                |
| + log_run(prompt,         |
|     response, backend,    |
|     model, latency_ms)    |
|     -> dict                |
| + compare(baseline_log,   |
|     treatment_log)        |
|     -> dict                |
+---------------------------+
        |
        | uses
        v
+---------------------------+       +---------------------------+
|  count_tokens(text,model) |       |   estimate_cost(model,    |
|  (token_analysis/counter) |       |   input_tok, output_tok)  |
|                           |       |   (token_analysis/pricing)|
+---------------------------+       +---------------------------+
```

## Known limitation (flag for report's Limitations section)
Token counting uses `tiktoken`'s `cl100k_base` encoding for BOTH
backends as an approximation, not each backend's exact tokenizer — see
the design note at the top of `token_analysis/counter.py` for the
reasoning and the exact-tokenizer alternative that was considered and
deferred.
