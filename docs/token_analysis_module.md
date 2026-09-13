# Token Analysis Module

> Part of **TRIM** — Intelligent Token Optimization Framework for LLM Applications
> **Owner:** Easha | **Status:** Week 1 complete (Mon–Wed) | **Test coverage:** 26/26 passing

---

## Overview

The Token Analysis Module is TRIM's **single source of truth** for measuring token usage and cost. Every other module in the system — Prompt Compression, Adaptive Context Manager — is evaluated by comparing its output against numbers produced here. If this module's counts drift or are inconsistent, every downstream metric in the project (compression ratio, cost savings, token reduction %) becomes unreliable.

In short: **nothing gets measured in TRIM without going through this module first.**

---

## Architecture

```
                    ┌─────────────────────┐
                    │   count_tokens()    │
                    │  (per-backend       │
                    │   tokenizer router) │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                            │
          tiktoken (OpenAI)         HF AutoTokenizer (Groq)
                 │                            │
                 └─────────────┬─────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    TokenAnalyzer     │
                    │  .analyze()           │──── pricing.py (cost rates)
                    │  .log_run()           │──── evaluation/logs/token_log.csv
                    │  .compare()           │
                    └───────────────────────┘
```

---

## Components

### 1. `count_tokens(text, model) -> int`
Routes to the **real tokenizer** for whichever backend is being measured — because different model families split text into tokens differently, so one universal counter would silently misreport costs.

| Backend | Tokenizer used | Why |
|---|---|---|
| OpenAI (`gpt-4o-mini`) | `tiktoken` | OpenAI's own local tokenizer — exact match to their billing, no API call needed |
| Groq (`openai/gpt-oss-20b`) | HuggingFace `AutoTokenizer` | Groq exposes no token-counting endpoint, so the model's real published tokenizer is used directly |

### 2. `TokenAnalyzer.analyze(prompt, response, model) -> dict`
Given one prompt/response pair, returns input tokens, output tokens, total tokens, and estimated USD cost (via `pricing.py`).

### 3. `TokenAnalyzer.log_run(...)`
Appends one row per LLM call to `evaluation/logs/token_log.csv` — timestamp, backend, model, token counts, cost, latency. This CSV *is* the experiment record the FYP-1/FYP-2 reports will pull numbers from.

### 4. `TokenAnalyzer.compare(baseline, treatment) -> dict`
Computes % token reduction, % cost reduction, and latency delta between two runs — the core metric used to prove compression/context-management actually works.

---

## Design Decisions Worth Noting

- **Model swap handled correctly:** the original plan assumed Groq served Llama-3.3-70B. Groq's actual served model changed team-wide to `openai/gpt-oss-20b`. The tokenizer and pricing logic were adapted to match reality rather than the outdated plan wording.
- **Real pricing, not estimates:** Groq's `gpt-oss-20b` rate was corrected from a placeholder guess to the actual published rate ($0.075/$0.30 per 1M tokens, sourced from groq.com/pricing, Sep 2026).
- **Caching:** both tokenizers are loaded once and cached, since reloading per call would be prohibitively slow once evaluation scales to hundreds of samples (Week 4).

---

## Test Coverage

| Test file | Covers | Result |
|---|---|---|
| `tests/test_token_analyzer.py` | Normal pairs, empty input, long input, invalid model, `compare()` | 20/20 passed |
| `tests/test_csv_logging.py` | File creation, header integrity, append behavior, value accuracy | 6/6 passed |

---

## Known Limitations (flagged honestly, not hidden)

- `gpt-oss-20b` token counts use its real published tokenizer — accurate. No known approximation issues remain as of this write-up.
- Groq pricing reflects *current list pricing*, not negotiated/enterprise rates — fine for an academic FYP context.