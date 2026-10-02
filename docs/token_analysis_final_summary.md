# Token Analysis Module: Final Summary (as built)

Owner: Easha. Status: complete. Last updated: Oct 2, 2026.

This document describes the Token Analysis Module as it was actually built for FYP-1, and notes where it differs from the original plan.

## 1. Purpose

The Token Analysis Module measures how many tokens a prompt and response use, what they cost, and how long they take. It is the baseline against which the Prompt Compression Module and the Adaptive Context Manager are evaluated. Every comparison in the project (token reduction, cost reduction, latency change) goes through this module.

## 2. Files

| File | Contents |
|---|---|
| `token_analysis/counter.py` | `count_tokens()`, the `TokenAnalyzer` class, and two tokenizer helper functions |
| `token_analysis/pricing.py` | `estimate_cost()` and the per model price table |
| `evaluation/logs/token_log.csv` | Log written by `TokenAnalyzer.log_run()` |

## 3. Components

### 3.1 `count_tokens(text, model)`
Returns the number of tokens in a text for a given model. OpenAI models use `tiktoken`. Models served on Groq use the matching HuggingFace tokenizer through `AutoTokenizer`. The default model is `gpt-4o-mini`. Two private helpers, `_get_tiktoken_encoder` and `_get_hf_tokenizer`, load the tokenizers.

### 3.2 `estimate_cost(model, input_tokens, output_tokens)`
Returns the estimated cost in US dollars, using published per 1,000 token prices stored in the `PRICING` dictionary in `pricing.py`. Input and output tokens are priced separately. Prices were checked in September 2026 from the OpenAI and Groq pricing pages. If a model has no entry, the function raises a clear `KeyError` instead of returning a wrong cost.

| Model | Input (USD per 1K tokens) | Output (USD per 1K tokens) |
|---|---|---|
| `gpt-4o-mini` | 0.00015 | 0.0006 |
| `openai/gpt-oss-20b` (Groq) | 0.000075 | 0.0003 |

### 3.3 `TokenAnalyzer`
| Method | What it does |
|---|---|
| `__init__(log_path)` | Sets the CSV log location. Default is `evaluation/logs/token_log.csv`. |
| `analyze(prompt, response, model)` | Returns a dictionary with input tokens, output tokens, total tokens and estimated cost for one prompt and response pair. |
| `log_run(prompt, response, backend, model, latency_ms)` | Runs the analysis and appends one row to the CSV log (timestamp, sizes, tokens, cost, latency). Returns the same values as a dictionary. |
| `compare(baseline_log, treatment_log)` | Compares two runs and returns token reduction %, cost reduction % and the latency difference. |

## 4. How other parts of the project use it

1. `pipeline/pipeline.py` calls `log_run()` for every query, so each run is recorded in one place.
2. `evaluation/metrics.py` uses `count_tokens()` and `estimate_cost()` to compute token reduction %, compression ratio and cost reduction %.
3. `evaluation/run_experiment.py` (SQuAD) and `evaluation/run_cnn_experiment.py` (CNN/DailyMail) use the same functions, so every experiment is measured identically.

## 5. Validation

1. Baseline run on SQuAD v2 (48 samples, Groq): average 217 total tokens per query, average cost $0.000017 per query, average latency 617 ms.
2. Token reduction measured at 23.0%, 40.0% and 61.7% for keep 70%, 50% and 30% of sentences.
3. Unit tests in `tests/` check the accuracy functions and the pipeline.

## 6. Differences from the original plan

| Plan | As built |
|---|---|
| Groq backend with Llama 3.3 70B | Experiments ran on `openai/gpt-oss-20b` on Groq, chosen to fit the free tier. Token counting still uses the matching tokenizer for the model used. |
| OpenAI backend run alongside Groq | Wired in, but not run in experiments because no OpenAI credits were available. Deferred. |
| `log_run(prompt, response, backend, latency_ms)` | An extra `model` argument was added so the right tokenizer and price are used. |
| Baseline from Week 1 | Re generated after the short answer prompt change. The old file is kept as `baseline_results_OLD_superseded.json`. |
| Token reduction on whole prompt | Measured on the context passage only. The question, instruction and answer are not compressed, so real per request savings are smaller. |

## 7. Known limitations

1. Prices are hard coded in `pricing.py` and must be updated by hand if providers change them. Only `gpt-4o-mini` and `openai/gpt-oss-20b` have entries, so using another model (for example Llama 3.3 70B) requires adding it first.
2. Cost figures are estimates from token counts, not provider invoices.
3. Cross backend comparison (OpenAI vs Groq) is not yet done.
4. The local HuggingFace model backend was not built (stretch goal, planned for FYP-2).