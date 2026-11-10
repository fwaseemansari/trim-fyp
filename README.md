# TRIM: Token Reduction & Intelligent Management

TRIM reduces the tokens sent to an LLM and measures what that costs in answer quality. Three modules sit behind one pipeline.

## Modules

| Module | Folder | What it does |
|---|---|---|
| Prompt Compression | `prompt_compression/` | Scores sentence windows against the question and keeps the most relevant ones under a token budget (TF-IDF, bi-encoder, or cross-encoder scoring; optional LLM rewrite of dropped text). |
| Context Manager | `context_manager/` | For multi-turn conversations, decides which turns stay in context (sliding window, relevance aware, memory summarization, hybrid). |
| Token Analysis | `token_analysis/` | Counts tokens per model, estimates cost, and logs every run. |

`pipeline/` ties the three modules together. `evaluation/` runs the experiments and scores the answers.

## Repository layout
prompt_compression/ compressor.py (factory + compressors), segmenters.py, scorers.py,
results.py (CompressionResult)
context_manager/ manager.py (ContextManager + strategies)
token_analysis/ counter.py (count_tokens, TokenAnalyzer), pricing.py
pipeline/ pipeline.py (Pipeline.process, run, run_conversation), config.py,
prompts.py, llm_clients.py (Groq / OpenAI backends)
evaluation/ run_experiment.py, quality_metrics.py, metrics.py, compare_accuracy.py,
summarize_results.py, hallucination_audit.py, compare_runs.py,
noise_floor.py, splitter_precheck.py, rescore_results.py,
results/ (CSV outputs)
data/ squad_300.json, locomo10.json, cnn_sample.json, prepare_datasets.py
docs/ design notes and final summaries (start with the two *_final_summary.md)
scripts/ check_requirements.py, fresh_clone_check.py
tests/ unit tests (no network needed unless stated)


## Setup

Requires Python 3.10+ (developed on 3.13).

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your keys (never commit it):