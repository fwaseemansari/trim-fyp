# Weekly Log

Reused directly in the report's Implementation section later — keep
each entry to 2-3 plain bullet points, written as you go.

## Week 1 (Sep 7 - Sep 13)

**Faiqa:**
- Set up the repo scaffold, `.gitignore`/`.env` handling, and
  `pipeline/config.py` for loading API keys.
- Built `pipeline/llm_clients.py` — `LLMClient` wrapper unifying Groq
  and OpenAI behind one `.generate(prompt, backend)` call, with latency
  measurement via `time.perf_counter()`.
- Wrote `pipeline/pipeline.py`'s `run()` skeleton and
  `pipeline/test_pipeline.py` (2 pytest tests: pipeline runs without
  error, log CSV gets a new row).
- Hit and fixed 2 real bugs along the way: a stale system environment
  variable silently overriding `.env` (fixed with
  `load_dotenv(override=True)`), and Groq deprecating free-tier access
  to `llama-3.3-70b-versatile` (switched to `openai/gpt-oss-20b`).

**Easha:** *(fill in once she starts — stand-in modules for
`token_analysis/counter.py`, `token_analysis/pricing.py`,
`evaluation/run_baseline.py`, and `docs/token_analysis_module.md` are
already in place for her to review/take over)*

**Dania:** *(fill in once she starts — stand-in modules for
`context_manager/manager.py` (sliding-window strategy),
`context_manager/test_manager.py`, `data/prepare_datasets.py`, and
`docs/dataset_notes.md` are already in place for her to review/take
over)*

## Week 2 (Sep 14 - Sep 20)

**Faiqa:**
- Built `prompt_compression/compressor.py` — extractive (TF-IDF
  sentence ranking) and LLM-based compression behind one
  `compress(text, query, method, level)` interface.
- Wired compression into `pipeline.py` via a `compression_enabled` flag
  for controlled on/off comparison.
- Added configurable compression levels (30/50/70% keep-ratio) and a
  sweep script (`evaluation/run_compression_sweep.py`).

**Easha:** *(fill in — stand-ins: `evaluation/metrics.py` with token
reduction %, compression ratio, cost reduction, and information
retention score; `evaluation/run_experiment.py`)*

**Dania:** *(fill in — stand-ins: `ContextManager` upgraded with
`relevance_aware` and `memory_summarization` strategies;
`pipeline.run_conversation()`; `docs/context_manager_module.md`)*
