# Pipeline and Integration: Final Summary (FYP I)

Status: describes the code as built. **TODO(Faiqa)** marks items to confirm.

## 1. Flow

```
                 +--> single-turn: context string
query ---------->|
                 +--> multi-turn: session (list of turns)

single-turn : [Compressor] -> prompt -> LLM -> TokenAnalyzer
multi-turn  : ContextManager -> [Compressor] -> prompt -> LLM -> TokenAnalyzer
              (stage order configurable, see section 4)
```

`Pipeline.process(query=..., context="" , session=None)` is the one entry point.
It routes by input type: a `session` selects the multi-turn path, otherwise the
single-turn path. `run()` and `run_conversation()` are thin backward-compatible
wrappers that build a `PipelineConfig` and call `process()`.

## 2. Components

| Component | Owner | Used for |
|---|---|---|
| `Pipeline` (`pipeline/pipeline.py`) | Faiqa | orchestration, cost accounting |
| `PipelineConfig`, `model_for_backend` (`pipeline/config.py`) | Faiqa | settings, model per backend |
| `build_prompt` (`pipeline/prompts.py`) | Faiqa | task-specific instructions |
| `LLMClient` (`pipeline/llm_clients.py`) | Faiqa | Groq / OpenAI calls with retry |
| Compressors (`prompt_compression/`) | Faiqa | see the compression summary |
| `ContextManager` (`context_manager/manager.py`) | Dania | selects turns for multi-turn input |
| `TokenAnalyzer`, `count_tokens`, `estimate_cost` (`token_analysis/`) | Easha | tokens, cost, run log |
| `evaluation/run_experiment.py` and metrics | Easha | experiment loop, EM/F1, retention |

Components can be injected (`Pipeline(llm=..., analyzer=..., compressor=...,
context_manager=...)`) so tests run without network access.

## 3. Configuration (`PipelineConfig`)

| Field | Default | Meaning |
|---|---|---|
| `compression_enabled` | False | turn compression on |
| `compression_method` | "selective" | extractive / selective / drop / rewrite |
| `compression_scorer` | "bi_encoder" | tfidf / bi_encoder / cross_encoder |
| `compression_level` | 0.5 | token-budget fraction |
| `compression_window_size` | 2 | sentences per window (2 or 3) |
| `backend` | "groq" | groq or openai |
| `context_strategy` | "sliding_window" | multi-turn strategy |
| `max_turns`, `max_tokens` | 6, 1000 | context manager limits |
| `stage_order` | "manager_then_compressor" | multi-turn stage order |
| `task_type` | "qa" | prompt template |

Note: `run()` defaults to method "extractive" and scorer "tfidf", which differ from
the config defaults above. Experiments pass these explicitly.

## 4. Multi-turn stage order

- `manager_then_compressor` (default): the context manager selects turns, then the
  compressor shrinks that assembled text. The compressor only sees turns that
  survived selection.
- `compressor_then_manager` (experimental): the whole session is compressed as one
  text and passed to the manager as a single turn.

**TODO(Faiqa):** neither order has been evaluated on LoCoMo in the material reviewed
here. Do not claim one is better until it is run.

## 5. Prompts

All templates live in `prompts.py`. The `qa` and `conversation` templates tell the
model to answer from the context only, in as few words as possible, with no
explanation. Before it was added, long answers were penalised by EM and F1; in Easha's tests it raised
baseline F1 from 0.27 to 0.66. Numbers produced before it was added are not comparable and
must be rerun. With no context, the bare query is sent.

## 6. Accounting

Each call returns the answer-call token counts and cost from `TokenAnalyzer`, plus,
when compression ran, the `CompressionResult` and `compression_cost_usd`.
`end_to_end_cost_usd` is the answer cost plus the compression cost, so LLM-based
compression is charged for its own calls.

## 7. Running experiments

`evaluation/run_experiment.py` calls `run()` per sample and writes one CSV row per
sample (token reduction, retention, costs, EM, F1, the LLM response). It saves each
row as it finishes and resumes a partly finished file by skipping existing sample ids.
The compressor and the embedding models are cached per process, so a 300-sample run
loads each model once.

## 8. Known gaps

- `Pipeline(evaluator=...)` is accepted but unused: answer scoring is done in
  `run_experiment.py`, not in the pipeline.
- No input validation: an empty query, or both `context` and `session`, is not
  rejected. (`session=[]` still takes the multi-turn path.)
- The model-name mapping exists in `config.model_for_backend`, `compressor.py`
  (`BACKEND_MODEL`) and `run_experiment.py` (`BACKEND_MODEL`). They agree today;
  consolidate on `model_for_backend`.
- LoCoMo: image captions (15.5% of turns) are not read, and relative dates are not
  converted to calendar dates (Dania's findings).
- Whole-text LLM condensing is not selectable through `build_compressor`.
- **TODO(Faiqa):** record the result of `scripts/fresh_clone_check.py` and the
  `v1.0-fyp1` tag here once done.