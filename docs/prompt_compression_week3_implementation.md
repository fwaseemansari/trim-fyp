# Prompt Compression — Week 3 Implementation

## Architecture

Week 3 uses a composable selective-compression design:

`Segmenter -> Scorer -> BudgetSelector -> CompressionResult`

The default advanced path is:

`2-sentence overlapping windows -> bi-encoder relevance -> token-budget selection -> sentence union in original order`

Scorers implement one interface and can be swapped without changing the compressor:

- TF-IDF — Week 2 baseline
- Bi-encoder — Week 3 semantic method
- Cross-encoder — Week 4 depth upgrade

Two selective output strategies are supported:

- `selective` / `drop` — retain the highest-relevance material under the token budget.
- `rewrite` — retain selected material and use LLM rewrites for dropped chunks when budget remains.

The token budget is based on the real tokenizer selected by the backend model. At least one segment is always retained.

## Pipeline

`Pipeline.process()` is now the single orchestration entry point. `run()` and `run_conversation()` remain compatibility wrappers.

For multi-turn input, the default order is:

`Context Manager -> Prompt Compression -> LLM`

The alternative `compressor_then_manager` order is available for a later stacked experiment.

## Traceability

Compression returns `CompressionResult`, containing:

- original and compressed token counts
- selected sentence indices
- relevance scores
- method/scorer names
- number of compression LLM calls
- compression input/output tokens
- metadata such as target budget and selected windows

This prevents experiment code from recompressing a passage just to recover the text or compression statistics.

## Week 3 evaluation support

`evaluation/run_experiment.py` now accepts the compression scorer/window configuration and records SQuAD EM/F1 together with token, retention, latency and end-to-end cost fields. Compression-call tokens/cost are included when a rewrite method uses an LLM.

## Testing

Added `tests/test_prompt_compression_week3.py` covering overlapping windows, original-order sentence union, non-empty output, and rewrite-call accounting.
