# TRIM: Token Reduction & Intelligent Management

TRIM reduces the tokens sent to an LLM, and measures what that costs in answer
quality. It has three modules behind one pipeline:

| Module | Folder | What it does |
|---|---|---|
| Prompt Compression | `prompt_compression/` | Scores sentence windows against the question and keeps the most relevant ones under a token budget (TF-IDF, bi-encoder or cross-encoder scoring; optional LLM rewrite of dropped text). |
| Context Manager | `context_manager/` | For multi-turn conversations, decides which turns stay in the context (sliding window, relevance-aware, memory summarization, hybrid). |
| Token Analysis | `token_analysis/` | Counts tokens per model, estimates cost, and logs every run. |

`pipeline/` ties them together; `evaluation/` runs the experiments and scores
the answers.

## Repository layout

```
prompt_compression/   compressor.py (factory + compressors), segmenters.py, scorers.py,
                      selectors.py, results.py (CompressionResult)
context_manager/      manager.py (ContextManager + strategies)
token_analysis/       counter.py (count_tokens, TokenAnalyzer), pricing.py
pipeline/             pipeline.py (Pipeline.process, run, run_conversation), config.py,
                      prompts.py, llm_clients.py (Groq / OpenAI backends)
evaluation/           run_experiment.py, quality_metrics.py, metrics.py, compare_accuracy.py,
                      summarize_results.py, hallucination_audit.py, results/ (CSV outputs)
data/                 squad_300.json, locomo10.json, cnn_sample.json, prepare_datasets.py
docs/                 design notes and final summaries (start with the two *_final_summary.md)
scripts/              check_requirements.py, fresh_clone_check.py
tests/                unit tests (no network needed unless stated)
```

## Setup

Requires Python 3.10+ (developed on 3.13).

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your keys (never commit it):

```
GROQ_API_KEY=...
OPENAI_API_KEY=...
```

The first run downloads the embedding models (`all-MiniLM-L6-v2` and
`cross-encoder/ms-marco-MiniLM-L-6-v2`) from the Hugging Face Hub, and the
tokenizers used for token counting.

## Quick start

```python
from pipeline.pipeline import run

result = run(
    query="Who was Jacksonville's mayor at the time of the consolidation?",
    context=open("some_passage.txt").read(),
    compression_enabled=True,
    compression_method="selective",
    compression_scorer="cross_encoder",
    compression_level=0.5,
)
print(result["response"], result["total_tokens"], result["end_to_end_cost_usd"])
```

Multi-turn input goes through `run_conversation(session, query, ...)` or
`Pipeline(config=...).process(query=..., session=...)`.

## Run the experiments

All experiments write one CSV row per sample to `evaluation/results/`.
`run_experiment` resumes: rerunning skips sample ids already in the output file.

```powershell
# edit the arguments at the bottom of evaluation/run_experiment.py, then:
python -m evaluation.run_experiment
```

Typical set for the SQuAD comparison (300 samples): one baseline run with
`compression_enabled=False`, then each scorer (`tfidf`, `bi_encoder`,
`cross_encoder`) at levels 0.3, 0.5 and 0.7. Compare each run against the
baseline with `evaluation/compare_accuracy.py`; build tables and charts with
`evaluation/summarize_results.py`.

Hallucination audit (blind labelling; see the module docstring for the labels):

```powershell
python -m evaluation.hallucination_audit sample --results <run CSVs> --samples data/squad_300.json --n 30
python -m evaluation.hallucination_audit summarize --sheet evaluation/results/audit_sheet_labeled.csv --key evaluation/results/audit_sheet_key.csv
```

Free-tier Groq has a daily token limit. A 300-sample run can exceed it; the
client waits when the API says how long, and resuming picks up where it stopped.

## Tests and checks

```powershell
python -m pytest -q --ignore=pipeline/test_pipeline.py   # skips tests that call live APIs
python scripts/check_requirements.py                     # imports vs requirements.txt
python scripts/fresh_clone_check.py                      # commit first; clones, installs, tests
```

## Headline results (SQuAD v2, 300 answerable questions, Groq gpt-oss-20b)

Paired against a no-compression baseline (F1 0.768). Token reduction is
measured on the context passage only, not the whole request.

| Scorer | Level | Context-token reduction | F1 | F1 change vs baseline |
|---|---|---|---|---|
| cross-encoder | 0.3 | 54.0% | 0.759 | -0.009 |
| cross-encoder | 0.5 | 47.5% | 0.773 | +0.005 |
| cross-encoder | 0.7 | 35.9% | 0.779 | +0.011 |
| tf-idf | 0.3 | 54.6% | 0.723 | -0.045 |
| bi-encoder | 0.5 | 47.3% | 0.725 | -0.044 |

The cross-encoder shows no detectable accuracy loss at about 54% context-token
reduction; the tf-idf and bi-encoder drops of about 0.04 are nominally
significant but do not survive correction for multiple comparisons. Scores use
Unicode-aware answer normalization (`evaluation/quality_metrics.py`). Full table,
confidence intervals and caveats: `docs/prompt_compression_final_summary.md`.

## Known limitations

- Compression `level` is a token budget, not a guarantee: achieved reduction is
  lower than the nominal level at high levels and the best window is always kept.
- The regex sentence splitter mis-splits abbreviations such as "Dr." or "St.".
- The LoCoMo pipeline ignores image captions (15.5% of turns) and does not
  convert relative dates ("last Saturday") to calendar dates.

## Team

Prompt Compression, pipeline and integration: Faiqa. Token Analysis and
evaluation metrics: Easha. Context Manager and datasets: Dania.
