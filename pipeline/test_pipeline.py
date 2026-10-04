"""
pytest tests for pipeline.py — Fri Week 1 task: confirm the pipeline
runs without error and that logging actually appends a CSV row.

Run with: pytest pipeline/test_pipeline.py -v

NOTE: these make REAL Groq API calls (needs a valid GROQ_API_KEY in
.env) — not mocked. Fine for now at this scale (a handful of calls),
worth revisiting (see the earlier code-review conversation) if these
end up running automatically on every push later.
"""

import csv
import os

from pipeline.pipeline import run
from token_analysis.counter import TokenAnalyzer


def test_pipeline_runs_without_error():
    result = run("Say hi in one word.", backend="groq")
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0


def test_log_gets_new_row():
    test_log_path = "evaluation/logs/test_pipeline_log.csv"
    analyzer = TokenAnalyzer(log_path=test_log_path)

    rows_before = 0
    if os.path.isfile(test_log_path):
        with open(test_log_path, encoding="utf-8") as f:
            rows_before = sum(1 for _ in csv.reader(f)) - 1  # minus header

    run("Say hi in one word.", backend="groq", analyzer=analyzer)

    with open(test_log_path, encoding="utf-8") as f:
        rows_after = sum(1 for _ in csv.reader(f)) - 1

    assert rows_after == rows_before + 1

    # cleanup so repeated test runs don't pile up rows in a test-only file
    os.remove(test_log_path)
