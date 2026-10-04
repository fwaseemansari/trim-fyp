"""Pytest configuration: tests that call real LLM APIs are opt-in.

Set RUN_LIVE_TESTS=1 to run them (needs valid keys in .env).
"""

import os

import pytest

LIVE_TEST_FILES = {"test_latency.py", "test_pipeline.py"}  # make real Groq/OpenAI calls


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live API test; set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if item.path.name in LIVE_TEST_FILES and "pipeline" in item.path.parts:
            item.add_marker(skip)