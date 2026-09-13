"""
tests/test_csv_logging.py

Verify: TokenAnalyzer.log_run() correctly appends
rows to evaluation/logs/token_log.csv - the shared baseline file every
other module's results get compared against.
"""

import csv
import os

from token_analysis.counter import TokenAnalyzer

TEST_LOG_PATH = "evaluation/logs/test_token_log.csv"  # separate file, won't touch the real log

# Clean slate: remove any leftover file from a previous test run
if os.path.exists(TEST_LOG_PATH):
    os.remove(TEST_LOG_PATH)

analyzer = TokenAnalyzer(log_path=TEST_LOG_PATH)

passed = 0
failed = 0


def check(description, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {description}")
        passed += 1
    else:
        print(f"  FAIL: {description}")
        failed += 1


# --- Test 1: file gets created on first log_run() call ---
print("Test 1: File creation")
analyzer.log_run(
    prompt="What is the capital of France?",
    response="Paris.",
    backend="openai",
    model="gpt-4o-mini",
    latency_ms=342.5,
)
check("CSV file was created", os.path.exists(TEST_LOG_PATH))

# --- Test 2: header row is written correctly ---
print("\nTest 2: Header row")
with open(TEST_LOG_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    expected_columns = {
        "timestamp", "backend", "model", "prompt_length_chars",
        "input_tokens", "output_tokens", "total_tokens", "cost_usd", "latency_ms",
    }
    check("header has all expected columns", set(reader.fieldnames) == expected_columns)

# --- Test 3: second call appends a new row, doesn't overwrite or duplicate header ---
print("\nTest 3: Appending (not overwriting)")
analyzer.log_run(
    prompt="Explain photosynthesis.",
    response="Plants convert sunlight into energy using chlorophyll.",
    backend="groq",
    model="openai/gpt-oss-20b",
    latency_ms=210.0,
)
with open(TEST_LOG_PATH, "r", encoding="utf-8") as f:
    reader = list(csv.DictReader(f))
check("exactly 2 data rows after 2 calls", len(reader) == 2)
check("row 1 backend is 'openai'", reader[0]["backend"] == "openai")
check("row 2 backend is 'groq'", reader[1]["backend"] == "groq")

# --- Test 4: logged numbers match what analyze() would compute directly ---
print("\nTest 4: Logged values are accurate")
direct = analyzer.analyze("Explain photosynthesis.",
                           "Plants convert sunlight into energy using chlorophyll.",
                           "openai/gpt-oss-20b")
check(
    "CSV total_tokens matches a direct analyze() call",
    int(reader[1]["total_tokens"]) == direct["total_tokens"],
)

# --- Cleanup ---
os.remove(TEST_LOG_PATH)
print(f"\n(test file cleaned up: {TEST_LOG_PATH} removed)")

# --- Summary ---
print(f"\n{'='*40}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*40}")