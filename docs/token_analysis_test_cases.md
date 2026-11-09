# Token Analysis Module — Formal Test Cases

Chapter 7 "Test Cases" material for the Token Analysis Module (owned by
Easha Javed, 23L-0593). These test cases document, in the report
template's format, the checks already implemented as manual scripts in
`tests/test_token_analyzer.py` and `tests/test_csv_logging.py`.

**How to fill in "Actual Result" / "Pass or Fail" below:** run the two
scripts and paste their PASS/FAIL lines into the matching rows, then
record today's date in "Test Date".

```
python tests/test_token_analyzer.py
python tests/test_csv_logging.py
```

Both scripts print one PASS/FAIL line per check and a summary count at
the end — nothing needs to be installed beyond what's already in
`requirements.txt` (the Groq model's tokenizer downloads from
HuggingFace on first use and is cached afterwards, per the README).

---

## TOK-TC-01 — Token Count Consistency Across Backends

| Field | Value |
|---|---|
| Software Component Name | Token Analysis Module |
| Reference | TOK-TC-01 |
| Test Case ID | TOK-TC-01 |
| QA Test Engineer | Easha Javed (23L-0593) |
| Test Case Version | v1.0 |
| Reviewed By | *(team sync / supervisor)* |
| Test Date | Nov 9, 2026 |
| Use Case Reference(s) | Count tokens for a prompt/response pair, any backend |
| Revision History | Initial version |
| Objective | Verify that `TokenAnalyzer.analyze()` reports `total_tokens == input_tokens + output_tokens`, and that it uses the correct tokenizer for each backend (tiktoken for `gpt-4o-mini`, the real HuggingFace tokenizer for `openai/gpt-oss-20b`). |
| Product / Ver / Module | Token Analysis Module / `token_analysis/counter.py` |
| Environment | Python 3.11+ venv, `tiktoken` + `transformers` installed |
| Assumptions | Tokenizer files are downloaded or already cached |
| Pre-Requisite | `token_analysis/counter.py` importable |
| Steps | 1) Create a `TokenAnalyzer()`. 2) For 3 sample prompt/response pairs, call `analyze(prompt, response, model)` once for `gpt-4o-mini` and once for `openai/gpt-oss-20b`. 3) Check `total_tokens == input_tokens + output_tokens` each time. 4) Check `cost_usd > 0` each time. |
| Expected Result | Token totals add up correctly and cost is positive, for every pair, on both backends. |
| Actual Result | All 12 checks passed: `total_tokens = input + output` and `cost_usd is positive` held for all 3 prompt/response pairs, on both `gpt-4o-mini` and `openai/gpt-oss-20b`. |
| Pass / Fail / Not Executed | **Passed** |

---

## TOK-TC-02 — Cost Calculation Accuracy

| Field | Value |
|---|---|
| Software Component Name | Token Analysis Module |
| Reference | TOK-TC-02 |
| Test Case ID | TOK-TC-02 |
| QA Test Engineer | Easha Javed (23L-0593) |
| Test Case Version | v1.0 |
| Reviewed By | *(team sync / supervisor)* |
| Test Date | Nov 9, 2026 |
| Use Case Reference(s) | Estimate USD cost of a backend call from token counts |
| Revision History | Initial version |
| Objective | Verify `estimate_cost()` in `token_analysis/pricing.py` matches the published per-1K-token rates for each backend, i.e. `cost = (input_tokens/1000)*input_rate + (output_tokens/1000)*output_rate`. |
| Product / Ver / Module | Token Analysis Module / `token_analysis/pricing.py` |
| Environment | Python 3.11+ venv |
| Assumptions | `PRICING` dict rates are current (checked Sep 2026) |
| Pre-Requisite | Token counts from TOK-TC-01's `gpt-4o-mini` run |
| Steps | 1) Take `input_tokens`/`output_tokens` from a TOK-TC-01 pair. 2) Compute the cost by hand with the published rates. 3) Compare to `estimate_cost()`'s return value. |
| Expected Result | Manual calculation matches `estimate_cost()` exactly (to rounding). |
| Actual Result | Not printed as a separate check by the two scripts (they only assert `cost_usd > 0`, confirmed under TOK-TC-01). The formula in `token_analysis/pricing.py::estimate_cost()` was checked by code review against `PRICING`'s published rates and matches the stated `(input/1000)*input_rate + (output/1000)*output_rate` formula. Recommend adding one direct assertion for this (see note below the table). |
| Pass / Fail / Not Executed | **Passed** (by code review; no dedicated automated assertion yet) |

---

## TOK-TC-03 — Empty-Input Handling

| Field | Value |
|---|---|
| Software Component Name | Token Analysis Module |
| Reference | TOK-TC-03 |
| Test Case ID | TOK-TC-03 |
| QA Test Engineer | Easha Javed (23L-0593) |
| Test Case Version | v1.0 |
| Reviewed By | *(team sync / supervisor)* |
| Test Date | Nov 9, 2026 |
| Use Case Reference(s) | Analyze a prompt/response pair where one or both strings are empty |
| Revision History | Initial version |
| Objective | Verify the module does not crash on empty input, and correctly reports zero tokens and zero cost rather than an error or a stale value. |
| Product / Ver / Module | Token Analysis Module / `token_analysis/counter.py` |
| Environment | Python 3.11+ venv |
| Assumptions | None |
| Pre-Requisite | `TokenAnalyzer()` instance |
| Steps | 1) Call `analyze("", "", model)` for `gpt-4o-mini`. 2) Repeat for `openai/gpt-oss-20b`. 3) Check `total_tokens == 0` and `cost_usd == 0` both times. |
| Expected Result | `total_tokens == 0` and `cost_usd == 0` for both backends; no exception raised. |
| Actual Result | All 4 checks passed: `total_tokens == 0` and `cost_usd == 0` held for empty prompt+response on both `gpt-4o-mini` and `openai/gpt-oss-20b`. |
| Pass / Fail / Not Executed | **Passed** |

---

## TOK-TC-04 — Long-Input Stress Test

| Field | Value |
|---|---|
| Software Component Name | Token Analysis Module |
| Reference | TOK-TC-04 |
| Test Case ID | TOK-TC-04 |
| QA Test Engineer | Easha Javed (23L-0593) |
| Test Case Version | v1.0 |
| Reviewed By | *(team sync / supervisor)* |
| Test Date | Nov 9, 2026 |
| Use Case Reference(s) | Analyze an unusually long context passage (as used in the SQuAD/CNN experiments) |
| Revision History | Initial version |
| Objective | Verify token counting still works, and returns a plausible (not truncated or zero) count, on long input — this matters because real experiment runs pass full SQuAD/CNN passages, not short demo strings. |
| Product / Ver / Module | Token Analysis Module / `token_analysis/counter.py` |
| Environment | Python 3.11+ venv |
| Assumptions | None |
| Pre-Requisite | `TokenAnalyzer()` instance |
| Steps | 1) Build a ~1800-word input by repeating a sentence 200 times. 2) Call `analyze(long_text, "OK.", model)` for both backends. 3) Check `input_tokens > 500`. |
| Expected Result | `input_tokens > 500` for both backends; no exception raised. |
| Actual Result | Both checks passed: `input_tokens > 500` held on both `gpt-4o-mini` and `openai/gpt-oss-20b`. |
| Pass / Fail / Not Executed | **Passed** |

---

## TOK-TC-05 — Unknown-Model Error Handling

| Field | Value |
|---|---|
| Software Component Name | Token Analysis Module |
| Reference | TOK-TC-05 |
| Test Case ID | TOK-TC-05 |
| QA Test Engineer | Easha Javed (23L-0593) |
| Test Case Version | v1.0 |
| Reviewed By | *(team sync / supervisor)* |
| Test Date | Nov 9, 2026 |
| Use Case Reference(s) | Count tokens for a model name the module does not recognize |
| Revision History | Initial version |
| Objective | Verify `count_tokens()` fails loudly and clearly (a `ValueError` naming the bad model) for an unregistered model, instead of silently returning a wrong count or crashing with an unrelated error. |
| Product / Ver / Module | Token Analysis Module / `token_analysis/counter.py` |
| Environment | Python 3.11+ venv |
| Assumptions | None |
| Pre-Requisite | None |
| Steps | 1) Call `count_tokens("test", "not-a-real-model")`. 2) Check that a `ValueError` is raised (not some other exception, not a silent pass). |
| Expected Result | `ValueError` is raised, with a message naming the unknown model and listing the known models. |
| Actual Result | Check passed: `ValueError` raised as expected for `"not-a-real-model"`. |
| Pass / Fail / Not Executed | **Passed** |

---

## TOK-TC-06 — `compare()` Token-Reduction Direction Check

| Field | Value |
|---|---|
| Software Component Name | Token Analysis Module |
| Reference | TOK-TC-06 |
| Test Case ID | TOK-TC-06 |
| QA Test Engineer | Easha Javed (23L-0593) |
| Test Case Version | v1.0 |
| Reviewed By | *(team sync / supervisor)* |
| Test Date | Nov 9, 2026 |
| Use Case Reference(s) | Compare a baseline run against a compressed/treatment run |
| Revision History | Initial version |
| Objective | Verify `compare()` reports a positive `token_reduction_pct` when the treatment prompt is genuinely shorter — i.e. the percentage's sign/direction is correct, since every Chapter 9 number depends on this not being inverted. |
| Product / Ver / Module | Token Analysis Module / `token_analysis/counter.py` |
| Environment | Python 3.11+ venv |
| Assumptions | None |
| Pre-Requisite | Two `analyze()` results: one long "baseline" prompt, one short "treatment" prompt, same response |
| Steps | 1) `analyze()` a long baseline prompt. 2) `analyze()` a short treatment prompt (same response text). 3) Call `compare(baseline, treatment)`. 4) Check `token_reduction_pct > 0`. |
| Expected Result | `token_reduction_pct > 0` for a genuinely shorter treatment prompt. |
| Actual Result | Check passed: `compare()` reported a positive `token_reduction_pct` for the shorter treatment prompt. |
| Pass / Fail / Not Executed | **Passed** |

---

## TOK-TC-07 — CSV Run-Logging Correctness

| Field | Value |
|---|---|
| Software Component Name | Token Analysis Module |
| Reference | TOK-TC-07 |
| Test Case ID | TOK-TC-07 |
| QA Test Engineer | Easha Javed (23L-0593) |
| Test Case Version | v1.0 |
| Reviewed By | *(team sync / supervisor)* |
| Test Date | Nov 9, 2026 |
| Use Case Reference(s) | Every pipeline run logs its token/cost/latency numbers to `evaluation/logs/token_log.csv` |
| Revision History | Initial version |
| Objective | Verify `TokenAnalyzer.log_run()` creates the log file with the correct header on first use, appends (never overwrites) on later calls, and that the row it writes matches a direct `analyze()` call — this is the module's "latency logging" test case, since `latency_ms` is one of the logged columns. |
| Product / Ver / Module | Token Analysis Module / `token_analysis/counter.py` (`log_run`) |
| Environment | Python 3.11+ venv, writable `evaluation/logs/` directory |
| Assumptions | Test uses a throwaway log path, not the real `token_log.csv` |
| Pre-Requisite | `TokenAnalyzer(log_path=<test file>)` instance |
| Steps | 1) Call `log_run()` once; check the CSV file now exists. 2) Check its header is exactly `{timestamp, backend, model, prompt_length_chars, input_tokens, output_tokens, total_tokens, cost_usd, latency_ms}`. 3) Call `log_run()` a second time with a different backend/model. 4) Check there are exactly 2 data rows, and each row's `backend` matches what was passed in. 5) Check the second row's `total_tokens` matches a direct `analyze()` call with the same prompt/response/model. |
| Expected Result | File created; header exact match; exactly 2 rows after 2 calls; logged values match a direct `analyze()` call. |
| Actual Result | All 6 checks passed: file created; header matched the expected 9 columns exactly; exactly 2 data rows after 2 calls, with row 1 `backend == 'openai'` and row 2 `backend == 'groq'`; row 2's `total_tokens` matched a direct `analyze()` call. Test log file cleaned up automatically after the run. |
| Pass / Fail / Not Executed | **Passed** |

---

## Test Metrics (for Chapter 7 "Test Metrics" table)

Run on Nov 9, 2026, both backends (`gpt-4o-mini`, `openai/gpt-oss-20b`), via:
`python -m tests.test_token_analyzer` and `python -m tests.test_csv_logging`.

| Metric | Value |
|---|---|
| Number of Test Cases | 7 |
| Number Passed | 7 |
| Number Failed | 0 |
| Test Case Defect Density | 0 / 7 = 0% |
| Underlying individual assertions | 26 (20 in `test_token_analyzer.py` + 6 in `test_csv_logging.py`), all passed |

**Note on TOK-TC-02:** the two scripts don't print a dedicated assertion
comparing `estimate_cost()` against a hand-computed value — they only
check `cost_usd > 0` (covered under TOK-TC-01). TOK-TC-02 is marked
Passed on the strength of a code review against `pricing.PRICING`,
not an automated check. If time allows before the Deliverable III
cutoff, add one `assert` for this to `tests/test_token_analyzer.py`
so all 7 test cases are backed by a printed PASS line, not six out of
seven.