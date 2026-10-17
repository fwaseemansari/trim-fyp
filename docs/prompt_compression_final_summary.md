# Prompt Compression Module: Final Summary (FYP I)

Status: describes the code as built at the end of FYP I.

## 1. Purpose

Reduce the number of context tokens sent to an LLM while keeping the information
needed to answer the question. The module takes a text and a query and returns a
`CompressionResult` (text, token counts, kept sentences, scores, LLM-call cost).

## 2. Architecture (as built)

```
text + query
   |  Segmenter  (segmenters.py)   sentences, or overlapping windows of 2-3 sentences
   v
segments
   |  Scorer     (scorers.py)      TF-IDF | bi-encoder | cross-encoder
   v
one score per segment
   |  Greedy selection (compressor.py: SelectiveCompressor)
   v
kept sentences, in document order  ->  CompressionResult (results.py)
```

| Part | Where | Notes |
|---|---|---|
| Segmenters | `segmenters.py` | `SentenceSegmenter`; `OverlappingWindowSegmenter(window_size 2 or 3, stride 1)`. Sentence splitter is a parameter: `regex` (default) or `nltk`. |
| Scorers | `scorers.py` | `TfidfScorer`, `BiEncoderScorer` (all-MiniLM-L6-v2), `CrossEncoderScorer` (ms-marco-MiniLM-L-6-v2). Models load lazily and are cached per process. |
| Compressors | `compressor.py` | `SelectiveCompressor` (drop the rest) and `RewriteCompressor` (LLM rewrites dropped windows). Built by `build_compressor(method, scorer, window_size, ...)`. |
| Result type | `results.py` | `CompressionResult`, including `compression_input_tokens` / `compression_output_tokens` for LLM calls. |
| Legacy API | `compress()` | Week 2 string-in/string-out; also hosts whole-text LLM condensing (`method="llm"`). |

Design choice: segmenter, scorer and selection are separate parts, so a new scorer
(for example the cross-encoder) changes one line of configuration, not the
compressor. This is what made the scorer comparison in section 5 cheap to run.

Selection is done by the greedy loop inside `SelectiveCompressor.compress`, which counts
the union of kept sentences. (An earlier `BudgetSelector` class summed per-segment costs
and would double-count sentences shared by overlapping windows; it was never used and has
been removed.)

## 3. Methods

| Method | Segments | Scoring | What happens to the rest |
|---|---|---|---|
| `extractive` | single sentences | TF-IDF | dropped (Week 2 baseline) |
| `selective` / `drop` | overlapping windows | tfidf / bi_encoder / cross_encoder | dropped |
| `rewrite` | as selective | as selective | rewritten by an LLM into short statements that must fit the leftover budget |

## 4. What `level` means

`level` is the fraction of the original token count the output may use. It is a
target. The best-scoring window is always kept even when it exceeds the budget,
and windows are whole units, so budget is often left unused. Achieved context-token
reduction on the 300 SQuAD questions:

| Nominal level | Kept tokens (approx.) | Achieved reduction |
|---|---|---|
| 0.3 | 46% | about 54% |
| 0.5 | 52% | about 47-48% |
| 0.7 | 64% | about 36% |

About 6% of contexts had zero reduction (two sentences or fewer: nothing to choose
between). Report achieved reduction, not the nominal level.

## 5. Results (SQuAD v2, 300 answerable questions, Groq gpt-oss-20b)

Paired comparison against a no-compression baseline run on the same 300 questions
with the same prompt: baseline F1 0.768, EM 0.557. Token reduction is context only;
question and instruction are not compressed. All scores use the Unicode-aware answer
normalization described at the end of this section.

| Scorer | Level | Token reduction | F1 | F1 change (95% CI) | p (paired t) |
|---|---|---|---|---|---|
| cross-encoder | 0.3 | 54.0% | 0.759 | -0.009 (-0.042 to +0.022) | 0.59 |
| cross-encoder | 0.5 | 47.5% | 0.773 | +0.005 (-0.024 to +0.034) | 0.74 |
| cross-encoder | 0.7 | 35.9% | 0.779 | +0.011 (-0.015 to +0.037) | 0.40 |
| tf-idf | 0.3 | 54.6% | 0.723 | -0.045 (-0.082 to -0.009) | 0.011 |
| tf-idf | 0.5 | 47.8% | 0.741 | -0.027 (-0.063 to +0.007) | 0.11 |
| tf-idf | 0.7 | 35.9% | 0.742 | -0.027 (-0.058 to +0.002) | 0.08 |
| bi-encoder | 0.3 | 53.9% | 0.727 | -0.041 (-0.078 to -0.006) | 0.023 |
| bi-encoder | 0.5 | 47.3% | 0.725 | -0.044 (-0.077 to -0.011) | 0.012 |
| bi-encoder | 0.7 | 35.8% | 0.744 | -0.024 (-0.052 to +0.005) | 0.11 |

What the data supports:
- The cross-encoder shows no detectable F1 loss at any level (changes below about 0.033
  cannot be resolved with 300 questions; see the noise floor below). The lower end of its
  intervals is about -0.04, so "no loss detected", not "lossless".
- It scores above the bi-encoder at every level (+0.032 at 0.3, +0.048 at 0.5, +0.035 at
  0.7; p = 0.041, 0.001, 0.007) and above tf-idf at every level (+0.036, +0.032, +0.038;
  p = 0.028, 0.041, 0.012). Across the 15 pairwise comparisons made in this report only
  cross-encoder vs bi-encoder at level 0.5 clearly survives a multiple-comparison
  correction (threshold about 0.003); treat the others as suggestive.
- The bi-encoder is not distinguishable from tf-idf (p = 0.33 to 0.85).
- The drops of about 0.04 for tf-idf at 0.3 and bi-encoder at 0.3 and 0.5 have p about
  0.01-0.02 against the first baseline run. For bi-encoder at 0.5 and tf-idf at 0.3 they were
  replicated against a second baseline run (-0.056, p = 0.0007 and -0.057, p = 0.0012) and
  survive a correction for nine comparisons (about 0.0056). The bi-encoder drop at 0.3 was
  not retested and stays suggestive.
- F1 hardly changes with level in this range (cross-encoder 0.759 to 0.779). Do not
  claim a trade-off curve.
- The cross-encoder's advantage comes from keeping the answer in the context:
  96.7% of answers survive at level 0.5, against 91.7% for the other two. When
  the answer survives, F1 is about 0.78-0.80 for all three scorers; when it is lost
  it is about 0.10-0.14.

Caveats: single backend and dataset, and each compressed run was made once.

Noise floor. A second baseline run on the same 300 questions scored F1 0.781 against 0.768
(difference +0.013, 95% CI -0.011 to +0.036, p = 0.29). 24% of questions changed F1 between
the two runs, and the SD of the per-question difference is 0.203. A paired comparison on 300
questions can therefore detect a mean F1 change of about 0.033 (80% power). The
cross-encoder's changes (-0.009 to +0.011 against run 1) are within this; the bi-encoder at
level 0.5 (-0.044 against run 1, -0.056 against run 2) and tf-idf at level 0.3 (-0.045,
-0.057) exceed it against both baseline runs. Against run 2 these two drops are -0.056
(95% CI -0.088 to -0.025, p = 0.0007) and -0.057 (-0.092 to -0.023, p = 0.0012); the
cross-encoder at level 0.5 is -0.008 (-0.035 to +0.020, p = 0.58). Run 1 remains the
reference in all tables.

Cost. Estimated answer-call cost for the 300 questions (visible prompt and response tokens
only; hidden reasoning tokens are not counted): baseline $0.0050, cross-encoder 0.5 $0.0032
(36.8% lower), bi-encoder 0.5 $0.0032 (36.8% lower), tf-idf 0.3 $0.0028 (44.1% lower). The
saving per request is smaller than the context-token reduction because the question,
instruction and answer are not compressed. Local selective methods make no compression
calls.

Scoring note. The official SQuAD script strips only ASCII punctuation. The model often
writes typographic characters instead (non-breaking hyphen U+2011 appeared 139 times in
3,000 responses, curly apostrophes 24 times), so "co‑NP" scored 0 against "co-NP".
TRIM's EM/F1 now apply NFKC normalization and remove Unicode punctuation as well
(`evaluation/quality_metrics.py`), and every run was re-scored offline from its saved
responses (`evaluation/rescore_results.py`). This raised each run's F1 by 0.006 to 0.016
and left every comparison's conclusion unchanged. Number-notation differences ("20:1" vs
"20 to 1") and paraphrases ("Over 50%" vs "More than 50%") are still scored as misses.

## 6. Failure analysis

Week 2 (20 SQuAD samples, TF-IDF) found three failure modes: spelling variation,
synonym gap with common-term dilution, and cross-sentence anaphora (details in
`compression_notes.md`). Embedding and cross-encoder scoring address the first two;
the third is only partly addressed by overlapping windows. Those notes' sample
(n=20) is small; they are examples, not an exhaustive taxonomy.

Hallucination audit (`evaluation/hallucination_audit.py`): 30 answers from each of
the baseline, bi-encoder 0.5 and cross-encoder 0.5 runs (90 total), labelled blind
to the run. Answer-lost cases were over-sampled, so rates are weighted back to the
full run.

| Run | Hallucinated (weighted, 95% CI) | Unsupported incl. correct-from-memory |
|---|---|---|
| No compression | 0.0% (0 to 4.4%) | 0.0% (0 to 4.4%) |
| Bi-encoder 0.5 | 2.0% (0 to 11.0%) | 2.0% (0 to 11.0%) |
| Cross-encoder 0.5 | 0.0% (0 to 6.3%) | 0.3% (0 to 6.7%) |

When the answer was removed the model mostly abstained ("I don't know") rather than
fabricating one. The intervals overlap, so this shows hallucination is not a major
failure mode here but cannot rank the scorers. Labelling was LLM-assisted: first-pass
labels were produced by an LLM (Claude), blind to the run, and then reviewed by the
author, who changed one label (row 14, unsure to hallucinated).

## 7. Cost accounting

Local selective methods make no LLM calls. `rewrite` makes one call per dropped
window; every call is counted in `compression_input_tokens` /
`compression_output_tokens`, and the pipeline adds it to `end_to_end_cost_usd`. So an
LLM-based method cannot look cheaper than it is.

Rewrite pilot (exploratory). `rewrite` was run once on the first 50 questions with the
cross-encoder at level 0.5. Against plain selection on the same questions F1 changed by
+0.0045 (95% CI -0.050 to +0.055, p = 0.87), and context-token reduction was 44.9% against
45.1%, so the rewrites that fitted the leftover budget added back almost nothing. It made
132 compression calls for 50 questions. Estimated end-to-end cost (visible tokens only) was
$0.00315, of which $0.00259 was the compression calls: 3.6 times the uncompressed baseline
($0.00087) and 5.6 times plain selection ($0.00057). With 50 questions only F1 changes above
about 0.08 are detectable, so this shows no benefit detected, not no benefit; in this setup
rewriting raised cost without a measurable accuracy gain.

Whole-text LLM condensing exists only in the legacy `compress()` and was not evaluated.
Every other result in this summary is for the local selective methods (extractive, and
selective with TF-IDF, bi-encoder or cross-encoder scoring).

## 8. Known limitations

- The default regex sentence splitter mis-splits initials and abbreviations ("Robert J.
  Shiller", "Dr.", "St.") and ignores other languages. It disagrees with nltk's Punkt on 64 of
  the 265 distinct SQuAD contexts (75 of the 300 questions). Rerunning the cross-encoder at
  level 0.5 with nltk changed the compressed context for 43 questions (14%), raised mean
  context-token reduction from 47.5% to 48.2% and kept the answer in the context at the same
  rate (96.7% against 97.0%). F1 changed by -0.016 (95% CI -0.039 to +0.005, p = 0.16),
  within the noise floor; on the 43 questions whose context differed it changed by +0.005
  (-0.040 to +0.051; only changes above about 0.09 are detectable on that subset). By
  arithmetic from these averages the 257 questions with identical inputs changed by about
  -0.02, which is run-to-run noise. The splitter choice therefore has no detectable effect on
  accuracy here, and regex remains the default (`compression_splitter`).
- `level` is a budget, not a guarantee (section 4).
- `RewriteCompressor`: overlapping windows mean a "dropped" window can repeat kept
  facts; rewrites are appended after the kept text, not interleaved in document order;
  cost grows with the number of dropped windows. In the pilot the rewritten text added only
  about 0.3 percentage points of context, so most calls added cost without adding text.
- `scores` is per segment while `kept_indices` is per sentence (different index spaces).
- Selection re-counts tokens of the whole candidate for every segment; fine for
  paragraphs, quadratic for long documents.
- Evaluation covers SQuAD only; multi-turn (LoCoMo) compression is not evaluated in
  this summary.