# Headline Numbers (frozen Oct 2, 2026)

Source files: evaluation/results/summary_all_runs.csv and evaluation/results/cnn_summary.csv. Quote only these numbers in the report and slides.

| Dataset | Setting | Token reduction | Quality |
|---|---|---|---|
| SQuAD v2, 48 samples | Baseline | 0.0% | F1 0.657, EM 0.438 |
| SQuAD v2 | Keep 70% | 23.0% | F1 0.658, EM 0.417 |
| SQuAD v2 | Keep 50% | 40.0% | F1 0.607, EM 0.375 |
| SQuAD v2 | Keep 30% | 61.7% | F1 0.603, EM 0.438 |
| CNN/DailyMail, 50 articles | Baseline | 0.0% | ROUGE-L 0.241 |
| CNN/DailyMail | Keep 50% | 42.3% | ROUGE-L 0.232, p = 0.367 |

None of the quality differences are statistically significant.# Headline Numbers (updated Oct 4, 2026)


Source files: SQuAD rows from `evaluation/results/experiment_*_300_rescored.csv` and
`experiment_selective_*_100_rescored.csv` (the `_100` files hold 300 rows; produced by
`evaluation/rescore_results.py`); CNN rows from `evaluation/results/cnn_summary.csv`.
Quote only these numbers in the report and slides.

SQuAD v2: 300 answerable questions, Groq gpt-oss-20b, every setting paired on the same 300
questions and the same prompt. Level is a token budget (fraction of the original context
tokens); token reduction is the achieved reduction on the context passage only. EM and F1 use
Unicode-aware answer normalization.

| Dataset | Setting | Token reduction | Quality | F1 change vs baseline (95% CI), p |
|---|---|---|---|---|
| SQuAD v2, 300 samples | Baseline | 0.0% | F1 0.768, EM 0.557 | n/a |
| SQuAD v2 | Cross-encoder, level 0.7 | 35.9% | F1 0.779, EM 0.570 | +0.011 (-0.015 to +0.037), p = 0.40 |
| SQuAD v2 | Cross-encoder, level 0.5 | 47.5% | F1 0.773, EM 0.573 | +0.005 (-0.024 to +0.034), p = 0.74 |
| SQuAD v2 | Cross-encoder, level 0.3 | 54.0% | F1 0.759, EM 0.567 | -0.009 (-0.042 to +0.022), p = 0.59 |
| SQuAD v2 | Bi-encoder, level 0.5 | 47.3% | F1 0.725, EM 0.547 | -0.044 (-0.077 to -0.011), p = 0.012 |
| SQuAD v2 | TF-IDF, level 0.5 | 47.8% | F1 0.741, EM 0.560 | -0.027 (-0.063 to +0.007), p = 0.11 |
| CNN/DailyMail, 50 articles | Baseline | 0.0% | ROUGE-L 0.241 | n/a |
| CNN/DailyMail | Keep 50% | 42.3% | ROUGE-L 0.232 | p = 0.367 |

How to read the SQuAD rows:
- The cross-encoder shows no detectable F1 loss at about 54% context-token reduction. The
  lower end of its confidence intervals is about -0.04, so this is "no loss detected", not
  "lossless".
- The bi-encoder at level 0.5 and TF-IDF and bi-encoder at level 0.3 show F1 drops of about
  0.04 that are nominally significant (p between 0.01 and 0.02) but do not survive correction
  for the many comparisons made; treat them as suggestive.
- The cross-encoder's advantage over the bi-encoder at level 0.5 (+0.048, p = 0.001) is the
  one comparison between methods that clearly survives that correction.
- Token reduction is measured on the context passage only. The question and instruction are
  not compressed, so the saving per request is smaller.

None of the CNN/DailyMail quality differences are statistically significant.

Superseded: the 48-sample SQuAD rows (baseline F1 0.657, EM 0.438; keep 70/50/30%) used an
older sample, an earlier prompt and sentence-ratio extractive compression, and ASCII-only
answer normalization. They are not comparable with the rows above and should not be quoted.