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

None of the quality differences are statistically significant.