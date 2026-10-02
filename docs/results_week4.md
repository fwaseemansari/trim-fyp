# Week 4 Results: Extractive Compression on CNN/DailyMail

Setup: 50 CNN/DailyMail articles, backend Groq (openai/gpt-oss-20b), extractive compression (TF-IDF sentence ranking) keeping 50% of sentences, compared against an uncompressed baseline. Quality is measured with ROUGE-L against the reference summaries. Token reduction is measured on the article text only, not on the instruction or the generated summary.

Scripts: `evaluation/run_cnn_experiment.py` (runner) and `evaluation/summarize_cnn.py` (summary table and paired t-test). Outputs: `evaluation/results/cnn_summary.csv` and `evaluation/results/cnn_summary.tex`.

## Summary table

| Setting | Token reduction | ROUGE-L |
|---|---|---|
| Baseline (no compression) | 0.0% | 0.241 |
| Keep 50% | 42.3% | 0.232 |

Paired t-test on ROUGE-L against the baseline: p = 0.367.

## Discussion

Compression cut the article text by 42.3%, while ROUGE-L moved from 0.241 to 0.232, a drop of about 0.009. This difference is not statistically significant at n = 50, so the result shows that no clear quality loss was detected. It does not show that compression is free.

The token reduction is close to the 40.0% measured on SQuAD at the same keep 50% setting (see `docs/results_week3.md`), which suggests the compressor behaves consistently across the two datasets.

Summarization is a different task from question answering, so the two datasets are not directly comparable on quality. SQuAD is scored with F1 and exact match on short answers, while CNN/DailyMail is scored with ROUGE-L on multi sentence summaries. The two sets of numbers should be reported side by side but not merged.

## Limitations

1. Only keep 50% was evaluated. The keep 70% run stopped at 31 of 50 articles when the free Groq daily token quota ran out, so `cnn_extractive_0.7.csv` is excluded from all results.
2. Only one backend and one compression method were evaluated, as with SQuAD. LLM based compression and the OpenAI backend are deferred until developer tier access is available.
3. The ROUGE-L difference is not statistically significant, so no claim of improvement or harm should be made.
4. Sample size is 50 articles, so small differences are within noise.

## Resume list

1. Re run keep 70% with all 50 articles once developer tier access is available.
2. Add keep 30% for a three level comparison matching SQuAD.
3. Update `cnn_summary.csv`, `cnn_summary.tex` and this file with the new rows.