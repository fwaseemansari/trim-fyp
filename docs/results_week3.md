# Week 3 Results: Extractive Compression on SQuAD v2

Setup: 48 answerable SQuAD v2 samples, backend Groq (openai/gpt-oss-20b), extractive compression (TF-IDF sentence ranking) at three levels, compared against an uncompressed baseline. Each level keeps the stated fraction of sentences. Token reduction and compression ratio are measured on the context passage only, not on the full prompt or the model's answer.

## Summary table

| Setting | Token reduction | Compression ratio | Info retention | Exact match | F1 | Answer in response |
|---|---|---|---|---|---|---|
| Baseline (no compression) | 0.0% | 1.00 | 1.000 | 0.438 | 0.657 | 0.583 |
| Keep 70% | 23.0% | 1.32 | 0.956 | 0.417 | 0.658 | 0.562 |
| Keep 50% | 40.0% | 1.77 | 0.907 | 0.375 | 0.607 | 0.479 |
| Keep 30% | 61.7% | 2.96 | 0.818 | 0.438 | 0.603 | 0.542 |

Paired t-test on F1 against the baseline: p = 0.989 (70%), 0.368 (50%), 0.399 (30%).

## Chart 1: Compression ratio vs task accuracy

As compression gets more aggressive, F1 falls only slightly: from 0.657 at the baseline to about 0.60 at both the 50% and 30% settings, while the context shrinks by up to 62%. Keeping 70% of the sentences cost no measurable accuracy. The 30% setting scoring about the same as the 50% setting is not a real improvement; with only 48 samples the differences between those two are within noise.

## Chart 2: Token reduction by method

Only the extractive method has been evaluated so far, so this chart has a single bar per compression level. Reductions of 23%, 40% and 62% match the targets closely. SQuAD passages are short (often 5 to 8 sentences), so the sentence-level granularity limits how finely the level can be tuned.

## Chart 3: Cost savings

Cost reduction equals token reduction here because the pricing model is linear in input tokens. These figures describe the context portion only. Because the question, the instruction and the answer are not compressed, total per-request savings are smaller than the percentages above.

## Information retention

Embedding similarity falls steadily with compression (0.956, 0.907, 0.818), which behaves as expected and gives a useful sanity check on the compressor, but it is only loosely tied to answer accuracy: at 30% retention is lowest while accuracy is similar to 50%.

## Limitations

1. None of the accuracy differences are statistically significant at n = 48, so these results cannot claim that compression is free, only that no large drop was detected.
2. Gold answers in the sample file are sometimes whole sentences rather than short spans, which lowers exact match and F1 for every setting, including the baseline.
3. Only one backend and one compression method have been evaluated. The LLM-based compression method and the OpenAI backend are still to be run.
4. The prompt was changed mid-week to ask for short answers. All numbers above come from runs made after that change.