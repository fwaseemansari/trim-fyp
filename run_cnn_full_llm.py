"""CNN/DailyMail LLM runs on the fixed 1,000-article subset (same file for the whole team).

    python -m data.prepare_datasets cnn-full     # once
    python run_cnn_full_llm.py

Ordered by priority, so a budget cut still leaves the key runs. Resumable.
Compare with: python -m evaluation.summarize_cnn <baseline csv> <other csvs...>
"""

from evaluation.run_cnn_experiment import run_cnn_experiment

R = "evaluation/results"
SAMPLE = "data/cnn_subset_1000.json"
BACKEND = "groq"
DELAY = 1.5


def go(name, **kw):
    run_cnn_experiment(sample_path=SAMPLE, n_samples=None, backend=BACKEND, delay_seconds=DELAY,
                       out_path=f"{R}/cnn_sub_{name}.csv", **kw)


go("none", compression_method="none")

for level in (0.5, 0.3, 0.7):
    go(f"selective_cross_encoder_{level}", compression_method="selective",
       compression_scorer="cross_encoder", compression_level=level)
    go(f"lead_{level}", compression_method="lead", compression_level=level)

# Comparison scorers at one level (drop these if the budget is tight).
for scorer in ("tfidf", "bi_encoder"):
    go(f"selective_{scorer}_0.5", compression_method="selective",
       compression_scorer=scorer, compression_level=0.5)
