from evaluation.run_experiment import run_experiment

R = "evaluation/results"
COMMON = dict(sample_path="data/squad_300.json", backend="groq", compression_window_size=2)

def go(name, n, **kw):
    run_experiment(n_samples=n, out_path=f"{R}/{name}.csv", **COMMON, **kw)

go("experiment_baseline_300", 300, compression_enabled=False)
for scorer in ("cross_encoder", "tfidf"):
    go(f"experiment_selective_{scorer}_0.5_300", 300, compression_enabled=True,
       compression_method="selective", compression_scorer=scorer, compression_level=0.5)
for level in (0.3, 0.7):
    for scorer in ("tfidf", "cross_encoder", "bi_encoder"):
        go(f"experiment_selective_{scorer}_{level}_100", 300, compression_enabled=True,
           compression_method="selective", compression_scorer=scorer, compression_level=level)