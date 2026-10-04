import json
import os
import shutil

import pandas as pd

csv_path = "evaluation/results/experiment_none.csv"
out_path = "evaluation/baseline_results.json"
backup_path = "evaluation/baseline_results_OLD_superseded.json"

df = pd.read_csv(csv_path)

# purani file ka backup
if os.path.exists(out_path):
    shutil.copy(out_path, backup_path)

result = {
    "note": "Regenerated from experiment_none.csv (short-answer prompt). Supersedes the Week 1 baseline.",
    "source": csv_path,
    "backend": str(df["backend"].iloc[0]),
    "n_samples": int(len(df)),
    "avg_total_tokens": round(float(df["total_tokens"].mean()), 1),
    "avg_cost_usd": round(float(df["cost_usd"].mean()), 6),
    "avg_latency_ms": round(float(df["latency_ms"].mean()), 1),
    "avg_exact_match": round(float(df["em"].mean()), 3),
    "avg_f1": round(float(df["f1"].mean()), 3),
    "avg_answer_in_response": round(float(df["answer_in_response"].mean()), 3),
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))