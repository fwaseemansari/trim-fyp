"""
Fri Week 2 Faiqa task: run the compression sweep across 30%/50%/70%
keep-ratios, 20 samples each, extractive method (the LLM method also
works if you pass method="llm" and have a funded OpenAI key, or if you
point it at Groq via the has_openai fallback already built into
compress()/LLMClient).

This produces the core "trade-off between token reduction and
task performance" evidence the plan calls out as your proposal's
central empirical claim.
"""
import time

from evaluation.run_experiment import run_experiment

LEVELS = [0.3, 0.5, 0.7]


def run_sweep(n_samples: int = 20, method: str = "extractive") -> None:
    for i, level in enumerate(LEVELS):
        print(f"\n--- Running level={level} ({int(level * 100)}% kept) ---")
        run_experiment(
            n_samples=n_samples,
            compression_enabled=True,
            compression_method=method,
            compression_level=level,
            out_path=f"evaluation/results/sweep_{method}_{int(level * 100)}pct.csv",
        )
        
        if i < len(LEVELS) - 1:
            print("  (pausing 5s between levels to stay clear of the rate limit)")
            time.sleep(5)


if __name__ == "__main__":
    run_sweep()
