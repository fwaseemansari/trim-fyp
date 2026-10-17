"""compare_runs and noise_floor on small synthetic result tables."""

import numpy as np
import pandas as pd
import pytest

from evaluation.compare_runs import compare
from evaluation.noise_floor import noise_floor, treatment_vs_baselines


def make(f1, **extra):
    n = len(f1)
    df = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "f1": f1, "em": [float(x == 1) for x in f1],
                       "token_reduction_pct": 40.0, **extra})
    return df.set_index("sample_id")


def test_compare_reports_change_calls_and_cost():
    base = make([1, 1, 0.5, 0.5, 0, 0, 1, 1, 0.5, 0.5])
    run = make([1, 0.5, 0.5, 0.5, 0, 0, 1, 0.5, 0.5, 0.5], compression_llm_calls=3,
               answer_call_cost_usd=0.01, compression_call_cost_usd=0.02, end_to_end_cost_usd=0.03)
    table = compare(base, {"rewrite": run}).set_index("run")
    assert table.loc["rewrite", "n"] == 10
    assert table.loc["rewrite", "d_f1"] == pytest.approx(-0.1, abs=1e-4)
    assert table.loc["rewrite", "ci_low"] <= table.loc["rewrite", "d_f1"] <= table.loc["rewrite", "ci_high"]
    assert table.loc["rewrite", "compression_calls"] == 30
    assert table.loc["rewrite", "end_to_end_cost_usd"] == pytest.approx(0.3)


def test_compare_uses_first_n_reference_questions():
    base = make([1.0] * 10)
    run = make([0.0] * 5 + [1.0] * 5)
    table = compare(base, {"r": run}, n=5).set_index("run")
    assert table.loc["r", "n"] == 5 and table.loc["r", "d_f1"] == pytest.approx(-1.0)


def test_noise_floor_of_identical_runs_is_zero():
    run = make(list(np.random.RandomState(0).rand(40)))
    nf = noise_floor(run, run.copy())
    assert nf["mean_diff"] == 0 and nf["share_changed"] == 0 and nf["min_detectable_f1_change"] == 0


def test_noise_floor_detects_changed_questions_and_sets_a_threshold():
    rng = np.random.RandomState(1)
    f1 = rng.choice([0.0, 0.5, 1.0], size=200)
    f2 = f1.copy()
    flip = rng.choice(200, size=40, replace=False)
    f2[flip] = 1.0 - f2[flip]
    nf = noise_floor(make(list(f1)), make(list(f2)))
    assert 0.1 <= nf["share_changed"] <= 0.2
    assert nf["min_detectable_f1_change"] == pytest.approx(2.8 * nf["sd_per_question"] / np.sqrt(200))


def test_treatment_is_flagged_against_the_noise():
    r1, r2 = make([0.5] * 20), make([0.5] * 20)
    small = treatment_vs_baselines(r1, r2, make([0.49] * 20), mde=0.05)
    large = treatment_vs_baselines(r1, r2, make([0.3] * 20), mde=0.05)
    assert not small["larger_than_noise"] and large["larger_than_noise"]


def test_compare_can_be_restricted_to_listed_ids():
    base = make([1.0] * 6)
    run = make([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    table = compare(base, {"r": run}, only_ids={"s0", "s1"}).set_index("run")
    assert table.loc["r", "n"] == 2 and table.loc["r", "d_f1"] == pytest.approx(-1.0)