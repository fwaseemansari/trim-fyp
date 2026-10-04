import pytest
from evaluation.quality_metrics import exact_match, f1_score

def test_exact_match_normalization():
    assert exact_match("The Eiffel Tower.", ["eiffel tower"]) == 1.0
    assert exact_match("Paris", ["London"]) == 0.0

def test_unanswerable_squad_v2():
    assert exact_match("", []) == 1.0
    assert exact_match("something", []) == 0.0
    assert f1_score("", []) == 1.0

def test_f1_partial_and_multi_gold():
    assert f1_score("big red dog", ["red dog"]) == pytest.approx(0.8)
    assert f1_score("red dog", ["cat", "red dog"]) == 1.0
    assert f1_score("zebra", ["red dog"]) == 0.0


def test_contains_answer():
    from evaluation.quality_metrics import contains_answer
    assert contains_answer("The final PM was **Lothar de Maizière**, from 1990.", ["Lothar de Maizière"]) == 1.0
    assert contains_answer("I do not know", ["Paris"]) == 0.0
    assert contains_answer("anything", []) is None