"""Unicode robustness of the EM / F1 normalization (see quality_metrics._normalize)."""

from evaluation.quality_metrics import contains_answer, exact_match, f1_score


def test_non_breaking_hyphen_matches_ascii_hyphen():
    assert exact_match("co\u2011NP", "co-NP") == 1.0
    assert exact_match("Multi\u2011stage centrifugal pumps", "multi-stage centrifugal pumps") == 1.0
    assert f1_score("mid\u2011Eocene", "the mid-Eocene") == 1.0


def test_curly_apostrophe_and_en_dash():
    assert f1_score("Seven Years\u2019 War", "Seven Years' War") == 1.0
    assert exact_match("2008\u20132009", "2008-2009") == 1.0


def test_superscript_and_special_spaces_fold():
    assert exact_match("1,000 m\u00b3/s", "1,000 m3/s") == 1.0
    assert exact_match("5\u202f500\u202f000 km", "5 500 000 km") == 1.0


def test_ascii_symbol_stripping_is_kept():
    # '^' is ASCII punctuation: the original behaviour must not regress.
    assert exact_match("10^13", "1013") == 1.0


def test_genuinely_different_answers_still_differ():
    assert exact_match("Over 50%", "50%") == 0.0
    assert exact_match("20:1", "20 to 1") == 0.0      # number notation is NOT normalised
    assert f1_score("Paris", "London") == 0.0


def test_contains_answer_uses_the_same_normalization():
    assert contains_answer("It is co\u2011NP, I think.", "co-NP") == 1.0
    assert contains_answer("something else", "co-NP") == 0.0