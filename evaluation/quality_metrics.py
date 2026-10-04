"""Week 3 quality metrics for TRIM: task accuracy (EM, F1) and ROUGE-L.

Week 2 efficiency metrics already live in evaluation/metrics.py, so they are
not duplicated here. Pure functions, no dependency on other TRIM modules.
"""
import re
import string
import unicodedata
from collections import Counter

_ASCII_PUNCTUATION = set(string.punctuation)


def _normalize(s):
    """SQuAD-style normalization, robust to Unicode typography.

    The official SQuAD script strips only ASCII punctuation. LLM output often
    uses typographic characters instead (non-breaking hyphen U+2011, curly
    quotes, en dashes, superscript digits), so "co-NP" scored 0 against
    "co\u2011NP". Steps: NFKC folds compatibility characters (superscript 3 ->
    3, no-break and narrow spaces -> space, U+2011 -> hyphen); then ASCII
    punctuation AND any Unicode punctuation (category P*) is removed; then
    articles are dropped and whitespace is collapsed.
    """
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(
        ch for ch in s
        if ch not in _ASCII_PUNCTUATION and not unicodedata.category(ch).startswith("P")
    )
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _as_list(golds):
    return [golds] if isinstance(golds, str) else list(golds)


def exact_match(prediction, golds):
    """1.0 if prediction equals any gold answer after normalization.
    SQuAD v2: if there is no gold answer (unanswerable), correct = empty prediction."""
    golds = _as_list(golds)
    if not golds:
        return 1.0 if _normalize(prediction) == "" else 0.0
    return float(any(_normalize(prediction) == _normalize(g) for g in golds))


def _f1_single(pred, gold):
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction, golds):
    """Best token-overlap F1 over all gold answers (SQuAD definition)."""
    golds = _as_list(golds)
    if not golds:
        return 1.0 if _normalize(prediction) == "" else 0.0
    return max(_f1_single(prediction, g) for g in golds)


def contains_answer(prediction, golds):
    """1.0 if any gold answer appears inside the prediction (after normalization).
    More forgiving than EM/F1 for chatty LLMs that wrap the answer in a long sentence."""
    golds = _as_list(golds)
    if not golds:
        return None
    pred = _normalize(prediction)
    return float(any(_normalize(g) and _normalize(g) in pred for g in golds))


def rouge_l(prediction, reference):
    """ROUGE-L F-measure via the rouge-score package."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure