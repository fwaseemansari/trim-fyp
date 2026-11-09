"""Formal Chapter 7 test cases for Prompt Compression (CMP-TC-06..09) and the
Adaptive Context Manager (CTX-TC-03..08). Offline: TF-IDF scoring and a stub
LLM, so no API keys, credits or model downloads are needed.

Run: python -m pytest tests/test_formal_modules.py -v
"""
import pytest

from context_manager.manager import ContextManager
from prompt_compression.compressor import build_compressor
from token_analysis.counter import count_tokens

TEXT = (
    "Alpha Corp is a software company. It was founded by Marta Quill in 1987. "
    "The company is headquartered in Lisbon. Its first product was a spreadsheet tool. "
    "The weather in Lisbon is mild. Cats are popular pets in many cities. "
    "The river flows to the Atlantic ocean. Bananas are rich in potassium."
)
QUERY = "Who founded Alpha Corp?"


def _comp():
    return build_compressor("selective", scorer="tfidf")


# ---------------- Prompt Compression ----------------
def test_CMP_TC_06_token_budget_respected():
    for level in (0.3, 0.5, 0.7):
        r = _comp().compress(TEXT, QUERY, level)
        budget = r.metadata["budget_tokens"]
        one_window = len(r.metadata["selected_segments"]) == 1
        assert r.compressed_tokens <= budget or one_window
        assert r.compressed_tokens <= r.original_tokens


def test_CMP_TC_07_invalid_arguments_rejected():
    for bad in (0, -0.2, 1.5):
        with pytest.raises(ValueError):
            _comp().compress(TEXT, QUERY, bad)
    with pytest.raises(ValueError):
        build_compressor("not-a-method")
    with pytest.raises(ValueError):
        build_compressor("selective", scorer="not-a-scorer")
    with pytest.raises(ValueError):
        build_compressor("rewrite")  # needs llm_client


def test_CMP_TC_08_degenerate_input_returned_unchanged():
    r = _comp().compress("", QUERY, 0.5)
    assert r.text == "" and r.original_tokens == r.compressed_tokens == 0
    short = "One sentence here. Another one follows."
    r = _comp().compress(short, QUERY, 0.3)
    assert r.text == short and r.token_reduction_pct == 0.0


def test_CMP_TC_09_relevant_sentence_is_kept():
    for level in (0.3, 0.5):
        r = _comp().compress(TEXT, QUERY, level)
        assert "Marta Quill" in r.text
        assert r.token_reduction_pct > 0


# ---------------- Adaptive Context Manager ----------------
class _Resp:
    def __init__(self, response):
        self.response = response


class _StubLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, backend="groq"):
        self.calls += 1
        return _Resp("Short summary.")


def _turns(n):
    return [{"speaker": "A", "text": f"Today I talked about topic number {i} with my friend."}
            for i in range(n)]


def test_CTX_TC_03_relevance_aware_stays_within_token_budget():
    budget = 40
    cm = ContextManager(strategy="relevance_aware", max_tokens=budget)
    for t in _turns(12):
        cm.add_turn(t, query="What was topic number 3?")
        assert sum(count_tokens(x["text"]) for x in cm.history) <= budget


def test_CTX_TC_04_relevance_aware_keeps_relevant_old_turn():
    old = {"speaker": "A", "text": "I am running the Boston marathon in April next year."}
    filler = [{"speaker": "B", "text": f"Unrelated chatter number {i} about lunch and weather."}
              for i in range(6)]
    budget = 3 * max(count_tokens(t["text"]) for t in [old] + filler)
    rel = ContextManager(strategy="relevance_aware", max_tokens=budget)
    win = ContextManager(strategy="sliding_window", max_turns=3)
    for t in [old] + filler:
        rel.add_turn(t, query="When is the Boston marathon?")
        win.add_turn(t, query="When is the Boston marathon?")
    assert old in rel.history
    assert old not in win.history


def test_CTX_TC_05_kept_turns_stay_in_chronological_order():
    cm = ContextManager(strategy="relevance_aware", max_tokens=60)
    turns = _turns(10)
    for t in turns:
        cm.add_turn(t, query="topic number 7")
    positions = [turns.index(t) for t in cm.history]
    assert positions == sorted(positions)


def test_CTX_TC_06_memory_summarization_triggers_on_dropped_turns():
    llm = _StubLLM()
    cm = ContextManager(strategy="memory_summarization", max_tokens=40, llm_client=llm)
    for t in _turns(8):
        cm.add_turn(t, query="topic number 2")
    assert llm.calls > 0 and len(cm.memory) == llm.calls
    assert "(Earlier context, summarized)" in cm.get_context()


def test_CTX_TC_07_hybrid_scoring_ranks_recent_over_old_relevant():
    t0 = {"speaker": "A", "text": "My dentist appointment is on Tuesday at nine in the morning."}
    t1 = {"speaker": "B", "text": "We also talked about the new garden furniture for the summer."}
    t2 = {"speaker": "A", "text": "Then we chatted about movies and popcorn for the weekend."}
    budget = max(count_tokens(t["text"]) for t in (t0, t1, t2))
    q = "When is my dentist appointment?"
    pure = ContextManager(strategy="relevance_aware", max_tokens=budget)
    hyb = ContextManager(strategy="hybrid_scoring", max_tokens=budget, recency_weight=0.9)
    for cm in (pure, hyb):
        cm.add_turn(t0)
        cm.add_turn(t1)      # no query yet: nothing is pruned
        cm.add_turn(t2, query=q)
    assert t0 in pure.history and t2 not in pure.history
    assert t2 in hyb.history and t0 not in hyb.history


def test_CTX_TC_08_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        ContextManager(strategy="not-a-strategy")
    with pytest.raises(ValueError):
        ContextManager(strategy="hybrid_scoring", recency_weight=1.5)
    with pytest.raises(ValueError):
        ContextManager(strategy="memory_summarization")  # needs llm_client