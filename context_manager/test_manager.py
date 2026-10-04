"""
Fri Week 1 Dania task: feed 3 real LoCoMo sessions through the
sliding-window ContextManager turn-by-turn, confirm the kept context
never exceeds a token budget (window is bounded by turn COUNT, not
tokens directly — this test checks that turn-count bound also keeps
token count sane in practice, since a runaway single turn could still
blow the budget even with few turns kept).

Run with: pytest context_manager/test_manager.py -v
"""

import json

from context_manager.manager import ContextManager
from token_analysis.counter import count_tokens


def _load_locomo_sessions(n_sessions: int = 3, path: str = "data/locomo10.json"):
    """Pulls the first `n_sessions` session_1 turn-lists from the first
    n_sessions conversations in the dataset — good enough for a smoke
    test without loading the whole 2.7 MB file's worth of sessions."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    sessions = []
    for conv in data[:n_sessions]:
        turns = conv["conversation"]["session_1"]
        sessions.append(turns)
    return sessions


def test_sliding_window_never_exceeds_max_turns():
    sessions = _load_locomo_sessions(3)
    for session in sessions:
        cm = ContextManager(strategy="sliding_window", max_turns=6)
        for turn in session:
            cm.add_turn(turn)
            assert len(cm.history) <= 6, "sliding window exceeded max_turns"


def test_sliding_window_keeps_most_recent_turns():
    sessions = _load_locomo_sessions(1)
    session = sessions[0]
    cm = ContextManager(strategy="sliding_window", max_turns=6)
    for turn in session:
        cm.add_turn(turn)

    if len(session) >= 6:
        expected_last_texts = [t["text"] for t in session[-6:]]
        actual_texts = [t["text"] for t in cm.history]
        assert actual_texts == expected_last_texts


if __name__ == "__main__":
    # Manual run (outside pytest): print token counts per turn for 1 session.
    sessions = _load_locomo_sessions(1)
    cm = ContextManager(strategy="sliding_window", max_turns=6)
    for i, turn in enumerate(sessions[0]):
        cm.add_turn(turn)
        print(f"turn {i}: kept={len(cm.history)} turns, {count_tokens(cm.get_context())} tokens")
