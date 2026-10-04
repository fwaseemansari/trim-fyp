# LoCoMo Dataset Notes

**Source:** `data/locomo10.json`, bundled directly in this repo (2.7 MB —
small enough to commit; no need to re-clone the GitHub repo).

**Structure:** a JSON list of 10 conversation objects. Each conversation has:
- `sample_id` — string id, e.g. `"conv-26"`
- `conversation` — a dict containing `speaker_a`, `speaker_b` (names), and
  a variable number of `session_N` keys (each a list of turns) plus a
  matching `session_N_date_time` string. Session count varies per
  conversation (not a fixed number).
- Each turn in a `session_N` list is `{"speaker": ..., "dia_id": "D1:1",
  "text": "..."}` — a single utterance, not a full exchange.
- `qa` — a list of question/answer pairs testing recall over the whole
  conversation: `{"question": ..., "answer": ..., "evidence": ["D1:3"],
  "category": 2}`. `evidence` points back to the `dia_id`(s) that contain
  the answer — useful for checking whether a context-management strategy
  kept the turn that actually matters.
- `event_summary`, `observation`, `session_summary` — additional
  per-session annotation dicts, not needed for Week 1-2 (context manager
  only needs `conversation` + `qa`).

**Scale (across all 10 conversations):**
- 5,882 total turns
- 1,986 total QA pairs (about 199 per conversation on average; 105 to 260
  per conversation, 199 in the first sample)

**Further notes from data inspection:**
- Each conversation is split into 19 to 32 dated sessions spread over
  different dates, simulating a relationship over time.
- A session holds roughly 10 to 47 turns (about 22 on average), around 490
  words of `text` per session on average.
- Every QA pair carries a `category` number (1-5) reflecting different
  reasoning types such as single-hop, multi-hop and temporal questions.
- The `evidence` field points to the exact turn(s) (e.g. `D1:3`) that
  contain the answer. This is what lets us check whether our Context
  Manager kept the right turns in its window when the correct answer
  required information from several sessions ago.

**Implication for the Context Manager:** a "session" here is a batch of
turns from one sitting (dated), and a conversation spans multiple
sessions over time — this is exactly the multi-session, long-horizon
memory problem the Adaptive Context Manager is meant to handle: by the
time you reach a later session, you can't fit all prior turns in context,
so which turns/summaries you keep determines whether later QA pairs are
still answerable.