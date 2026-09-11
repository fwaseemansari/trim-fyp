# Context Manager Notes — Memory Summarization Failure Cases

**Status:** template — run `context_manager/manager.py` with
`strategy="memory_summarization"` on a full LoCoMo session (10+ turns),
then manually compare `self.memory` summaries against that session's
`qa` entries (in `data/locomo10.json`) whose `evidence` points to a
dropped turn. Fill in real findings below — these become the report's
Limitations discussion, so they need to be genuine observations, not
guessed.

## How to check
1. Pick one conversation from `data/locomo10.json`, take a session with
   10+ turns.
2. Run it through `ContextManager(strategy="memory_summarization", ...)`,
   turn by turn, using each `qa` entry's question as the `query` in
   `add_turn()` (or a fixed representative query if testing one pass).
3. For each `qa` entry whose `evidence` `dia_id` maps to a turn that got
   dropped (summarized), check: does the LLM-generated summary in
   `self.memory` still contain the fact needed to answer that question?

## Findings
*(fill in after running — e.g. which kinds of facts survive
summarization reliably, which get lost — dates and names are common
failure points for 1-sentence summaries)*

-
-
-
