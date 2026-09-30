"""
Real test of the memory_summarization strategy -- needs a working .env
with GROQ_API_KEY (this makes actual API calls, unlike the other two
strategies).

Run from the repo root: python context_manager/run_memory_test.py
"""

import json

from context_manager.manager import ContextManager
from pipeline.llm_clients import LLMClient

LOCOMO_PATH = "data/locomo10.json"

with open(LOCOMO_PATH) as f:
    data = json.load(f)

sample = data[0]
conv = sample["conversation"]
session_keys = [
    k for k in conv
    if k.startswith("session_") and not k.endswith("date_time") and isinstance(conv[k], list)
]

# Use just the first 2 sessions to keep this quick and cheap to run.
turns = []
for k in session_keys[:2]:
    turns.extend(conv[k])

# Pick a real question whose evidence is likely to get pushed out of the
# active window, so we can see whether the memory summary still captures it.
qa = sample["qa"][5]
query = qa["question"]
print(f"Query: {query}")
print(f"Ground-truth answer: {qa['answer']}")
print(f"Evidence turn(s): {qa['evidence']}")
print()

llm_client = LLMClient()

cm = ContextManager(strategy="memory_summarization", max_tokens=300, llm_client=llm_client)
for turn in turns:
    cm.add_turn(turn, query=query)

print("=" * 60)
print("Active context (kept turns):")
print("=" * 60)
print(cm.get_context())

print()
print("=" * 60)
print("Memory summaries (dropped turns, condensed):")
print("=" * 60)
for m in cm.memory:
    print(f"- {m}")

print()
print("=" * 60)
print("MANUAL CHECK: does the memory or active context above still")
print("contain the fact needed to answer the ground-truth question?")
print("Write your answer in docs/context_manager_notes.md.")
print("=" * 60)
