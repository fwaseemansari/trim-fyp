"""
Cross-backend check (Week 3 Wed task): run the same LoCoMo sessions
through the Context Manager against both Groq and OpenAI, and note any
backend-specific quirks (e.g. different tokenizer counts for the same
text).

NOT RUN in the sandbox this was written in -- needs live API access
(blocked there). Run this yourself once your .env is confirmed working.
"""

import json
from token_analysis.counter import count_tokens
from context_manager.manager import ContextManager
from pipeline.llm_clients import LLMClient

with open("data/locomo10.json") as f:
    data = json.load(f)

sample = data[0]
conv = sample["conversation"]
session_keys = [
    k for k in conv
    if k.startswith("session_") and not k.endswith("date_time") and isinstance(conv[k], list)
]
turns = []
for k in session_keys[:2]:
    turns.extend(conv[k])

query = sample["qa"][5]["question"]

print("=== Tokenizer comparison on the same context text ===")
cm = ContextManager(strategy="relevance_aware", max_tokens=400)
for t in turns:
    cm.add_turn(t, query=query)
context_text = cm.get_context()

groq_tokens = count_tokens(context_text, model="openai/gpt-oss-20b")
openai_tokens = count_tokens(context_text, model="gpt-4o-mini")  # tiktoken
print(f"Groq/Llama tokenizer count:   {groq_tokens}")
print(f"OpenAI/tiktoken count:        {openai_tokens}")
print(f"Difference:                   {abs(groq_tokens - openai_tokens)} tokens "
      f"({100*abs(groq_tokens - openai_tokens)/max(groq_tokens, openai_tokens):.1f}%)")

print()
print("=== Same query, both backends ===")
llm_client = LLMClient()
for backend in ["groq", "openai"]:
    result = llm_client.generate(f"Context:\n{context_text}\n\nQuestion: {query}", backend=backend)
    print(f"\n[{backend}] latency: {result.latency_ms:.0f}ms")
    print(f"[{backend}] response: {result.response[:200]}")

print()
print("MANUAL STEP: write any quirks observed above (tokenizer count gap,")
print("response style differences, latency differences) into docs/backend_notes.md")
