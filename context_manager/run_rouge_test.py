"""
Response quality metric (Week 3 task, metric #6/7): ROUGE-L between LLM
answers generated WITH vs WITHOUT the Context Manager, against LoCoMo's
gold QA answers.

NOT RUN in the sandbox this was written in -- needs live LLM calls
(blocked there). Run this yourself once your .env is confirmed working.
Needs: pip install rouge-score
"""

import json
from rouge_score import rouge_scorer
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
all_turns = []
for k in session_keys:
    all_turns.extend(conv[k])

llm_client = LLMClient()
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

qa_sample = [qa for qa in sample["qa"] if qa.get("evidence")][:10]  # keep small, costs real API calls

with_cm_scores, without_cm_scores = [], []

for qa in qa_sample:
    query = qa["question"]
    gold_answer = qa["answer"]

    # WITHOUT context manager: last 5 raw turns only (naive baseline)
    naive_context = "\n".join(f"{t['speaker']}: {t['text']}" for t in all_turns[-5:])
    naive_prompt = f"Context:\n{naive_context}\n\nQuestion: {query}\nAnswer in one short sentence."
    naive_result = llm_client.generate(naive_prompt, backend="groq")

    # WITH context manager: relevance-aware retrieval
    cm = ContextManager(strategy="relevance_aware", max_tokens=400)
    for t in all_turns:
        cm.add_turn(t, query=query)
    cm_prompt = f"Context:\n{cm.get_context()}\n\nQuestion: {query}\nAnswer in one short sentence."
    cm_result = llm_client.generate(cm_prompt, backend="groq")

    without_cm_scores.append(scorer.score(str(gold_answer), naive_result.response)["rougeL"].fmeasure)
    with_cm_scores.append(scorer.score(str(gold_answer), cm_result.response)["rougeL"].fmeasure)

    print(f"Q: {query}")
    print(f"  Gold: {gold_answer}")
    print(f"  Without CM: {naive_result.response[:80]}")
    print(f"  With CM:    {cm_result.response[:80]}")
    print()

print(f"Average ROUGE-L WITHOUT Context Manager: {sum(without_cm_scores)/len(without_cm_scores):.3f}")
print(f"Average ROUGE-L WITH Context Manager:    {sum(with_cm_scores)/len(with_cm_scores):.3f}")
