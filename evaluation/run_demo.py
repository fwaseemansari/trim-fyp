"""
Sat Week 2 sync-call demo, per the plan: run one SQuAD sample through
compression, one LoCoMo session through the context manager, print the
logged metrics for both. Run this live on the call rather than relying
on a screen recording (that comes in Week 3's plan instead).
"""

import json

from pipeline.pipeline import run, run_conversation


def demo_compression():
    with open("data/squad_sample.json", encoding="utf-8") as f:
        sample = json.load(f)[0]

    print("=== Compression demo ===")
    print(f"Question: {sample['question']}")

    result_off = run(sample["question"], context=sample["context"], compression_enabled=False, backend="groq")
    print(f"\nWithout compression: {result_off['total_tokens']} tokens, "
          f"{result_off['latency_ms']:.0f}ms, answer: {result_off['response'][:100]}")

    result_on = run(sample["question"], context=sample["context"], compression_enabled=True,
                     compression_method="extractive", compression_level=0.5, backend="groq")
    print(f"With compression (50%): {result_on['total_tokens']} tokens, "
          f"{result_on['latency_ms']:.0f}ms, answer: {result_on['response'][:100]}")


def demo_context_manager():
    with open("data/locomo10.json", encoding="utf-8") as f:
        data = json.load(f)

    conv = data[0]
    session = conv["conversation"]["session_1"]
    query = conv["qa"][0]["question"]

    print("\n=== Context manager demo ===")
    print(f"Session length: {len(session)} turns")
    print(f"Question: {query}")

    result = run_conversation(session, query, context_strategy="sliding_window", max_turns=6, backend="groq")
    print(f"Answer: {result['response'][:150]}")
    print(f"Per-turn token counts: {result['per_turn_tokens']}")


if __name__ == "__main__":
    demo_compression()
    demo_context_manager()
