# Context Manager — Final Summary (as actually built)

## What was planned (per Development Plan, Weeks 1-4)
Sliding-window baseline -> relevance-aware selection -> memory
summarization -> hybrid recency+relevance scoring (depth upgrade) ->
scaled evaluation across all LoCoMo sessions.

## What is actually built and verified
- `SlidingWindowStrategy` — keep last N turns. Verified via pytest on
  real LoCoMo data.
- `RelevanceAwareStrategy` — TF-IDF cosine similarity scoring, keeps
  turns within a token budget ranked by relevance. Verified on the full
  LoCoMo dataset (150 QA pairs, 10 conversations): 56.7% evidence
  retention vs. sliding window's 2.7%.
- `MemorySummarizationStrategy` — same TF-IDF scoring, but dropped turns
  are condensed via LLM call instead of discarded. Code verified to run
  correctly; actual summary *quality* still needs a live test with API
  access (script ready: `run_memory_test.py`).
- `HybridScoringStrategy` (Week 4 depth upgrade) — blends TF-IDF
  relevance with a normalized recency term. Built and tested; the
  finding was that recency blending **hurts** retention on this dataset
  (see `context_manager_notes_full.md`), since LoCoMo's evidence isn't
  recency-correlated. Kept in the codebase for comparison purposes, but
  `relevance_aware` remains the recommended default.

## Deviations from the plan
- The plan expected hybrid scoring to be an improvement ("this is the
  Context Manager's equivalent depth upgrade"); the actual result is a
  documented negative finding instead. This is scientifically valid and
  worth reporting as-is rather than reframing it as a win.
- Cross-backend comparison and ROUGE-L response-quality testing are
  fully scripted but not executed, since both need live Groq/OpenAI API
  access not available in the environment this work was done in.
- Evaluation scale: 150 sampled QA pairs across all 10 conversations
  (not the full ~2,000 QA pairs in the dataset) to keep runtime
  practical; the sampling is randomized with a fixed seed for
  reproducibility.

## Known limitation discovered during this work
15.5% of LoCoMo turns carry image content (`img_url` + `blip_caption`)
not captured by the current `text`-only pipeline — this affects the
whole team's evaluation numbers on any image-referencing QA pair, not
just the Context Manager. Flagged to the team; not yet resolved.

## What's left for Week 5 / Phase 2
- Run the memory-summarization, cross-backend, and ROUGE-L scripts with
  real API access and record genuine results.
- Decide, as a team, whether to fold `blip_caption` into evaluated text
  or explicitly scope image-based QA pairs out.
- Scale the evidence-retention test beyond the 150-pair sample if time
  allows (Phase 2 permits larger sample sizes since methods are frozen
  by then).
