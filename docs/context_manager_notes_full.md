# Context Manager Notes — Findings (Weeks 1-4)

## How this was tested
Ran `sliding_window`, `relevance_aware`, and `hybrid_scoring` strategies
against real LoCoMo data — first a single conversation (`conv-26`, 419
turns, all 197 QA pairs), then the **full dataset**: all 10 conversations,
15 sampled QA pairs each with evidence (150 total), real turns, real
evidence-field checking.

Token budgets used a local word-count approximation instead of the real
tiktoken/HF tokenizer (blocked by network access in the test environment).
The TF-IDF/hybrid ranking logic itself is real and unaffected by this —
re-run with the actual `count_tokens` once you have local API/tokenizer
access, to get exact token-budget numbers.

## Headline results (full dataset, 150 QA pairs, all 10 conversations)
| Strategy | Evidence retained |
|---|---|
| sliding_window (last 10 turns) | 4/150 (2.7%) |
| relevance_aware (TF-IDF) | 85/150 (56.7%) |
| hybrid_scoring (recency_weight=0.3) | 26/150 (17.3%) |

## Key finding #1: sliding window fails almost immediately
Sliding window essentially never retains the correct evidence once a
conversation exceeds its window size (10 turns) — evidence scattered
across hundreds of turns is structurally invisible to a fixed recency
window, regardless of how important the information is.

## Key finding #2: evidence retention decays with distance (the "point of failure" graph)
Tested retention vs. how far back the evidence is (in turns from the end
of the conversation), on conv-26:
- 0-100 turns back: 73% retained (relevance-aware)
- 100-250 turns back: 67%
- 250-400 turns back: 37%
This is the "point where the baseline starts failing that the
relevance-aware version still gets right" result the plan asks for —
except relevance-aware *also* degrades with distance, just less sharply
than sliding window's near-total failure. See `chart_retention_vs_distance.png`.

## Key finding #3: hybrid recency+relevance scoring made things WORSE, not better
This was a real surprise. Swept `recency_weight` from 0.0 to 0.5 on a
25-question sample:
| recency_weight | retention |
|---|---|
| 0.0 (= pure relevance) | 52% |
| 0.1 | 28% |
| 0.2 | 20% |
| 0.3 | 12% |
| 0.5 | 0% |

**Why:** LoCoMo's QA evidence is scattered essentially uniformly across
a conversation's whole timeline, not concentrated near the end. Blending
in *any* recency signal actively displaces correct-but-old evidence in
favor of irrelevant-but-recent turns. For this specific task (retrieving
scattered historical facts), pure relevance scoring beats hybrid scoring.
This is a genuine negative result, not a bug — worth reporting honestly
as a finding, including in the report's Discussion section: naive
recency blending is not automatically an improvement, and the right
design depends on whether the target information is actually
recency-correlated.

## Key finding #4 (significant, team-wide): 15.5% of turns carry image content the pipeline currently ignores
LoCoMo turns can include an `img_url` + `blip_caption` field (e.g. a
turn's `text` says "take a look at this" while `blip_caption` says "a
photo of a painting of a sunset over a lake" — the actual answerable
content). **This affects 910 of 5,882 turns (15.5%) across the dataset.**

Right now, `ContextStrategy.get_context()` only reads `turn["text"]`,
so any question whose evidence lives in an image caption is
unanswerable by the pipeline **regardless of which context strategy is
used** — this isn't a Context Manager-specific bug, it's a data-handling
gap that likely affects the whole team's evaluation numbers on any
QA pair tied to an image turn. Worth flagging to Faiqa/Easha: either
fold `blip_caption` into the turn text LoCoMo-side, or explicitly scope
image-referencing questions out of the evaluation set and note it as a
stated limitation.

## Failure case root-cause analysis (Week 4 task)
1. **"When did Melanie paint a sunrise?"** (evidence D1:12) — this is
   exactly the image-caption issue above: the evidence turn's `text` is
   "You'd be a great counselor!... take a look at this," while the
   actual sunrise-painting content is in `blip_caption`, invisible to
   TF-IDF scoring entirely.
2. **"What fields would Caroline be likely to pursue in her education?"**
   (evidence D1:11, D1:9) — an inferential question with no single
   lexical match; the evidence turns say "keen on counseling" and
   "continue my edu," neither of which shares vocabulary with "fields"
   or "pursue," so TF-IDF's word-overlap approach can't connect them.
3. **"What did Caroline research?"** (evidence D2:8) — short, generic
   query terms ("research") likely matched many unrelated turns equally
   well, diluting the correct turn's relative score.

## Limitations to raise in the report
- TF-IDF is lexical, not semantic — fails on paraphrased/inferential
  questions (root causes #2, #3 above). A sentence-transformers
  embedding approach is the natural next step.
- The image-caption gap (#4) is a data-pipeline limitation independent
  of context strategy — should be flagged as a known limitation
  regardless of which strategy "wins."
- Hybrid recency+relevance scoring, as implemented, underperforms pure
  relevance on this task — kept in the codebase as `hybrid_scoring` for
  completeness/comparison, but relevance_aware remains the stronger
  default strategy based on this evidence.
- `memory_summarization`'s actual LLM-summary quality still needs a live
  test with real API access — run `context_manager/run_memory_test.py`
  locally once your `.env` is confirmed working.
- Cross-backend tokenizer/latency comparison (`run_backend_check.py`)
  and response-quality ROUGE-L testing (`run_rouge_test.py`) are written
  and ready but not run here — both need live API access.

## Live findings (real API runs, Sep 30)
- Memory test: the needed fact ("charity race last Saturday") survived
  directly in active context. But absolute-date gold answers (e.g.
  "25 May 2023") can't be produced, since session date metadata is
  never injected into the prompt — same issue surfaced independently
  in the ROUGE test below.
- Backend check: Groq/gpt-oss-20b and OpenAI/tiktoken tokenizer counts
  matched exactly (453 vs 453) on this sample. OpenAI call failed with
  `insufficient_quota` (account has no credits) — a real, concrete
  instance of the budget constraint already noted in the report.
- ROUGE-L: 0.089 (without CM) vs 0.093 (with CM) on 10 real QA pairs.
  The small numeric gap understates the real improvement — CM answers
  are substantively correct but phrased relatively ("yesterday," "last
  Saturday") while gold answers are absolute dates, which ROUGE-L
  penalizes as a near-total mismatch despite being factually right.
- One genuine wrong answer: "What did Caroline research?" → CM
  answered "pottery techniques" (gold: "Adoption agencies") — likely a
  generation-level failure, not a retrieval failure; worth inspecting
  that specific context window separately.