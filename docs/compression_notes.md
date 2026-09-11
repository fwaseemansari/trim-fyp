# Extractive Compression — Failure Case Analysis (Week 2 Sweep)

**Status:** real findings, from manually inspecting `evaluation/results/sweep_extractive_*pct.csv`
(30%/50%/70% levels, 20 SQuAD v2 samples each). 3 samples lost their
answer at every compression level tested (30-70%), not just at
aggressive compression — meaning compression *level* isn't what
determines these failures; something structural about TF-IDF scoring
is.

## Method
Compared the `answer_still_present` column across all 3 sweep CSVs,
found the sample IDs that were `False` at every level
(`Compare-Object` on the failing-ID lists per level returned no
difference — same 3 samples every time). Manually read each sample's
question, answer, and full context against what TF-IDF actually ranked.

## Findings — 3 distinct failure modes, not one

**1. Spelling/dialect variation** (`5710eca0a58dae1900cd6b3e`)
Query used "naturalized" (American spelling); source passage used
"naturalise" (British spelling, consistent with the passage's UK
English throughout). Zero token overlap on the one word that should
anchor the match. `TfidfVectorizer` does no stemming/lemmatization by
default, so these are two unrelated tokens to it.

**2. Synonym gap + common-term dilution** (`57108d69b654c5140001f985`)
Query's key term "emigration" never appears in the passage at all
(passage uses "revocation," "refugees," "migration," "influx"
instead) — contributes zero signal. Meanwhile the term that DOES
overlap ("Dutch Republic") appears in 4 of the passage's 6 sentences,
so IDF down-weights it as a common/weak signal rather than a strong,
distinguishing one — backwards from what's needed here.

**3. Cross-sentence anaphora** (`57340d124776f419006617c3`)
The answer sentence ("Most went to Cuba...") has almost no
independent lexical overlap with the query — "Most" is a pronoun
referring back to "Spanish Catholic population" named in the
*previous* sentence. TF-IDF scores each sentence independently, so it
can't see that this sentence's meaning depends on its neighbor.

## Implication for Week 3 (chunk-level, embedding-based compression)
Modes 1 and 2 are exactly what embedding-based relevance scoring
(`sentence-transformers`) should fix — semantic similarity doesn't
require literal string overlap. Mode 3 is a genuinely open limitation:
sentence-independent scoring (TF-IDF OR embeddings) will still miss
sentences that only make sense combined with a neighbor. Worth naming
honestly in Chapter 9 rather than implying the Week 3 upgrade solves
everything — this is a legitimate boundary of the sentence-level
compression approach in general, not a bug in this implementation.

## Caveat
n=3 failures out of 20 samples — real and specific, but too small a
sample to claim these 3 modes are exhaustive. Worth re-checking once
the sample size scales to 100+ in later weeks: are these still the
only 3 failures, or do new modes appear.