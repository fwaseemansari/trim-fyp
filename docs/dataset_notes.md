# LoCoMo Dataset Notes

The LoCoMo dataset (locomo10.json) contains 10 long-term conversations 
between pairs of speakers, with each conversation split into 19 to 32 
sessions (spread across different dates, simulating a real relationship 
over time). Each session holds roughly 15-25 turns, with an average 
session length of about 550 words. Every conversation comes with a large 
set of QA pairs (105 to 260 per conversation, 199 in the first sample) 
used to test long-term memory; each QA pair has a question, an answer, 
a "category" number (1-5, reflecting different reasoning types such as 
single-hop, multi-hop, and temporal questions), and an "evidence" field 
pointing to the exact turn(s) (e.g., D1:3) that contain the answer. This 
evidence field is what will let us check whether our Context Manager 
kept the right turns in its window when the correct answer required 
information from several sessions ago.