"""
TRIM API — thin FastAPI layer over the existing pipeline code.

Run from the repo root (not from inside backend/) so the sibling packages
(token_analysis, prompt_compression, context_manager, pipeline) import
cleanly:

    uvicorn backend.main:app --reload --port 8000

No Groq/OpenAI key is required for these endpoints — tokenizing, compressing
and context management are all local operations. Only a future
/api/summarize-style endpoint that actually calls an LLM would need one.
"""
import os
import sys

# Make sure the repo root is importable regardless of the working directory
# uvicorn was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from token_analysis.counter import count_tokens
from token_analysis.pricing import estimate_cost
from prompt_compression.compressor import build_compressor
from context_manager.manager import ContextManager

app = FastAPI(title="TRIM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- /api/tokenize ----------
class TokenizeRequest(BaseModel):
    text: str
    model: str = "gpt-4o-mini"


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest):
    try:
        tokens = count_tokens(req.text, req.model)
        cost = estimate_cost(req.model, tokens, 0)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    return {"tokens": tokens, "estimated_cost_usd": round(cost, 6), "model": req.model}


# ---------- /api/compress ----------
class CompressRequest(BaseModel):
    text: str
    query: str = ""
    method: str = "selective"      # extractive | selective | drop
    scorer: str = "cross_encoder"  # tfidf | bi_encoder | cross_encoder
    level: float = 0.5
    backend: str = "groq"


@app.post("/api/compress")
def compress(req: CompressRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    try:
        compressor = build_compressor(req.method, backend=req.backend, scorer=req.scorer)
        result = compressor.compress(req.text, req.query, req.level)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "text": result.text,
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "token_reduction_pct": result.token_reduction_pct,
        "kept_indices": result.kept_indices,
        "method": result.method,
        "scorer": result.scorer,
        "segment_count": result.segment_count,
    }


# ---------- /api/context ----------
class Turn(BaseModel):
    speaker: str = ""
    text: str


class ContextRequest(BaseModel):
    turns: list[Turn]
    strategy: str = "sliding_window"   # sliding_window | relevance_aware | hybrid_scoring
    max_turns: int = 6
    max_tokens: int = 1000
    query: str = ""


@app.post("/api/context")
def context(req: ContextRequest):
    try:
        manager = ContextManager(
            strategy=req.strategy, max_turns=req.max_turns, max_tokens=req.max_tokens
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    all_turns = [t.model_dump() for t in req.turns]
    for t in all_turns:
        manager.add_turn(t, req.query)

    kept_texts = {id(t): t for t in manager.history}  # not reliable across calls, so compare by content
    kept_set = {t["text"] for t in manager.history}
    annotated = [{"speaker": t["speaker"], "text": t["text"], "kept": t["text"] in kept_set} for t in all_turns]

    return {
        "turns": annotated,
        "context": manager.get_context(),
        "total_tokens": manager.total_tokens(),
        "kept_count": len(manager.history),
        "dropped_count": len(all_turns) - len(manager.history),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
