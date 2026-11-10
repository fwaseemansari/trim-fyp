# TRIM frontend + API

## Where to put this

Copy both `backend/` and `frontend/` folders into your `trim-fyp` repo root,
next to `pipeline/`, `token_analysis/`, `prompt_compression/`, `context_manager/`.

```
trim-fyp/
  backend/        <- new
  frontend/       <- new
  pipeline/
  token_analysis/
  prompt_compression/
  context_manager/
  ...
```

## Run the backend (from the repo root, not from inside backend/)

```powershell
cd trim-fyp
.\venv\Scripts\activate
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Check it worked: open http://localhost:8000/api/health in a browser, should show `{"status":"ok"}`.
No Groq/OpenAI key needed for this, tokenizing, compression and context management are all local.

## Run the frontend (separate terminal)

```powershell
cd trim-fyp\frontend
npm install
npm run dev
```

Open http://localhost:5173. The nav bar shows "backend connected" once both are running.

## Tech stack

- **Backend**: FastAPI (Python), a thin layer directly over your existing `token_analysis`,
  `prompt_compression`, and `context_manager` packages, no reimplementation, it imports and
  calls your real functions and classes.
- **Frontend**: React 18 + Vite (fast dev server, no webpack config needed), plain CSS
  (no Tailwind, kept the install simple), calling the backend with `fetch`.
- **Why FastAPI over Flask**: automatic request validation via Pydantic, and it's a natural fit
  since your pipeline is already typed (dataclasses in `results.py`).

## What's real vs not yet wired

- Token counting, compression (TF-IDF / bi-encoder / cross-encoder), and the context manager
  (sliding window / relevance-aware / hybrid) all call your actual Python code, live.
- The LLM summarization and rewrite-compression paths (anything needing a Groq/OpenAI key) are
  **not** wired into the UI yet, intentionally, so the demo works for anyone without an API key.
  Easy to add as a `/api/summarize` endpoint later if you want it in the report demo.
