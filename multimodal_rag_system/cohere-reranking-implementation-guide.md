# Adding Cohere Reranking to the Retrieval Logic — Step-by-Step Guide

> **For trainees:** This guide adds a **reranking stage** to the existing Multimodal
> RAG system. It builds directly on the system from
> `multimodal-rag-implementation-guide.md` — make sure that version runs end-to-end
> first (you can ingest a PDF and get answers from `POST /api/v1/query`).
>
> Every step explains _what_ you are changing, _why_ it matters, and _exactly_ what
> code to write. Follow the steps in order.

---

## Table of Contents

1. [Why add reranking? (the two-stage retrieval idea)](#1-why-add-reranking-the-two-stage-retrieval-idea)
2. [How it fits this project](#2-how-it-fits-this-project)
3. [Prerequisite — a Cohere API key](#3-prerequisite--a-cohere-api-key)
4. [Step 1 — Add the `cohere` dependency](#4-step-1--add-the-cohere-dependency)
5. [Step 2 — Add Cohere settings to `.env`](#5-step-2--add-cohere-settings-to-env)
6. [Step 3 — Create `src/core/rerank.py`](#6-step-3--create-srccorererankpy)
7. [Step 4 — Add a `retrieve()` orchestrator in `src/core/db.py`](#7-step-4--add-a-retrieve-orchestrator-in-srccoredbpy)
8. [Step 5 — Use `retrieve()` in `query_service.py`](#8-step-5--use-retrieve-in-query_servicepy)
9. [Step 6 — Expose `fetch_k` / `use_rerank` in the API](#9-step-6--expose-fetch_k--use_rerank-in-the-api)
10. [Step 7 — Test it](#10-step-7--test-it)
11. [How the data flows now](#11-how-the-data-flows-now)
12. [Tuning `fetch_k` and `top_n`](#12-tuning-fetch_k-and-top_n)
13. [Common errors and fixes](#13-common-errors-and-fixes)
14. [Optional — the LangChain `CohereRerank` alternative](#14-optional--the-langchain-coherererank-alternative)

---

## 1. Why add reranking? (the two-stage retrieval idea)

Right now retrieval is **one stage**: `similarity_search()` embeds the query and asks
pgvector for the `k` nearest chunks by cosine distance. That is fast and cheap, but the
embedding model is a *bi-encoder* — it embeds the query and each chunk **separately**,
then compares vectors. It never looks at the query and a chunk *together*, so it often:

- ranks a chunk highly because it shares vocabulary, even if it does not actually answer
  the question, and
- buries the truly relevant chunk just outside your top `k`.

**Reranking** fixes this with a second, more precise stage. A reranker is a
*cross-encoder*: it reads the query **and** a candidate chunk **at the same time** and
scores how relevant the chunk is to that specific query. Cross-encoders are far more
accurate than vector similarity — but too slow to run over your whole database. So we
combine the two:

```
Stage 1 (recall)     Vector search → fetch a WIDE candidate set (e.g. top 20).
                     Cheap, fast, "get everything that might be relevant."

Stage 2 (precision)  Cohere rerank → re-score those 20 against the query and
                     keep the best k (e.g. 5). Slow per item, but only 20 items.
```

This is called **two-stage retrieval** (or *retrieve-then-rerank*). You over-fetch with
the cheap method, then let the expensive-but-accurate method pick the final winners.

> **Why Cohere?** Cohere offers a hosted rerank endpoint (`rerank-v3.5`) — one API call,
> no model to host yourself. It takes a query plus a list of documents and returns
> relevance scores. It works on plain text, which fits this project perfectly: every
> chunk (text, table markdown, **and** image *description*) is stored as text in the
> `content` column, so all three chunk types can be reranked.

---

## 2. How it fits this project

You will **not** rewrite `similarity_search()`. You will wrap it:

**Before (one stage):**

```
query_service._build_messages()
        │
        ▼
db.similarity_search(query, k=5)   →   5 chunks   →   LLM
```

**After (two stages):**

```
query_service._build_messages()
        │
        ▼
db.retrieve(query, k=5, fetch_k=20)
        │
        ├── db.similarity_search(query, k=20)        ← Stage 1: vector recall (20 candidates)
        │
        └── rerank.rerank_chunks(query, candidates, top_n=5)  ← Stage 2: Cohere reranks → best 5
                                                              │
                                                              ▼
                                                       5 chunks → LLM
```

Files you will touch:

| File | Change |
|------|--------|
| `pyproject.toml` | add `cohere` dependency (via `uv add`) |
| `.env` / `.env.example` | add `COHERE_API_KEY`, `COHERE_RERANK_MODEL` |
| `src/core/rerank.py` | **new file** — the Cohere rerank call |
| `src/core/db.py` | add a `retrieve()` orchestrator (vector search → rerank) |
| `src/api/v1/services/query_service.py` | call `retrieve()` instead of `similarity_search()` |
| `src/api/v1/schemas/query_schema.py` | add `fetch_k` and `use_rerank` request fields |
| `src/api/v1/routes/query.py` | pass the new fields through |

---

## 3. Prerequisite — a Cohere API key

1. Create a free account at <https://dashboard.cohere.com/>.
2. Go to **API Keys** and copy a key (a free *trial* key is enough for this exercise).
3. Keep it handy for Step 2.

> **Trial-key limits:** Cohere's free trial keys are rate-limited (a small number of
> rerank calls per minute). That is fine for development and testing. If you hit a
> `429 Too Many Requests`, wait a moment and retry.

---

## 4. Step 1 — Add the `cohere` dependency

This project is **`uv`-managed** (see `CLAUDE.md`). Add the official Cohere Python SDK:

```bash
# Run from the project root: multimodal_rag_system/
uv add cohere
```

This installs `cohere` and records it in `pyproject.toml` and `uv.lock`. Verify:

```bash
uv run python -c "import cohere; print(cohere.__version__)"
```

You should see a version number (4.x or newer) with no import error.

---

## 5. Step 2 — Add Cohere settings to `.env`

Open your **`.env`** file (the real one — never commit it) and add two lines:

```bash
# Cohere reranking
COHERE_API_KEY="your-cohere-key-here"
COHERE_RERANK_MODEL="rerank-v3.5"
```

Then mirror the *keys* (with placeholder values) in **`.env.example`** so the next person
knows they are required:

```bash
# Cohere reranking
COHERE_API_KEY="...your-cohere-key-here..."
COHERE_RERANK_MODEL="rerank-v3.5"
```

> **Model choice:** `rerank-v3.5` is Cohere's latest, multilingual rerank model and a
> good default. Other options include `rerank-english-v3.0` (English-only) and
> `rerank-multilingual-v3.0`. We read the name from an env var so you can swap it without
> touching code.
>
> **Reminder:** Do **not** put real secrets in `.env.example`; it is tracked by git.

---

## 6. Step 3 — Create `src/core/rerank.py`

This new module is the **only** place that talks to Cohere. Keeping it separate means
`db.py` does not need to know rerank internals, and you can unit-test or swap the
reranker later without touching retrieval.

> Remember the project convention: **no `__init__.py` files** — `src` is a namespace
> package. This file just needs to live at `src/core/rerank.py` and be run from the
> project root so `from src...` imports resolve.

Create `src/core/rerank.py` with the following content:

```python
import os

import cohere
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Cohere rerank model name — read from the environment so it can be swapped
# without code changes. rerank-v3.5 is Cohere's latest multilingual reranker.
# ---------------------------------------------------------------------------
_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

# ---------------------------------------------------------------------------
# Cohere client — module-level singleton.
#
# Created lazily on first use (not at import time) so that importing this
# module does not fail when COHERE_API_KEY is absent, e.g. during ingestion
# or tests that never call the reranker. We use the V2 client, whose rerank()
# accepts a plain list of strings as `documents`.
# ---------------------------------------------------------------------------
_client: cohere.ClientV2 | None = None


def _get_client() -> cohere.ClientV2:
    """Return the module-level Cohere client, creating it on first call."""
    global _client
    if _client is None:
        _client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    return _client


def rerank_chunks(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """Re-score candidate chunks against the query and return the best `top_n`.

    Args:
        query:   The user's natural-language question.
        chunks:  Candidate chunk dicts from db.similarity_search(). Each must
                 have a "content" key (text, table markdown, or image
                 description) — that is the text Cohere actually reranks.
        top_n:   How many chunks to keep after reranking.

    Returns:
        A NEW list of chunk dicts, sorted best-first, each with an added
        "rerank_score" key (Cohere's relevance score, 0..1). The original
        chunk fields (content, chunk_type, image_base64, similarity, …) are
        preserved.

    Why rerank on `content`:
        Cohere's reranker is text-based. In this project every chunk — text,
        table, AND image — is stored as text in `content` (image chunks carry
        the vision-generated description). So all three types can be reranked
        with one call, exactly like they are all embedded from `content`.
    """
    if not chunks:
        return []

    # The text Cohere scores: one document per candidate chunk.
    documents = [chunk["content"] for chunk in chunks]

    response = _get_client().rerank(
        model=_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),  # never ask for more than we have
    )

    # response.results is sorted best-first. Each item has:
    #   .index           → position in the `documents` list we sent
    #   .relevance_score → Cohere's relevance score for that document
    reranked: list[dict] = []
    for result in response.results:
        chunk = dict(chunks[result.index])        # copy so we don't mutate caller's dict
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)

    return reranked
```

### What to notice

- **`response.results[i].index`** maps back to *your* input order. You must use it to pull
  the original chunk — do **not** assume Cohere returns them in the order you sent.
- **`top_n=min(top_n, len(documents))`** avoids an error when you over-fetched fewer
  candidates than `top_n` (e.g. a nearly empty database).
- We **copy** each chunk (`dict(...)`) before adding `rerank_score`, so we never mutate the
  list the caller handed us.

---

## 7. Step 4 — Add a `retrieve()` orchestrator in `src/core/db.py`

Per `CLAUDE.md`, retrieval lives in `src/core/db.py`. Add a thin orchestrator that runs
**Stage 1** (`similarity_search`) then **Stage 2** (`rerank_chunks`). The existing
`similarity_search()` stays exactly as-is — we are layering on top of it.

### 7.1 Add the import

Near the top of `src/core/db.py`, with the other imports, add:

```python
from src.core.rerank import rerank_chunks
```

> No circular-import risk: `rerank.py` does **not** import `db.py`.

### 7.2 Add the `retrieve()` function

Add this new function in `src/core/db.py`, right **after** `similarity_search()` (so it
sits next to the function it wraps):

```python
# ---------------------------------------------------------------------------
# Two-stage retrieval: vector recall + Cohere rerank
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    chunk_type: str | None = None,
    use_rerank: bool = True,
) -> list[dict]:
    """Retrieve the top `k` chunks, optionally refined by a Cohere reranker.

    Two-stage retrieval:
        Stage 1 (recall):    similarity_search() returns `fetch_k` candidates
                             by vector cosine distance — wide and cheap.
        Stage 2 (precision): rerank_chunks() re-scores those candidates with
                             Cohere and keeps the best `k`.

    Args:
        query:      Natural-language question.
        k:          Final number of chunks to return (after reranking).
        fetch_k:    Candidate pool size for Stage 1. Should be >= k; a larger
                    pool gives the reranker more to choose from (better recall)
                    at the cost of a slightly bigger rerank call.
        chunk_type: Optional filter — 'text', 'table', or 'image'.
        use_rerank: If False, skip Cohere and behave exactly like the old
                    single-stage similarity_search(query, k). Handy for A/B
                    comparison and for running without a COHERE_API_KEY.

    Returns:
        List of up to `k` chunk dicts (same shape as similarity_search), and
        when reranking is on, each carries an extra "rerank_score".
    """
    if not use_rerank:
        # Single-stage fallback: original behaviour, no Cohere call.
        return similarity_search(query, k=k, chunk_type=chunk_type)

    # Stage 1 — over-fetch a wide candidate pool by vector similarity.
    candidates = similarity_search(query, k=fetch_k, chunk_type=chunk_type)

    # Stage 2 — let Cohere re-score the pool and keep the best k.
    return rerank_chunks(query, candidates, top_n=k)
```

### What to notice

- `use_rerank=False` gives you a **clean off-switch** — same results as before. This is how
  you demonstrate the *difference* reranking makes (Step 10).
- `fetch_k` controls recall, `k` controls how much context reaches the LLM. They are now
  two separate knobs (see [Section 12](#12-tuning-fetch_k-and-top_n)).

---

## 8. Step 5 — Use `retrieve()` in `query_service.py`

Now route the query pipeline through the new orchestrator.

### 8.1 Update the import

In `src/api/v1/services/query_service.py`, change:

```python
from src.core.db import similarity_search
```

to:

```python
from src.core.db import retrieve
```

### 8.2 Update `_build_messages` signature and call

Change the function signature to accept the new knobs and call `retrieve()`:

```python
def _build_messages(
    query: str,
    k: int,
    chunk_type: str | None,
    fetch_k: int = 20,
    use_rerank: bool = True,
) -> tuple[list, list[dict]]:
    """Retrieve relevant chunks and build the multimodal LangChain message list."""
    chunks = retrieve(
        query,
        k=k,
        fetch_k=fetch_k,
        chunk_type=chunk_type,
        use_rerank=use_rerank,
    )
    # ... rest of the function is unchanged ...
```

> Only the first line of the body changes (`similarity_search(...)` → `retrieve(...)`).
> Everything that builds `message_parts` and `sources` stays the same.

### 8.3 Surface the rerank score in `sources` (recommended)

Still inside `_build_messages`, where the `source_entry` dict is built, add the rerank
score so clients (and you, while debugging) can see it. Find this block:

```python
        source_entry: dict = {
            "chunk_type": ct,
            "page_number": page,
            "section": section,
            "source_file": chunk.get("source_file", ""),
            "element_type": chunk.get("element_type"),
            "similarity": round(chunk.get("similarity", 0), 4),
        }
```

and add one line:

```python
        source_entry: dict = {
            "chunk_type": ct,
            "page_number": page,
            "section": section,
            "source_file": chunk.get("source_file", ""),
            "element_type": chunk.get("element_type"),
            "similarity": round(chunk.get("similarity", 0), 4),
            "rerank_score": round(chunk["rerank_score"], 4) if "rerank_score" in chunk else None,
        }
```

Now every source shows both its **vector similarity** (Stage 1) and its **rerank score**
(Stage 2) — great for seeing how the ranking changed.

### 8.4 Thread the knobs through the public functions

Update `query_documents()` and `stream_query_documents()` to accept and forward the new
parameters.

`query_documents`:

```python
def query_documents(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
    fetch_k: int = 20,
    use_rerank: bool = True,
) -> dict:
    messages, sources = _build_messages(
        query, k, chunk_type, fetch_k=fetch_k, use_rerank=use_rerank
    )
    response = _llm.invoke(messages)
    return {
        "answer": _extract_text(response.content),
        "sources": sources,
    }
```

`stream_query_documents` (only the signature and the `_build_messages` call change):

```python
async def stream_query_documents(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
    fetch_k: int = 20,
    use_rerank: bool = True,
):
    messages, sources = _build_messages(
        query, k, chunk_type, fetch_k=fetch_k, use_rerank=use_rerank
    )
    # ... rest unchanged ...
```

---

## 9. Step 6 — Expose `fetch_k` / `use_rerank` in the API

### 9.1 Add request fields in `query_schema.py`

In `src/api/v1/schemas/query_schema.py`, extend `QueryRequest`:

```python
from pydantic import BaseModel, Field
from typing import List, Optional


# ---- Request ----
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to send to the LLM (after reranking)")
    fetch_k: int = Field(
        20, ge=1, le=100,
        description="Candidate pool size for vector search before reranking (should be >= k)",
    )
    use_rerank: bool = Field(
        True, description="If true, refine vector results with the Cohere reranker"
    )
    chunk_type: Optional[str] = Field(
        None, description="Filter by content type: 'text', 'table', or 'image'"
    )


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
```

### 9.2 Pass them through in `routes/query.py`

Update both endpoints in `src/api/v1/routes/query.py` to forward the new fields:

```python
@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    result = query_documents(
        request.query,
        k=request.k,
        chunk_type=request.chunk_type,
        fetch_k=request.fetch_k,
        use_rerank=request.use_rerank,
    )
    return QueryResponse(**result)


@router.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    return StreamingResponse(
        stream_query_documents(
            request.query,
            k=request.k,
            chunk_type=request.chunk_type,
            fetch_k=request.fetch_k,
            use_rerank=request.use_rerank,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

> Keep the existing docstring on `query_stream_endpoint` — only the call arguments change.

---

## 10. Step 7 — Test it

You need the usual runtime prerequisites in place: **Postgres + pgvector running with
`schema.sql` loaded**, **at least one PDF ingested**, a live **`OPENAI_API_KEY`**, and now
also a **`COHERE_API_KEY`**.

### 10.1 Smoke-test the reranker in isolation

Before touching the API, confirm Cohere works on its own:

```bash
uv run python -c "
from src.core.rerank import rerank_chunks
chunks = [
    {'content': 'The capital of France is Paris.'},
    {'content': 'Bananas are a good source of potassium.'},
    {'content': 'Paris hosted the 2024 Summer Olympics.'},
]
out = rerank_chunks('What city is the capital of France?', chunks, top_n=2)
for c in out:
    print(round(c['rerank_score'], 4), c['content'])
"
```

Expected: the two Paris sentences come back on top, each with a `rerank_score`. If you see
an auth error, your `COHERE_API_KEY` is missing or wrong.

### 10.2 Test the full pipeline (reranking ON)

Start the API:

```bash
uv run uvicorn main:app --reload
```

Then query (use a question your ingested PDF can answer):

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What was the quarterly revenue?",
        "k": 5,
        "fetch_k": 20,
        "use_rerank": true
      }' | python -m json.tool
```

In `sources`, each entry now has **both** `similarity` and `rerank_score`.

### 10.3 Compare against reranking OFF

Send the same query with `"use_rerank": false` and compare the `sources`:

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What was the quarterly revenue?",
        "k": 5,
        "use_rerank": false
      }' | python -m json.tool
```

With reranking **off** you get the raw vector top-5. With it **on**, the order (and often
the membership) of the top-5 changes — that is the reranker promoting the chunks that
actually answer the question. This side-by-side is the whole point of the exercise.

---

## 11. How the data flows now

```
POST /api/v1/query  {query, k=5, fetch_k=20, use_rerank=true}
        │
        ▼
routes/query.py  →  query_service.query_documents()
        │
        ▼
query_service._build_messages()
        │
        ▼
db.retrieve(query, k=5, fetch_k=20, use_rerank=True)
        │
        ├─ Stage 1: db.similarity_search(query, k=20)
        │     • embed query with OpenAI
        │     • pgvector cosine search (ORDER BY embedding <=> query)
        │     • return 20 candidate chunks (with `similarity`)
        │
        └─ Stage 2: rerank.rerank_chunks(query, candidates, top_n=5)
              • send query + 20 chunk `content` strings to Cohere
              • Cohere cross-encoder scores each
              • return best 5 (with `rerank_score`)
        │
        ▼
build multimodal prompt (text + table + image parts)  →  ChatOpenAI (vision)
        │
        ▼
{answer, sources[]}   ← each source has similarity AND rerank_score
```

The vector index does the cheap heavy lifting (narrow a whole database down to 20); Cohere
does the expensive precision work (pick the best 5 of 20).

---

## 12. Tuning `fetch_k` and `top_n`

`fetch_k` (candidate pool) and `k` / `top_n` (final results) are independent knobs:

| Knob | Too low | Too high | Sensible start |
|------|---------|----------|----------------|
| `fetch_k` | Reranker can't recover a chunk vector search missed | Slower + more reranker cost per query | **20** (try 30–50 for big corpora) |
| `k` | LLM may lack context to answer | More tokens, more cost, risk of distraction | **5** |

Rules of thumb:

- Always keep **`fetch_k >= k`** (otherwise the reranker has nothing extra to choose from
  and the whole stage is pointless).
- Reranking can only **reorder/keep** what Stage 1 retrieved — it cannot conjure a chunk
  that vector search never returned. If the right answer is missing even at high `fetch_k`,
  the problem is upstream (chunking, embeddings, or the data), not the reranker.
- Cohere bills/limits by rerank call and document count, so larger `fetch_k` = more cost
  and latency. Increase it only until answer quality stops improving.

---

## 13. Common errors and fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `CohereAPIError: invalid api token` / 401 | `COHERE_API_KEY` missing or wrong | Set it in `.env`; restart the server so `load_dotenv()` re-reads it |
| `ModuleNotFoundError: No module named 'cohere'` | dependency not installed | `uv add cohere`, then run with `uv run` |
| `429 Too Many Requests` | Cohere trial-key rate limit | Wait and retry; lower `fetch_k`; or upgrade the key |
| `ImportError: cannot import name 'retrieve'` | edited the wrong file / typo | Confirm `retrieve()` is defined in `src/core/db.py` and `query_service.py` imports it |
| Results identical with rerank on/off | `use_rerank` not threaded through, or `fetch_k == k` | Verify the route forwards `use_rerank`/`fetch_k`; set `fetch_k > k` |
| `KeyError: 'content'` in `rerank_chunks` | candidate dicts lack `content` | Ensure you pass the output of `similarity_search()` straight through |
| Works in API but `python -m ...` import fails | run from project root | Run modules from `multimodal_rag_system/` so `from src...` resolves (no `__init__.py`) |

---

## 14. Optional — the LangChain `CohereRerank` alternative

This project already uses LangChain, so you *could* use its built-in
`langchain_cohere.CohereRerank` compressor instead of the raw SDK. It wraps the same
endpoint behind LangChain's `compress_documents()` interface.

```bash
uv add langchain-cohere
```

```python
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

_reranker = CohereRerank(model="rerank-v3.5")  # reads COHERE_API_KEY from env

def rerank_chunks(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    docs = [Document(page_content=c["content"], metadata={"i": i}) for i, c in enumerate(chunks)]
    ranked = _reranker.compress_documents(docs, query=query)[:top_n]
    out = []
    for d in ranked:
        chunk = dict(chunks[d.metadata["i"]])
        chunk["rerank_score"] = d.metadata.get("relevance_score")
        out.append(chunk)
    return out
```

**Which to choose?** For *learning*, the raw `cohere` SDK in Step 3 is clearer — you can
see the exact request and the `index`/`relevance_score` it returns. The LangChain wrapper
is convenient if you later plug reranking into a LangChain retriever chain
(`ContextualCompressionRetriever`). Either way, the rest of this guide (`retrieve()`,
`query_service`, schema, routes) is unchanged.
```
