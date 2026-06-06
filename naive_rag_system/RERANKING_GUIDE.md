# Reranking RAG with Cohere Rerank — Step-by-Step Guide

Build a **two-stage retriever** on top of the existing hybrid retriever
(`app/retrieval/retrieval_hybrid.py`). Stage 1 casts a wide net cheaply;
Stage 2 reorders that shortlist accurately with Cohere's reranker.

---

## 1. Why rerank?

Hybrid search (vector + full-text) is **fast but approximate**:

- **Bi-encoder** (embeddings): the query and each chunk are vectorized
  *separately*, so the match is only roughly right. Fast enough to run over the
  whole collection.
- **Cross-encoder** (Cohere Rerank): reads the query and a chunk *together* in
  one pass, so it judges relevance far more accurately — but it is too slow to
  run over every chunk.

The two-stage pattern gets the best of both:

```
query ─► query_documents(k=25) ─► Cohere rerank ─► top 5
         Stage 1: RECALL           Stage 2: PRECISION
         vector + FTS + RRF        cross-encoder reorders
         (cheap, wide net)         (accurate, paid API call)
```

We deliberately **over-fetch** (`fetch_k=25`) so the right chunk is *somewhere*
in the pool, then let the reranker pull it to the top and trim to `k=5`.

---

## 2. Prerequisites

| Requirement | How to check |
|---|---|
| Postgres + pgvector running | `psql` connects to the DB in `PG_CONNECTION_STRING` |
| `hr_support_desk` collection ingested | run `app/ingestion/ingestion_v1.py` first |
| `COHERE_API_KEY` in `.env` | ✅ already present |
| `COHERE_RERANK_MODEL` in `.env` | ✅ already present (e.g. `rerank-v3.5`) |

> The `.env.example` documents both Cohere variables. Never commit the real
> `.env`.

---

## 3. Install the Cohere SDK

```bash
uv add cohere
```

This adds `cohere` to `pyproject.toml` and `uv.lock`.

---

## 4. Create the reranking retriever

Create **`app/retrieval/retrieval_rerank.py`**. It imports `query_documents`
from the hybrid retriever as its Stage-1 candidate generator — so
`retrieval_hybrid.py` needs **no changes**.

```python
# app/retrieval/retrieval_rerank.py
#
# Reranking RAG: two-stage retrieval built on top of retrieval_hybrid.py
#   Stage 1 (recall):    over-fetch a wide candidate pool with hybrid search
#   Stage 2 (precision): re-score with Cohere's cross-encoder, keep the best k
#
# HOW TO RUN:
#   macOS/Linux:        PYTHONPATH=. uv run app/retrieval/retrieval_rerank.py
#   Windows (PowerShell): $env:PYTHONPATH="."; uv run app/retrieval/retrieval_rerank.py

import os
from dotenv import load_dotenv
import cohere

from app.retrieval.retrieval_hybrid import query_documents

load_dotenv()

# rerank-v3.5 is Cohere's latest general-purpose reranker (English + multilingual).
_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

# Build the client lazily and cache it: lets the module import even with no key,
# and surfaces a clear error only when reranking is actually called.
_cohere_client: cohere.ClientV2 | None = None


def _get_cohere_client() -> cohere.ClientV2:
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. Add it to your .env (see .env.example)."
            )
        _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client


def rerank_search(query: str, k: int = 5, fetch_k: int = 25) -> list[dict]:
    """
    Retrieve `fetch_k` candidates with hybrid search, then return the top `k`
    after Cohere reranking.

    Args:
        query:   User query string.
        k:       Number of final results to return after reranking.
        fetch_k: Size of the candidate pool to over-fetch before reranking.
                 Larger = better recall but a slightly slower / costlier rerank.

    Returns:
        List of dicts with 'content', 'metadata', and 'rerank_score',
        ordered by relevance (best first).
    """
    # Stage 1: over-fetch a wide candidate pool from the hybrid retriever.
    candidates = query_documents(query, k=fetch_k)
    if not candidates:
        return []

    # Stage 2: rerank the candidates with Cohere's cross-encoder.
    documents = [c["content"] for c in candidates]
    response = _get_cohere_client().rerank(
        model=_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=k,
    )

    # response.results is sorted best-first; result.index points back into `candidates`.
    reranked: list[dict] = []
    for result in response.results:
        original = candidates[result.index]
        reranked.append({
            "content":      original["content"],
            "metadata":     original["metadata"],
            "rerank_score": round(float(result.relevance_score), 4),
        })

    return reranked


if __name__ == "__main__":
    query = "What is the leave policy for employees?"
    results = rerank_search(query, k=5, fetch_k=25)

    print(f"\nTop {len(results)} reranked results for: '{query}'\n{'=' * 60}")
    for i, item in enumerate(results, 1):
        metadata = item["metadata"]
        print(f"\n[{i}] score={item['rerank_score']} | "
              f"Source: {metadata.get('source')} | Page: {metadata.get('page')}")
        print(item["content"][:400])
```

### Two details worth understanding

- **Lazy client (`_get_cohere_client`)** — Cohere's client raises if built with
  no key, so constructing it at import time would make the file un-importable
  until a key exists. Building it inside a cached helper avoids that.
- **`result.index`** — Cohere returns results sorted best-first, each carrying
  the *original* index into the documents you sent. We use it to map the
  relevance score back onto our own dict (keeping `metadata`).

---

## 5. Run it

```bash
# macOS/Linux
PYTHONPATH=. uv run app/retrieval/retrieval_rerank.py

# Windows (PowerShell)
$env:PYTHONPATH="."; uv run app/retrieval/retrieval_rerank.py
```

Expected output: 5 chunks, each with a calibrated `score` (0–1), best first.

---

## 6. Prove it works — before vs. after

Run the same query through both retrievers to see the reranker reorder results:

```python
from app.retrieval.retrieval_hybrid import query_documents
from app.retrieval.retrieval_rerank import rerank_search

q = "What is the leave policy for employees?"

print("HYBRID:")
for d in query_documents(q, k=5):
    print("  ", d["content"][:70])

print("\nRERANKED:")
for d in rerank_search(q, k=5):
    print(f"   {d['rerank_score']:.3f}  {d['content'][:70]}")
```

The reranker should promote the genuinely on-topic chunk and attach a
relevance score to each result.

---

## 7. Tuning knobs (Accuracy / Latency / Cost)

| Knob | Effect |
|---|---|
| `fetch_k` ↑ | Better recall, but more text sent to Cohere → higher latency + cost. Sweet spot: 20–50. |
| `k` | Final context size handed to your LLM. |
| `COHERE_RERANK_MODEL` | `rerank-v3.5` (latest, multilingual) vs `rerank-english-v3.0` (English-only, cheaper). |

The reranker is **one paid API call per query** — that is the cost/latency you
trade for the accuracy gain.

---

## 8. Optional: the LangChain alternative

LangChain offers `CohereRerank` + `ContextualCompressionRetriever`, but that
path expects `Document` objects and a LangChain base retriever. Since
`retrieval_hybrid.py` returns plain **dicts**, the direct Cohere SDK approach
above is simpler and more transparent.

```python
# Sketch only — requires a Document-returning base retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

compressor = CohereRerank(model="rerank-v3.5", top_n=5)
retriever  = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=your_base_retriever,  # must return Documents
)
```

---

## Summary

1. `uv add cohere`
2. Keep `COHERE_API_KEY` + `COHERE_RERANK_MODEL` in `.env`
3. Add `app/retrieval/retrieval_rerank.py` (over-fetch from hybrid → Cohere rerank)
4. Run it; compare hybrid vs. reranked output
5. Tune `fetch_k` / `k` for your accuracy–latency–cost target
