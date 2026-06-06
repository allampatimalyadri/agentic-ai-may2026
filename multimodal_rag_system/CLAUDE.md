# CLAUDE.md

Guidance for Claude Code when working in this repository.

do not access .env . you access .env.example

## What this is

A **multimodal RAG system** built as a teaching project for a corporate Agentic-AI
training course. It ingests PDFs (text, tables, and images) with Docling, embeds every
chunk into a shared vector space, stores them in PostgreSQL + pgvector, and answers
natural-language questions via a FastAPI endpoint backed by a vision-capable LLM.

This directory is one sub-project inside a larger course repo (git root is
`agentic-ai-course-may2026/`, alongside `naive_rag_system`, `langchain_examples`, etc.).
Treat `multimodal_rag_system/` as the project root for this work.

`multimodal-rag-implementation-guide.md` is the trainee-facing, line-by-line build guide.
It is the source of intent — when code and guide disagree, the guide shows what the
finished system is _supposed_ to look like.

## Layout

```
schema.sql                              pgvector DDL: documents + multimodal_chunks tables
main.py                                 FastAPI app entry point: `app`, `/`, `/health`, mounts query router at /api/v1
src/ingestion/docling_parser.py         PDF → typed chunks (text/table/image) via Docling
src/ingestion/ingestion.py             orchestrates parse → split → embed → store
src/core/db.py                          embeddings, connection pool, store + similarity search
src/api/v1/routes/query.py             FastAPI router: POST /query and /query/stream (SSE)
src/api/v1/services/query_service.py   RAG: retrieve chunks → build multimodal prompt → LLM
src/api/v1/schemas/query_schema.py     Pydantic request/response models
data/                                   ingested PDFs + data/images/ (extracted PNGs)
```

Data flow: `ingestion.run_ingestion(pdf)` → `parse_document()` (Docling) →
`_split_text()` for long text → `db.store_chunks()` (embed + insert) →
query time: `query_service.query_documents()` → `db.similarity_search()` (cosine via `<=>`) →
LLM answer with sources.

## Commands

```bash
# Install/sync dependencies (uv-managed, Python 3.13)
uv sync

# One-time DB setup (Postgres with pgvector must be running)
psql "<conn>" -f schema.sql

# Ingest a PDF (defaults to data/RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf if no arg given)
uv run python -m src.ingestion.ingestion path/to/file.pdf

# Run the API
uv run uvicorn main:app --reload
# Swagger UI at http://localhost:8000/docs ; query at POST /api/v1/query
```

There is no test suite, linter config, or CI in this project.

## Conventions

- **Package manager is `uv`** — use `uv run` / `uv sync`, not bare `pip`/`python`.
- **No `__init__.py` files**; `src` is imported as a namespace package. Run modules from the
  project root (`python -m src.ingestion.ingestion`) so `from src...` imports resolve.
- **pgvector literals are built as strings** (`"[" + ",".join(...) + "]"` then `::vector`)
  to avoid the pgvector Python package. Embedding dimension is **1536** and must match
  `VECTOR(1536)` in `schema.sql`.
- **Three chunk types only**: `text`, `table`, `image` (enforced by a CHECK constraint).
  Tables and images are never split; long text is windowed (1500 chars, 300 overlap).
- **Images live on disk**, not in Postgres: bytes are written to `data/images/` and only the
  path is stored. `similarity_search()` re-reads and base64-encodes them for callers.
- **Ingestion is idempotent**: `upsert_document()` reuses the doc UUID per filename, and
  `store_chunks()` deletes existing chunks for that `doc_id` before re-inserting.
- `# Issue N fix` / `# Feature N` comments are training-exercise markers, not TODOs.

## Provider & runtime state

The system is **OpenAI-based and runnable end-to-end** (an earlier Gemini migration was
reverted). Consistent across the stack:

- **Code uses OpenAI** via `langchain_openai` — `ChatOpenAI` for the vision LLM
  (`query_service.py`, `docling_parser.py`) and `OpenAIEmbeddings` for embeddings
  (`db.py`). Both `openai` and `langchain-openai` are in `pyproject.toml` and installed.
- **Env vars** read by the code are `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` (default
  `gpt-4o-mini`), `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`), and
  `PG_CONNECTION_STRING` — matching `.env.example`. `text-embedding-3-small` is 1536-dim,
  matching `VECTOR(1536)` in `schema.sql`.
- **`main.py` is the real FastAPI app** (`app`, `/`, `/health`, query router at `/api/v1`).

Runtime prerequisites before queries return anything useful:

1. **Postgres + pgvector running**, with `schema.sql` loaded (`vector` extension enabled).
2. **At least one PDF ingested** — otherwise `similarity_search()` retrieves nothing.
3. **A live `OPENAI_API_KEY`** in `.env` — embeddings and the vision LLM both call OpenAI.

The app *starts* without the DB or key, but `/api/v1/query` errors until both are in place.

## Security note

`.env.example` currently holds a **placeholder** `OPENAI_API_KEY` (`sk-...your-key-here...`),
which is correct. `.gitignore` excludes `.env` but **not** `.env.example`, so keep real
secrets out of the example file. Note it does carry a real-looking `PG_CONNECTION_STRING`
with local-dev credentials. Do not read or commit `.env`; use `.env.example` for reference.
