# Custom schema ingestion — no langchain_pg_* tables.
# You own the DDL: table name, columns, indexes, everything.
# Paired with: app/retrieval/retrieval_custom.py
#
# HOW TO RUN:
#   PYTHONPATH=. uv run app/ingestion/ingestion_v1_custom.py
#
# What LangChain's PGVector creates (hidden from you):
#   langchain_pg_collection  — one row per collection
#   langchain_pg_embedding   — embeddings + cmetadata (jsonb blob)
#
# What WE create instead:
#   hr_chunks                — explicit, typed columns you define

import os
import hashlib
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.db import get_embeddings   # reuse the OpenAI embeddings setup

load_dotenv()
PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")

# ── Your schema ───────────────────────────────────────────────────────────────
# Change table name, add columns (category, language, author…), pick index type.
DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS hr_chunks (
    id            TEXT PRIMARY KEY,        -- deterministic MD5 hash → idempotent re-ingestion
    source        TEXT        NOT NULL,    -- file path / URL
    page          INTEGER,
    chunk_index   INTEGER,
    content       TEXT        NOT NULL,    -- raw chunk text
    embedding     vector(1536),            -- must match EMBEDDING_MODEL dimensions
    last_updated  TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- HNSW: fast approximate nearest-neighbour (good for production)
-- Alternative: IVFFlat — smaller index, needs VACUUM ANALYZE after inserts
CREATE INDEX IF NOT EXISTS hr_chunks_hnsw_idx
    ON hr_chunks USING hnsw (embedding vector_cosine_ops);
"""


def create_schema(conn):
    conn.execute(DDL)
    conn.commit()
    print("Schema ready.")


def ingest_pdf(file_path: str):
    print("Ingestion started")

    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Pages: {len(docs)}")

    # 2. Metadata enrichment
    for doc in docs:
        doc.metadata.update({
            "source": file_path,
            "page": doc.metadata.get("page"),
            "last_updated": str(os.path.getmtime(file_path)),
        })

    # 3. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    # 4. Generate embeddings in one batch call (more efficient than per-chunk)
    embeddings_model = get_embeddings()
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings_model.embed_documents(texts)  # list[list[float]]

    # 5. Insert directly into YOUR table
    with psycopg.connect(PG_CONNECTION) as conn:
        register_vector(conn)      # teaches psycopg how to serialize Python list → pgvector
        create_schema(conn)

        # ON CONFLICT DO NOTHING = safe to re-run without duplicates
        insert_sql = """
            INSERT INTO hr_chunks
                (id, source, page, chunk_index, content, embedding, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """

        rows = [
            (
                hashlib.md5(
                    f"{chunk.metadata['source']}-{chunk.metadata.get('page')}-{i}".encode()
                ).hexdigest(),
                chunk.metadata["source"],
                chunk.metadata.get("page"),
                i,
                chunk.page_content,
                vector,
                chunk.metadata.get("last_updated"),
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        conn.executemany(insert_sql, rows)
        conn.commit()
        print(f"Inserted {len(rows)} chunks into hr_chunks")

    print("======= Ingestion Completed Successfully! =======")


if __name__ == "__main__":
    ingest_pdf("data/HR_Support_Desk_KnowledgeBase.pdf")
