# Custom schema retrieval — queries hr_chunks directly with raw SQL.
# Paired with: app/ingestion/ingestion_v1_custom.py
#
# HOW TO RUN:
#   PYTHONPATH=. uv run app/retrieval/retrieval_custom.py
#
# Or from a Python REPL / main.py:
#   from app.retrieval.retrieval_custom import retrieve
#   results = retrieve("What is the leave policy?", k=4)
#   for row in results:
#       print(row["content"])
#       print(row["similarity"])

import os
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from app.core.db import get_embeddings   # OpenAI embeddings — same model used at ingest time

load_dotenv()
PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")

# ── pgvector distance operators ───────────────────────────────────────────────
# <=>   cosine distance       (most common for text; use vector_cosine_ops index)
# <->   L2 / Euclidean        (use vector_l2_ops index)
# <#>   negative inner product (use vector_ip_ops index)
#
# similarity = 1 - cosine_distance  (1.0 = identical, 0.0 = orthogonal)

SEARCH_SQL = """
    SELECT
        id,
        source,
        page,
        chunk_index,
        content,
        1 - (embedding <=> %s::vector) AS similarity
    FROM hr_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
"""


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Return the top-k most similar chunks for a given query."""

    # Embed the query (single string → embed_query, NOT embed_documents)
    query_vector = get_embeddings().embed_query(query)

    with psycopg.connect(PG_CONNECTION) as conn:
        register_vector(conn)
        rows = conn.execute(SEARCH_SQL, (query_vector, query_vector, k)).fetchall()
        cols = ["id", "source", "page", "chunk_index", "content", "similarity"]
        return [dict(zip(cols, row)) for row in rows]


if __name__ == "__main__":
    query = "What is the leave policy for employees?"
    results = retrieve(query, k=4)

    print(f"\nTop {len(results)} results for: '{query}'\n{'=' * 60}")
    for i, row in enumerate(results, 1):
        print(f"\n[{i}] similarity={row['similarity']:.4f} | source={row['source']} | page={row['page']}")
        print(row["content"][:400])
