# HOW TO RUN:
#   PYTHONPATH=. uv run app/retrieval/retrieval.py
#
# Or from a Python REPL / main.py:
#   from app.retrieval.retrieval import retrieve
#   results = retrieve("What is the leave policy?", k=4)
#   for doc in results:
#       print(doc.page_content)
#       print(doc.metadata)

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv()

PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "hr_support_desk"


def get_embeddings():
    # task_type="RETRIEVAL_QUERY" is the counterpart to "RETRIEVAL_DOCUMENT" used at ingest time
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=GEMINI_API_KEY,
        output_dimensionality=1536,
        task_type="RETRIEVAL_QUERY",
    )


def get_vector_store(embeddings):
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONNECTION,
        use_jsonb=True,
    )


def retrieve(query: str, k: int = 4) -> list:
    """Return the top-k most similar document chunks for a given query."""
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    query = "What is the leave policy for employees?"
    docs = retrieve(query, k=4)

    print(f"\nTop {len(docs)} results for: '{query}'\n{'=' * 60}")
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(doc.page_content[:400])
