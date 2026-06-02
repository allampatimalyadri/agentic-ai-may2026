# Paired with ingestion_v1.py (OpenAI embeddings via app/core/db.py)
#
# HOW TO RUN:
# in Mac: 
#   PYTHONPATH=. uv run app/retrieval/retrieval_openai.py
# in Windows:
#   $env:PYTHONPATH="."; uv run app/retrieval/retrieval_openai.py

from app.core.db import get_vector_store

def retrieve(query: str, k: int = 4) -> list:
    """Return the top-k most similar document chunks for a given query."""
    vector_store = get_vector_store(collection_name="hr_support_desk")
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    query = "How to apply for leave?"
    docs = retrieve(query, k=4)

    print(f"\nTop {len(docs)} results for: '{query}'\n{'=' * 60}")
    for i, doc in enumerate(docs, 1):
        print(f"""\n[{i}] Source: {doc.metadata.get('source')} | 
              Page: {doc.metadata.get('page')}""")
        print(doc.page_content[:400])

