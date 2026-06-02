# Step 1:

Build the ingestion pipeline

uv add python-dotenv langchain "langchain[google-genai]" langchain-community langchain-postgres pypdf toktoken

# Step 2:

Build the retrieval pipeline

Todo for Arun:

1. Hierachical Agents with Decision Routing
2. Later explain runnable message history in langchain

# Indexing

CREATE INDEX ON langchain_pg_embedding
USING ivfflat ((embedding::vector(1536)) vector_cosine_ops)
WITH (lists = 100);

====

# 3 parameters for project success

1. Accuracy
2. Latency
3. Cost
