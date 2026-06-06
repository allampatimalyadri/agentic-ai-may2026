# Load the file from data folder 
# extract the content 
# arrive at the right chunking strategy 
# chunk_size? chunk_overlap? 

# Load the embedding model and vectorize the chunks 
# Connect to postgres and activate pgvector extension
# store the vectorized data into vector db with original text 

from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.db import get_vector_store

load_dotenv()
PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")

def ingest_pdf(file_path): 
    print("Ingestion Started")

    #1. Load PDF 
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print("Pages : ", len(docs))

    # 2. Metadata enrichment
    for doc in docs: 
        doc.metadata.update({
            "source": file_path,
            "document_extension": "pdf",
            "page": doc.metadata.get("page"),
            "last_updated": os.path.getmtime(file_path)
        })

    # 3. Chunking 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # characters
        chunk_overlap=200 # characters
    )

    chunks = splitter.split_documents(docs)
    print("Total Chunks", len(chunks))

    # 4 and 5 
    # generate the embeddings store in vector db
    vector_store = get_vector_store(collection_name="hr_support_desk", pre_delete_collection=True)

    vector_store.add_documents(chunks)


    print("======Ingestion Completed Successfully!=======")

if __name__ == "__main__":
    ingest_pdf("data/HR_Support_Desk_KnowledgeBase.pdf")


# to execute: 
# in windows 
# $env:PYTHONPATH="."; uv run app/ingestion/ingestion.py

# in macOS/Linux
# PYTHONPATH=. uv run app/ingestion/ingestion.py
