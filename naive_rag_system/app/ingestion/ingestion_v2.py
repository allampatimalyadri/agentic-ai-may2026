from dotenv import load_dotenv
import os
import hashlib
from datetime import datetime

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv()

PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "hr_support_desk"


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
        output_dimensionality=1536,
        task_type="RETRIEVAL_DOCUMENT",  # optimized for indexing
    )


def get_vector_store(embeddings):
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONNECTION,
        use_jsonb=True,
    )


def load_pdf(file_path: str) -> list[dict]:
    """Load PDF using pypdf directly and return LangChain Documents."""
    reader = PdfReader(file_path)
    docs = []
    last_updated = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue  # skip blank/scanned pages

        docs.append(Document(
            page_content=text,
            metadata={
                "source": file_path,
                "document_extension": "pdf",
                "page": page_num,
                "total_pages": len(reader.pages),
                "category": "hr_support_desk",
                "last_updated": last_updated,
            }
        ))

    return docs


def ingest_pdf(file_path: str):
    """Ingest a PDF file and store chunks in pgvector."""

    # 1. Load PDF
    docs = load_pdf(file_path)
    if not docs:
        print("No extractable text found. Is this a scanned PDF?")
        return
    print(f"Loaded {len(docs)} pages.")

    # 2. Chunk by tokens (accurate for Gemini context window)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # closest public tokenizer to Gemini
        chunk_size=512,
        chunk_overlap=64,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Deterministic IDs — prevents duplicate ingestion on re-run
    ids = [
        hashlib.md5(
            f"{chunk.metadata['source']}-{chunk.metadata['page']}-{i}".encode()
        ).hexdigest()
        for i, chunk in enumerate(chunks)
    ]

    # 4. Embed + Store
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    # vector_store.add_documents(chunks, ids=ids)

    
    # FIXME: I am running a  for loop to add documents with ids. but it should ideally work with batch add_documents. 
    for chunk, id in zip(chunks, ids):
        vector_store.add_documents([chunk], ids=[id])
    

    print(f"Ingestion complete — {len(chunks)} chunks stored in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf("data/HR_Support_Desk_KnowledgeBase.pdf")