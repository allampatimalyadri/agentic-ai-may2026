from fastapi import FastAPI

from src.api.v1.routes.query import router as query_router

# Create the FastAPI app instance.
# The title appears in the auto-generated docs at http://localhost:8000/docs
app = FastAPI(title="Multimodal RAG API")


@app.get("/")
def read_root():
    """GET / — Basic health indicator."""
    return {"message": "Multimodal RAG API is running."}


@app.get("/health")
def health_check():
    """GET /health — Used by load balancers and monitoring tools."""
    return {"status": "ok"}


# Register the query router under /api/v1.
# All routes defined in query.py are accessible at /api/v1/<route>.
app.include_router(query_router, prefix="/api/v1")
