# setup fastapi app
from fastapi import FastAPI
from app.routes.products_routes import router as products_router

app = FastAPI()

# localhost:8000/
@app.get("/")
def root():
    return "The app is running! Check localhost:8000/docs"


# localhost:8000/api/v1/hello
@app.get("/api/v1/hello")
def read_hello():
    return {"Hello": "World"}


# registering nested routes with the main app
app.include_router(products_router)

# to run this, 
# uv run uvicorn main:app --reload
