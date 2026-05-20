# custom data model for product -- schema for product
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.products_service import add_product, get_products, get_product_by_id, update_product_by_id

class Product(BaseModel):
    id: Optional[int] = None
    name: str
    price: float
    category: str
    description: Optional[str] = None


router = APIRouter(
    prefix="/api/v1/products",
    tags=["products"]
)


# localhost:8000/api/v1/products - GET
@router.get("/")
def read_products():
    return get_products()


# localhost:8000/api/v1/products - POST
@router.post("/")
def create_product(product: Product):
    print("Received product:", product)
    # In a real application, you would save the product to a database here
    return add_product(product)


# localhost:8000/api/v1/products/1 or 2  - GET
@router.get("/{product_id}")
def read_product(product_id: int):
    print("Fetching product with ID:", product_id) # from url parameter
    return get_product_by_id(product_id)


# localhost:8000/api/v1/products/1 or 2  - PUT
@router.put("/{product_id}")
def update_product(product_id: int, product: Product):
    print("Updating product with ID:", product_id) # from url parameter
    print("Updateable product data:", product) # from request body
    # In a real application, you would update the product in a database here
    return update_product_by_id(product_id, product)

# TODO:  Implement Delete Product Endpoint
# expected output for DELETE endpoint:
# {
#     "message": "Product deleted successfully",
#     "product_id": 1
# }