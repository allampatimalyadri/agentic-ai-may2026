We are building REST API for managing products. It is a CRUD App

1. Create Product [DONE]
   REST API Endpoint: localhost:8000/api/v1/products
   Http Method: POST
   Product data (Request body) required: Yes

2. List Products [DONE]
   REST API Endpoint: localhost:8000/api/v1/products
   Http Method: GET
   Product data (Request body) required: no

3. Product Details by Id [DONE]
   REST API Endpoint: localhost:8000/api/v1/products/1
   Http Method: GET
   Product data (Request body) required: no

4. Update Product by ID [DONE]
   REST API Endpoint: localhost:8000/api/v1/products/1
   Http Method: PUT / PATCH  
   Product data (Request body) required: yes

5. Delete Product by ID [TODO]
   REST API Endpoint: localhost:8000/api/v1/products/1
   Http Method: DELETE  
   Product data (Request body) required: no

To run

```
uv venv
.venv\Scripts\activate
uv sync
uv run uvicorn app.main:app --reload
```
