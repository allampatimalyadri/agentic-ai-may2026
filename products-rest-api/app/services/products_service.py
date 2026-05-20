def get_products():
    # you have to connect to db and fetch products here
    return [{
        "id": 1,
        "name": "iPhone 18 Pro",
        "price": 999.99,
        "category": "Electronics",
        "description": "The latest iPhone with advanced features and improved performance."
    }, {
        "id": 2,
        "name": "MacBook Pro 16",
        "price": 2499.99,
        "category": "Electronics",
        "description": "A powerful laptop designed for professionals with a stunning Retina display."
    }, {
        "id": 3,
        "name": "AirPods Pro",
        "price": 249.99,
        "category": "Electronics"
    }]


def add_product(product: dict):
    print("Received product:", product)
    # In a real application, you would save the product to a database here
    return {
        "message": "Product added successfully", 
        "product": product
    }


def get_product_by_id(product_id: int):
    print("In Service: Fetching product with ID:", product_id)
    # In a real application, you would fetch the product from a database here
    return {
        "id": product_id, 
        "name": "Sample Product", 
        "price": 99.99, 
        "category": "Sample Category"
    }


def update_product_by_id(product_id: int, product: dict):
    print("In Service: Updating product with ID:", product_id)
    print("In Service: Updateable product data:", product)
    # In a real application, you would update the product in a database here
    return {
        "message": "Product updated successfully", 
        "product_id": product_id, 
        "updated_product": product
    }