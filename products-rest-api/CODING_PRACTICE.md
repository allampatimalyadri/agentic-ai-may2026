# Python Naming and Casing Conventions

## 1. File Names

Use `snake_case`

```python
user_service.py
product_routes.py
database_config.py
```

Avoid:

```python
UserService.py
productRoutes.py
```

---

## 2. Variable Names

Use `snake_case`

```python
user_name = "Arun"
total_price = 500
is_active = True
```

---

## 3. Function Names

Use `snake_case`

```python
def get_user():
    pass


def calculate_total():
    pass
```

---

## 4. Class Names

Use `PascalCase`

```python
class UserService:
    pass


class ProductManager:
    pass
```

---

## 5. Constant Names

Use `UPPER_CASE`

```python
MAX_USERS = 100
API_KEY = "xyz"
```

---

## 6. Package Names

Keep package names short and lowercase

```python
services
routes
models
utils
```

Avoid:

```python
MyServices
ProductUtils
```

---

## 7. Private Methods and Variables

Prefix with `_`

```python
class UserService:
    def _validate_user(self):
        pass
```

---

## 8. Boolean Variables

Use meaningful names

```python
is_admin = True
has_access = False
can_edit = True
```

---

## 9. Environment Variables

Use `UPPER_CASE`

```env
DATABASE_URL=
SECRET_KEY=
PORT=8000
```

---

# Recommended Project Structure

```text
app/
│
├── main.py
├── routes/
├── services/
├── models/
├── schemas/
├── utils/
└── config/
```

---

# Quick Summary

| Item     | Convention | Example           |
| -------- | ---------- | ----------------- |
| File     | snake_case | `user_service.py` |
| Variable | snake_case | `user_name`       |
| Function | snake_case | `get_user()`      |
| Class    | PascalCase | `UserService`     |
| Constant | UPPER_CASE | `MAX_USERS`       |
| Package  | lowercase  | `services`        |

---

# Best Practice Tips

- Keep names meaningful and simple
- Avoid very short variable names
- Be consistent across the project
- Follow PEP 8 conventions
- Prefer readability over clever naming
