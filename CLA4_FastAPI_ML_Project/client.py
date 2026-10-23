import requests

BASE_URL = "http://127.0.0.1:8000"
print("GET /health:", requests.get(f"{BASE_URL}/health", timeout=10).status_code)
passenger = {
    "Pclass": 3, "Sex": "female", "Age": 22,
    "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"
}
r = requests.post(f"{BASE_URL}/predict", json=passenger, timeout=10)
print("POST /predict:", r.status_code, r.json())
