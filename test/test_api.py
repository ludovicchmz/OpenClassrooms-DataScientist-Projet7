import requests
import json
import os

def test_api_prediction():
    with open("sample_input.json", "r") as f:
        data = json.load(f)

    response = requests.post("http://localhost:1234/invocations", json=data)

    assert response.status_code == 200
    assert "predictions" in response.json()
