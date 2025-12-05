import pytest
import json
import pandas as pd
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_with_valid_data():
    # Charger un exemple valide depuis sample_input.json
    with open("test/sample_input.json", "r") as f:
        data = json.load(f)

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "probabilité" in response.json()
    assert "classe" in response.json()
    assert len(response.json()["probabilité"]) == len(data["inputs"])
