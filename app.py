from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI()

# Charger le modèle MLflow (remplace par ton URI si différent)
MODEL_URI = "models:/CreditScoringModel/latest"
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.post("/predict")
def predict(data: dict):
    # Convertir les inputs en DataFrame (LightGBM attend ce format)
    inputs = pd.DataFrame(data["inputs"])
    # Faire la prédiction
    prediction = model.predict(inputs)
    return {"prediction": prediction.tolist()}

@app.get("/health")
def health():
    return {"status": "OK"}
