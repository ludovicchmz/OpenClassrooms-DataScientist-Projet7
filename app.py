from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import os
import json  

app = FastAPI()

# 1. Charger le modèle MLflow 
MODEL_URI = "models:/CreditScoringModel/latest"
model = mlflow.pyfunc.load_model(MODEL_URI)

# 2. Charger le seuil optimal 
try:
    with open("seuil_optimal.json", "r") as f:
        seuil_optimal = json.load(f)["seuil"]
except FileNotFoundError:
    raise RuntimeError("Fichier seuil_optimal.json introuvable. Veuillez le générer depuis le notebook.")

# 3. Définir le schéma des données d'entrée
from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    inputs: List[dict] 

# 4. Route /predict
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convertir les inputs en DataFrame
        inputs = pd.DataFrame(data.inputs)

        # Faire la prédiction
        prediction = model.predict(inputs)

        # Appliquer le seuil pour retourner la classe
        classe = ["refusé" if p < seuil_optimal else "accepté" for p in prediction]

        return {
            "probabilité": prediction.tolist(),
            "classe": classe,
            "seuil_utilisé": seuil_optimal 
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 5. Route /health
@app.get("/health")
def health():
    return {"status": "OK"}
