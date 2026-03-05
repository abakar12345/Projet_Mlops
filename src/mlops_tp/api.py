from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path



# ============================================================
# CHEMINS 
# ============================================================
from .config import ARTIFACTS_DIR, MODEL_PATH, METRICS_PATH, FEATURE_SCHEMA_PATH

# ============================================================
# CHARGEMENT DES ARTEFACTS (une seule fois au démarrage)
# ============================================================
pipeline = joblib.load(ARTIFACTS_DIR / "model.joblib")

with open(ARTIFACTS_DIR / "metrics.json") as f:
    metrics_data = json.load(f)

with open(ARTIFACTS_DIR / "feature_schema.json") as f:
    schema = json.load(f)

model           = pipeline["model"]
scaler          = pipeline["scaler"]
encoder_gender  = pipeline["encoder_gender"]
feature_columns = pipeline["feature_columns"]

# ============================================================
# APPLICATION FASTAPI
# ============================================================
app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction de churn bancaire",
    version="0.1.0"
)

# ============================================================
# SCHÉMA DE LA REQUÊTE
# ============================================================
class PredictRequest(BaseModel):
    features: dict

    @field_validator("features")
    def check_required_fields(cls, features):
        required = list(schema["features_originales"].keys())
        missing  = [f for f in required if f not in features]
        if missing:
            raise ValueError(f"Variables manquantes : {missing}")
        return features

# ============================================================
# GET /health
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

# ============================================================
# GET /metadata
# ============================================================
@app.get("/metadata")
def metadata():
    last_run = metrics_data["runs"][-1]
    return {
        "model_version":  "0.1.0",
        "model_type":     "RandomForestClassifier",
        "task":           "classification",
        "target":         "Exited",
        "classes":        {"0": "Reste", "1": "Quitte"},
        "features":       schema["features_originales"],
        "features_exclues": schema["features_exclues"],
        "n_features":     schema["n_features"],
        "last_training":  {
            "timestamp":  last_run["timestamp"],
            "metrics":    last_run["metrics"]
        }
    }

# ============================================================
# POST /predict
# ============================================================
@app.post("/predict")
def predict(request: PredictRequest):
    start = time.time()

    try:
        features = request.features

        # -- Validation des types --
        schema_orig = schema["features_originales"]
        for col, meta in schema_orig.items():
            if col not in features:
                raise HTTPException(
                    status_code=422,
                    detail=f"Variable manquante : '{col}'"
                )
            val      = features[col]
            expected = meta["type"]

            if expected == "int" and not isinstance(val, int):
                raise HTTPException(
                    status_code=422,
                    detail=f"'{col}' doit être un entier (int), reçu : {type(val).__name__}"
                )
            if expected == "float" and not isinstance(val, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"'{col}' doit être un nombre, reçu : {type(val).__name__}"
                )
            if expected == "str" and not isinstance(val, str):
                raise HTTPException(
                    status_code=422,
                    detail=f"'{col}' doit être une chaîne (str), reçu : {type(val).__name__}"
                )
            # Validation des catégories
            if "categories" in meta and val not in meta["categories"]:
                raise HTTPException(
                    status_code=422,
                    detail=f"'{col}' : valeur '{val}' inconnue. Attendu : {meta['categories']}"
                )

        # -- Construction du DataFrame --
        df_input = pd.DataFrame([features])

        # -- Encodage Gender --
        df_input['Gender'] = encoder_gender.transform(df_input['Gender'])

        # -- One-Hot Encoding Geography + Card Type --
        df_input = pd.get_dummies(df_input, columns=['Geography', 'Card Type'], drop_first=False)

        # -- Aligner les colonnes avec celles du train --
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)

        # -- Normalisation --
        X_scaled = scaler.transform(df_input)

        # -- Prédiction --
        prediction = int(model.predict(X_scaled)[0])
        proba      = model.predict_proba(X_scaled)[0]

        latency_ms = round((time.time() - start) * 1000, 2)

        return {
            "prediction":    "yes" if prediction == 1 else "no",
            "task":          "classification",
            "proba":         {"yes": round(float(proba[1]), 4),
                              "no":  round(float(proba[0]), 4)},
            "model_version": "0.1.0",
            "latency_ms":    latency_ms
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")