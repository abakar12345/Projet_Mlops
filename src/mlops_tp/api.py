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


# CHARGEMENT DES ARTEFACTS (une seule fois au démarrage)
import os

# Permet de surcharger le chemin du modèle via variable d'environnement
_model_path = os.getenv("MODEL_PATH", str(MODEL_PATH))

try:
    pipeline        = joblib.load(_model_path)
    model           = pipeline["model"]
    scaler          = pipeline["scaler"]
    encoder_gender  = pipeline["encoder_gender"]
    feature_columns = pipeline["feature_columns"]
except FileNotFoundError:
    pipeline        = None
    model           = None
    scaler          = None
    encoder_gender  = None
    feature_columns = []
    print(f"  Modèle introuvable : {_model_path}")

try:
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        metrics_data = json.load(f)
    with open(ARTIFACTS_DIR / "feature_schema.json") as f:
        schema = json.load(f)
except FileNotFoundError as e:
    metrics_data = {}
    schema       = {"features_originales": {}, "features_exclues": [], "n_features": 0}
    print(f"  Artefact manquant : {e}")

# ============================================================
# APPLICATION FASTAPI
# ============================================================
app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction de churn bancaire",
    version="0.1.0"
)

# ============================================================
# SCHÉMAS DE REQUÊTE
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


class BatchPredictRequest(BaseModel):
    """Liste de clients à scorer en une seule requête."""
    customers: list[dict]

    @field_validator("customers")
    def check_not_empty(cls, customers):
        if not customers:
            raise ValueError("La liste 'customers' ne peut pas être vide.")
        if len(customers) > 1000:
            raise ValueError("Batch limité à 1000 clients maximum.")
        return customers


# ============================================================
# FONCTION DE PRÉTRAITEMENT (partagée predict + batch)
# ============================================================
def _preprocess(features: dict) -> np.ndarray:
    """
    Reproduit exactement le prétraitement de train.py :
    encodage Gender → get_dummies → reindex → StandardScaler.
    """
    schema_orig = schema["features_originales"]

    for col, meta in schema_orig.items():
        if col not in features:
            raise HTTPException(status_code=422, detail=f"Variable manquante : '{col}'")
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
        if "categories" in meta and val not in meta["categories"]:
            raise HTTPException(
                status_code=422,
                detail=f"'{col}' : valeur '{val}' inconnue. Attendu : {meta['categories']}"
            )

    df_input = pd.DataFrame([features])
    df_input['Gender'] = encoder_gender.transform(df_input['Gender'])
    df_input = pd.get_dummies(df_input, columns=['Geography', 'Card Type'], drop_first=False)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(df_input)


def _preprocess_batch(customers: list[dict]) -> np.ndarray:
    """
    Prétraitement vectorisé pour un batch de clients.
    Valide chaque ligne individuellement puis transforme en bloc.
    """
    schema_orig = schema["features_originales"]
    rows = []

    for i, features in enumerate(customers):
        for col, meta in schema_orig.items():
            if col not in features:
                raise HTTPException(
                    status_code=422,
                    detail=f"Client [{i}] — variable manquante : '{col}'"
                )
            val      = features[col]
            expected = meta["type"]
            if expected == "int" and not isinstance(val, int):
                raise HTTPException(
                    status_code=422,
                    detail=f"Client [{i}] — '{col}' doit être int, reçu {type(val).__name__}"
                )
            if expected == "float" and not isinstance(val, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"Client [{i}] — '{col}' doit être float, reçu {type(val).__name__}"
                )
            if expected == "str" and not isinstance(val, str):
                raise HTTPException(
                    status_code=422,
                    detail=f"Client [{i}] — '{col}' doit être str, reçu {type(val).__name__}"
                )
            if "categories" in meta and val not in meta["categories"]:
                raise HTTPException(
                    status_code=422,
                    detail=f"Client [{i}] — '{col}': '{val}' inconnu. Attendu : {meta['categories']}"
                )
        rows.append(features)

    df = pd.DataFrame(rows)
    df['Gender'] = encoder_gender.transform(df['Gender'])
    df = pd.get_dummies(df, columns=['Geography', 'Card Type'], drop_first=False)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(df)


# ============================================================
# GET /health
# ============================================================
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "timestamp":    datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_path":   str(MODEL_PATH),      # aide au debug en prod
        "version":      "0.1.0"
    }


# ============================================================
# GET /metadata
# ============================================================
@app.get("/metadata")
def metadata():
    last_run = metrics_data["runs"][-1]
    return {
        "model_version":    "0.1.0",
        "model_type":       "RandomForestClassifier",
        "task":             "classification",
        "target":           "Exited",
        "classes":          {"0": "Reste", "1": "Quitte"},
        "features":         schema["features_originales"],
        "features_exclues": schema["features_exclues"],
        "n_features":       schema["n_features"],
        "last_training": {
            "timestamp": last_run["timestamp"],
            "metrics":   last_run["metrics"]
        }
    }


# ============================================================
# GET /schema  — AJOUT
# ============================================================
@app.get("/schema")
def get_schema():
    """Retourne le schéma complet des features (utile pour le frontend Streamlit)."""
    return schema


# ============================================================
# POST /predict  — inchangé dans son comportement
# ============================================================
@app.post("/predict")
def predict(request: PredictRequest):
    start = time.time()
    if pipeline is None: 
        raise HTTPException(status_code=503, detail="Modèle non chargé — lance train.py d'abord")
    
    try:
        X_scaled   = _preprocess(request.features)
        prediction = int(model.predict(X_scaled)[0])
        proba      = model.predict_proba(X_scaled)[0]
        latency_ms = round((time.time() - start) * 1000, 2)

        return {
            "prediction":    "yes" if prediction == 1 else "no",
            "task":          "classification",
            "proba":         {
                "yes": round(float(proba[1]), 4),
                "no":  round(float(proba[0]), 4)
            },
            "model_version": "0.1.0",
            "latency_ms":    latency_ms
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


# ============================================================
# POST /predict/batch  — AJOUT
# ============================================================
@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    """
    Prédit le churn pour un batch de clients (max 1000).

    Retourne une liste de prédictions dans le même ordre que l'entrée,
    plus un résumé global (nb churn détectés, latence totale).
    """
    start = time.time()
    if pipeline is None:  # ← AJOUTE CES 2 LIGNES
        raise HTTPException(status_code=503, detail="Modèle non chargé — lance train.py d'abord")
    
    try:
        X_scaled    = _preprocess_batch(request.customers)
        predictions = model.predict(X_scaled)
        probas      = model.predict_proba(X_scaled)
        latency_ms  = round((time.time() - start) * 1000, 2)

        results = [
            {
                "index":      i,
                "prediction": "yes" if int(pred) == 1 else "no",
                "proba": {
                    "yes": round(float(proba[1]), 4),
                    "no":  round(float(proba[0]), 4)
                }
            }
            for i, (pred, proba) in enumerate(zip(predictions, probas))
        ]

        return {
            "task":           "classification",
            "model_version":  "0.1.0",
            "count":          len(results),
            "churn_detected": int(sum(1 for r in results if r["prediction"] == "yes")),
            "latency_ms":     latency_ms,
            "predictions":    results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")