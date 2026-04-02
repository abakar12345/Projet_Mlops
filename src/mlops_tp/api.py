from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from fastapi.responses import HTMLResponse

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

@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }
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
# GET /predict — Formulaire HTML
# ============================================================
@app.get("/predict", response_class=HTMLResponse)
def predict_form():
    return """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Churn Prediction</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f6f9; display: flex; justify-content: center; padding: 40px 20px; }
        .container { background: white; padding: 36px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); width: 100%; max-width: 700px; }
        h1 { font-size: 22px; color: #1a1a2e; margin-bottom: 6px; }
        p.subtitle { color: #666; font-size: 13px; margin-bottom: 28px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .field { display: flex; flex-direction: column; gap: 5px; }
        label { font-size: 13px; font-weight: 600; color: #333; }
        input, select { padding: 9px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; width: 100%; }
        input:focus, select:focus { outline: none; border-color: #4a90e2; }
        button { margin-top: 24px; width: 100%; padding: 13px; background: #4a90e2; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; }
        button:hover { background: #357abd; }
        #result { margin-top: 24px; padding: 20px; border-radius: 8px; display: none; }
        .result-yes { background: #fdecea; border: 1px solid #e74c3c; }
        .result-no  { background: #eafaf1; border: 1px solid #2ecc71; }
        .result-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .result-yes .result-title { color: #e74c3c; }
        .result-no  .result-title { color: #27ae60; }
        .proba-bar { background: #eee; border-radius: 4px; height: 10px; margin: 6px 0 12px; }
        .proba-fill { height: 10px; border-radius: 4px; }
        .proba-fill-yes { background: #e74c3c; }
        .proba-fill-no  { background: #2ecc71; }
        .meta { font-size: 12px; color: #888; margin-top: 8px; }
        .loading { text-align: center; color: #888; margin-top: 20px; display: none; }
    </style>
</head>
<body>
<div class="container">
    <h1>Churn Prediction</h1>
    <p class="subtitle">Renseignez les informations du client pour prédire s'il va quitter la banque.</p>

    <form id="predictForm">
        <div class="grid">
            <div class="field">
                <label>Credit Score</label>
                <input type="number" name="CreditScore" value="650" min="300" max="850" required>
            </div>
            <div class="field">
                <label>Âge</label>
                <input type="number" name="Age" value="35" min="18" max="92" required>
            </div>
            <div class="field">
                <label>Genre</label>
                <select name="Gender">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            <div class="field">
                <label>Géographie</label>
                <select name="Geography">
                    <option value="France">France</option>
                    <option value="Germany">Germany</option>
                    <option value="Spain">Spain</option>
                </select>
            </div>
            <div class="field">
                <label>Ancienneté (années)</label>
                <input type="number" name="Tenure" value="5" min="0" max="10" required>
            </div>
            <div class="field">
                <label>Solde (Balance)</label>
                <input type="number" name="Balance" value="60000" step="0.01" required>
            </div>
            <div class="field">
                <label>Nombre de produits</label>
                <select name="NumOfProducts">
                    <option value="1">1</option>
                    <option value="2" selected>2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                </select>
            </div>
            <div class="field">
                <label>Salaire estimé</label>
                <input type="number" name="EstimatedSalary" value="50000" step="0.01" required>
            </div>
            <div class="field">
                <label>A une carte de crédit ?</label>
                <select name="HasCrCard">
                    <option value="1">Oui</option>
                    <option value="0">Non</option>
                </select>
            </div>
            <div class="field">
                <label>Membre actif ?</label>
                <select name="IsActiveMember">
                    <option value="1">Oui</option>
                    <option value="0">Non</option>
                </select>
            </div>
            <div class="field">
                <label>Score de satisfaction (1-5)</label>
                <input type="number" name="Satisfaction Score" value="3" min="1" max="5" required>
            </div>
            <div class="field">
                <label>Points gagnés</label>
                <input type="number" name="Point Earned" value="400" min="0" max="1000" required>
            </div>
            <div class="field">
                <label>Type de carte</label>
                <select name="Card Type">
                    <option value="GOLD">GOLD</option>
                    <option value="SILVER">SILVER</option>
                    <option value="PLATINUM">PLATINUM</option>
                    <option value="DIAMOND">DIAMOND</option>
                </select>
            </div>
        </div>

        <button type="submit">Prédire</button>
    </form>

    <div class="loading" id="loading">Analyse en cours...</div>

    <div id="result"></div>
</div>

<script>
document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const form = e.target;
    const loading = document.getElementById('loading');
    const result  = document.getElementById('result');

    loading.style.display = 'block';
    result.style.display  = 'none';

    const intFields   = ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Satisfaction Score', 'Point Earned'];
    const floatFields = ['Balance', 'EstimatedSalary'];

    const features = {};
    new FormData(form).forEach((val, key) => {
        if (intFields.includes(key))        features[key] = parseInt(val);
        else if (floatFields.includes(key)) features[key] = parseFloat(val);
        else                                features[key] = val;
    });

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });

        const data = await response.json();
        loading.style.display = 'none';

        if (!response.ok) {
            result.className = '';
            result.style.display = 'block';
            result.style.background = '#fff3cd';
            result.style.border = '1px solid #ffc107';
            result.innerHTML = '<strong>Erreur :</strong> ' + (data.detail || 'Erreur inconnue');
            return;
        }

        const isChurn   = data.prediction === 'yes';
        const probaChurn = Math.round(data.proba.yes * 100);
        const probaReste = Math.round(data.proba.no  * 100);

        result.className = isChurn ? 'result-yes' : 'result-no';
        result.style.display = 'block';
        result.innerHTML = `
            <div class="result-title">
                ${isChurn ? '⚠️ Ce client risque de quitter la banque' : '✅ Ce client va probablement rester'}
            </div>
            <div>Probabilité de churn : <strong>${probaChurn}%</strong></div>
            <div class="proba-bar"><div class="proba-fill proba-fill-yes" style="width:${probaChurn}%"></div></div>
            <div>Probabilité de rester : <strong>${probaReste}%</strong></div>
            <div class="proba-bar"><div class="proba-fill proba-fill-no" style="width:${probaReste}%"></div></div>
            <div class="meta">Latence : ${data.latency_ms} ms &nbsp;|&nbsp; Modèle : ${data.model_version}</div>
        `;
    } catch(err) {
        loading.style.display = 'none';
        result.style.display = 'block';
        result.innerHTML = '<strong>Erreur réseau</strong>';
    }
});
</script>
</body>
</html>
"""


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