
import json
import numpy as np
import pandas as pd
import joblib

from .config import MODEL_PATH, FEATURE_SCHEMA_PATH


# ---------------------------------------------------------------------------
# Chargement du pipeline (lazy, mis en cache au premier appel)
# ---------------------------------------------------------------------------

_pipeline = None


def load_pipeline():
    """Charge le pipeline depuis MODEL_PATH (singleton)."""
    global _pipeline
    if _pipeline is None:
        assert MODEL_PATH.exists(), f"Modèle introuvable : {MODEL_PATH}"
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def load_feature_schema() -> dict:
    """Charge le schéma de features depuis FEATURE_SCHEMA_PATH."""
    assert FEATURE_SCHEMA_PATH.exists(), f"Schéma introuvable : {FEATURE_SCHEMA_PATH}"
    with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prétraitement - reproduit exactement la logique de train.py
# ---------------------------------------------------------------------------

def preprocess(data: pd.DataFrame) -> np.ndarray:
    """
    Applique le même prétraitement que dans train.py :
      1. Suppression des colonnes inutiles (si présentes)
      2. Encodage LabelEncoder sur Gender
      3. One-Hot Encoding sur Geography et Card Type
      4. Réindexation sur les colonnes attendues par le modèle
      5. Normalisation StandardScaler

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes (une ou plusieurs lignes).
        Colonnes attendues : celles de feature_schema.json / "features_originales".

    Returns
    -------
    np.ndarray — données prêtes à passer au modèle.
    """
    pipeline = load_pipeline()

    df = data.copy()

    # 1. Suppression des colonnes inutiles si l'utilisateur les a incluses
    cols_to_drop = [c for c in ['RowNumber', 'CustomerId', 'Surname', 'Exited', 'Complain'] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 2. Encodage Gender (LabelEncoder fitté à l'entraînement)
    le = pipeline["encoder_gender"]
    df['Gender'] = le.transform(df['Gender'])

    # 3. One-Hot Encoding Geography + Card Type
    df = pd.get_dummies(df, columns=['Geography', 'Card Type'], drop_first=False)

    # 4. Réindexation — garantit l'ordre et les colonnes identiques au train
    feature_columns = pipeline["feature_columns"]
    df = df.reindex(columns=feature_columns, fill_value=0)

    # 5. Normalisation
    scaler = pipeline["scaler"]
    X = scaler.transform(df)

    return X


# ---------------------------------------------------------------------------
# API d'inférence publique
# ---------------------------------------------------------------------------

def predict_single(input_data: dict) -> dict:
    """
    Prédit le churn pour un seul client.

    Parameters
    ----------
    input_data : dict
        Dictionnaire représentant un client (clés = features originales).

    Returns
    -------
    dict avec :
        - prediction  : int  (0 = reste, 1 = quitte)
        - probability : float (probabilité de churn, entre 0 et 1)
        - label       : str  ("Reste" ou "Quitte")
    """
    df = pd.DataFrame([input_data])
    X  = preprocess(df)

    model = load_pipeline()["model"]
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])  # proba classe 1 (churn)

    return {
        "prediction":  pred,
        "probability": round(proba, 4),
        "label":       "Quitte" if pred == 1 else "Reste",
    }


def predict_batch(input_data: list[dict]) -> list[dict]:
    """
    Prédit le churn pour un batch de clients.

    Parameters
    ----------
    input_data : list[dict]
        Liste de dictionnaires, chacun représentant un client.

    Returns
    -------
    list[dict] — même structure que predict_single, un élément par client.
    """
    if not input_data:
        return []

    df = pd.DataFrame(input_data)
    X  = preprocess(df)

    model       = load_pipeline()["model"]
    predictions = model.predict(X)
    probas      = model.predict_proba(X)[:, 1]

    return [
        {
            "prediction":  int(pred),
            "probability": round(float(proba), 4),
            "label":       "Quitte" if pred == 1 else "Reste",
        }
        for pred, proba in zip(predictions, probas)
    ]