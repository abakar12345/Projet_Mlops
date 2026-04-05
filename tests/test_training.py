# test_train.py
import os
import json
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.mlops_tp.config import MODEL_PATH, METRICS_PATH, FEATURE_SCHEMA_PATH


# pipeline complet entraîné

@pytest.fixture(scope="module")
def trained_pipeline():
    """Charge et retourne le pipeline sauvegardé."""
    assert MODEL_PATH.exists(), f"Le fichier {MODEL_PATH} est introuvable"
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


# TEST 1 : Le fichier model.joblib existe

def test_model_file_exists():
    assert MODEL_PATH.exists(), f"{MODEL_PATH} introuvable dans artifacts/"

# ============================================================
# TEST 2 : Le fichier model.joblib contient les bonnes clés
# ============================================================
def test_model_keys(trained_pipeline):
    expected_keys = {"scaler", "encoder_gender", "model", "feature_columns"}
    missing_keys = expected_keys - trained_pipeline.keys()
    assert not missing_keys, f"Clés manquantes : {missing_keys}"


# TEST 3 : Le modèle est bien un RandomForestClassifier

def test_model_type(trained_pipeline):
    assert isinstance(trained_pipeline["model"], RandomForestClassifier), \
        "Le modèle n'est pas un RandomForestClassifier"


# TEST 4 : Le scaler est bien fitté (a des attributs mean_)

def test_scaler_is_fitted(trained_pipeline):
    scaler = trained_pipeline["scaler"]
    assert hasattr(scaler, "mean_"), \
        "Le scaler ne semble pas avoir été fitté (mean_ absent)"


# TEST 5 : feature_columns contient bien 18 features

def test_feature_columns_count(trained_pipeline):
    n = len(trained_pipeline["feature_columns"])
    assert n == 18, f"Attendu 18 features, obtenu {n}"


# TEST 6 : metrics.json existe et contient au moins 1 run

def test_metrics_file():
    assert METRICS_PATH.exists(), f"{METRICS_PATH} introuvable"
    with open(METRICS_PATH) as f:
        data = json.load(f)
    assert "runs" in data and len(data["runs"]) > 0, \
        "metrics.json ne contient aucun run"


# TEST 7 : feature_schema.json existe et est valide

def test_feature_schema_file():
    assert FEATURE_SCHEMA_PATH.exists(), f"{FEATURE_SCHEMA_PATH} introuvable"
    with open(FEATURE_SCHEMA_PATH) as f:
        schema = json.load(f)
    assert "features_originales" in schema, "Clé 'features_originales' manquante"
    assert "features_apres_encodage" in schema, "Clé 'features_apres_encodage' manquante"
    assert "cible" in schema, "Clé 'cible' manquante"