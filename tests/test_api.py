import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlops_tp.api import app

client = TestClient(app)


# DONNÉES DE TEST

valid_payload = {
    "features": {
        "CreditScore": 600,
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
        "Satisfaction Score": 3,
        "Point Earned": 400,
        "Geography": "France",
        "Card Type": "GOLD"
    }
}


# TESTS /health

def test_health_status_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_response_structure():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "model_loaded" in data

def test_health_model_loaded():
    response = client.get("/health")
    assert response.json()["model_loaded"] is True


# TESTS /metadata

def test_metadata_status_200():
    response = client.get("/metadata")
    assert response.status_code == 200

def test_metadata_response_structure():
    response = client.get("/metadata")
    data = response.json()
    assert "model_version" in data
    assert "task" in data
    assert "features" in data

def test_metadata_task_is_classification():
    response = client.get("/metadata")
    assert response.json()["task"] == "classification"


# TESTS /predict - Succès (200)

def test_predict_status_200():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200

def test_predict_response_structure():
    response = client.post("/predict", json=valid_payload)
    data = response.json()
    assert "prediction" in data
    assert "task" in data
    assert "proba" in data
    assert "model_version" in data
    assert "latency_ms" in data

def test_predict_prediction_is_known_class():
    response = client.post("/predict", json=valid_payload)
    prediction = response.json()["prediction"]
    assert prediction in ["yes", "no"], \
        f"Classe inconnue : {prediction}"

def test_predict_proba_between_0_and_1():
    response = client.post("/predict", json=valid_payload)
    proba = response.json()["proba"]
    assert 0 <= proba["yes"] <= 1
    assert 0 <= proba["no"] <= 1

def test_predict_proba_sums_to_1():
    response = client.post("/predict", json=valid_payload)
    proba = response.json()["proba"]
    total = proba["yes"] + proba["no"]
    assert abs(total - 1.0) < 1e-4, \
        f"Les probas doivent sommer à 1, obtenu {total}"

def test_predict_latency_is_positive():
    response = client.post("/predict", json=valid_payload)
    assert response.json()["latency_ms"] > 0


# TESTS /predict - Erreurs (422)

def test_predict_missing_feature_returns_422():
    """Une variable manquante doit retourner 422."""
    payload = {"features": {"CreditScore": 600}}  # incomplet
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_wrong_type_returns_422():
    """Un mauvais type doit retourner 422."""
    payload = valid_payload.copy()
    payload["features"] = valid_payload["features"].copy()
    payload["features"]["CreditScore"] = "six-cents"  # str au lieu de int
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_category_returns_422():
    """Une catégorie inconnue doit retourner 422."""
    payload = valid_payload.copy()
    payload["features"] = valid_payload["features"].copy()
    payload["features"]["Geography"] = "Maroc"  # catégorie inconnue
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_empty_features_returns_422():
    """Un body vide doit retourner 422."""
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422