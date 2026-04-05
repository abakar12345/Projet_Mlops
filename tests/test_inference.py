import pytest
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.mlops_tp.config import MODEL_PATH  


# pipeline + exemple de données

@pytest.fixture(scope="module")
def trained_pipeline():
    """Charge le pipeline sauvegardé via config.py"""
    assert MODEL_PATH.exists(), f"Le fichier modèle {MODEL_PATH} est introuvable"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_input(trained_pipeline):
    """Crée une fausse observation correctement formatée."""
    feature_columns = trained_pipeline["feature_columns"]
    # Une ligne de zéros — juste pour tester la forme
    X_sample = np.zeros((1, len(feature_columns)))
    return X_sample



# TEST 1 : predict renvoie bien une classe connue (0 ou 1)

def test_predict_returns_known_class(trained_pipeline, sample_input):
    model = trained_pipeline["model"]
    prediction = model.predict(sample_input)

    assert len(prediction) == 1, "predict doit retourner 1 valeur"
    assert prediction[0] in [0, 1], f"Classe inconnue : {prediction[0]} (attendu 0 ou 1)"



# TEST 2 : predict_proba renvoie des valeurs entre 0 et 1

def test_predict_proba_between_0_and_1(trained_pipeline, sample_input):
    model = trained_pipeline["model"]
    proba = model.predict_proba(sample_input)

    assert proba.shape == (1, 2), f"Shape attendu (1, 2), obtenu {proba.shape}"
    assert np.all(proba >= 0) and np.all(proba <= 1), "Les probabilités doivent être entre 0 et 1"



# TEST 3 : les probabilités somment à 1

def test_predict_proba_sums_to_1(trained_pipeline, sample_input):
    model = trained_pipeline["model"]
    proba = model.predict_proba(sample_input)
    total = proba.sum(axis=1)[0]

    assert abs(total - 1.0) < 1e-6, f"Les probas doivent sommer à 1, obtenu {total}"



# TEST 4 : predict est cohérent avec predict_proba

def test_predict_consistent_with_proba(trained_pipeline, sample_input):
    model = trained_pipeline["model"]
    prediction = model.predict(sample_input)[0]
    proba      = model.predict_proba(sample_input)[0]

    classe_proba = np.argmax(proba)
    assert prediction == classe_proba, f"predict ({prediction}) incohérent avec predict_proba ({classe_proba})"



# TEST 5 : le modèle gère un batch de plusieurs observations

def test_predict_batch(trained_pipeline):
    model           = trained_pipeline["model"]
    feature_columns = trained_pipeline["feature_columns"]
    X_batch         = np.zeros((10, len(feature_columns)))
    predictions     = model.predict(X_batch)

    assert len(predictions) == 10, f"Attendu 10 prédictions, obtenu {len(predictions)}"
    assert all(p in [0, 1] for p in predictions), "Certaines prédictions ne sont pas dans [0, 1]"