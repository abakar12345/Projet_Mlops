# Churn Prediction — MLOps

Projet réalisé dans le cadre du cours IA et Génie Logiciel MLOps.
L'objectif est d'industrialiser un modèle de prédiction de churn bancaire :
entraînement reproductible, API REST, interface utilisateur, suivi avec MLflow,
conteneurisation Docker et déploiement cloud avec CI/CD.

---

## Contexte

Le dataset utilisé est le **Customer Churn Records** (Kaggle, 10 000 clients).
La cible est la colonne `Exited` : un client quitte-t-il la banque (1) ou reste-t-il (0) ?

Le dataset est déséquilibré (~80% / 20%), ce qui justifie l'utilisation du
`class_weight='balanced'` et du F1-Score comme métrique principale plutôt que l'accuracy.

---

## Structure du projet
```
TP_MLOps/
├── data/                        # Dataset CSV
├── src/mlops_tp/
│   ├── config.py                # Chemins centralisés via variables d'env
│   ├── train.py                 # Entraînement + tracking MLflow
│   ├── api.py                   # API FastAPI
│   ├── inference.py             # Prétraitement et prédiction
│   ├── eda_profiling.py         # Rapport YData
│   └── artifacts/               # Modèle, métriques, schéma
├── tests/                       # Tests pytest
├── app_streamlit.py             # Dashboard interactif
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
└── .github/workflows/ci.yml     # Pipeline CI/CD
```

---

## Lancement

### Sans Docker
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — MLflow
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2 — Entraînement
PYTHONPATH=src python -m mlops_tp.train

# Terminal 3 — API
PYTHONPATH=src uvicorn mlops_tp.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 4 — Streamlit
PYTHONPATH=src streamlit run app_streamlit.py
```

### Avec Docker Compose
```bash
docker compose up --build
```

| Service | URL locale |
|---|---|
| MLflow | http://localhost:5000 |
| API (Swagger) | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

---

## Modèle

**Algorithme** : RandomForestClassifier (scikit-learn)

Pipeline de prétraitement :
1. Suppression des colonnes inutiles (`RowNumber`, `CustomerId`, `Surname`, `Complain`)
2. Encodage de `Gender` via LabelEncoder
3. One-Hot Encoding de `Geography` et `Card Type`
4. Normalisation via StandardScaler

Split : 70% train / 15% validation / 15% test (stratifié)

| Métrique | Valeur |
|---|---|
| Accuracy | 0.8427 |
| F1-Score | 0.6230 |
| ROC-AUC | 0.8683 |

---

## API REST

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Statut de l'API |
| GET | `/metadata` | Infos sur le modèle |
| GET | `/schema` | Schéma des features |
| POST | `/predict` | Prédiction pour un client |
| POST | `/predict/batch` | Prédiction en batch (max 1000) |

Exemple de requête :
```bash
curl -X POST https://churn-api-kynr.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "CreditScore": 650, "Gender": "Male", "Age": 35,
      "Tenure": 5, "Balance": 75000.0, "NumOfProducts": 2,
      "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000.0,
      "Satisfaction Score": 3, "Point Earned": 400,
      "Geography": "France", "Card Type": "GOLD"
    }
  }'
```

---

## Tests
```bash
pytest tests/ -v
```

| Fichier | Ce qui est couvert |
|---|---|
| `test_training.py` | Existence du modèle, clés du pipeline, type du modèle |
| `test_inference.py` | Prédictions, probabilités, cohérence |
| `test_api.py` | Codes HTTP 200/422, format des réponses |

---

## CI/CD

Le fichier `.github/workflows/ci.yml` déclenche automatiquement à chaque push sur `main` :

1. Installation des dépendances
2. Exécution des tests pytest
3. Build des images Docker

Si tous les tests passent, Render redéploie automatiquement l'API et le dashboard Streamlit.

**Services déployés :**
- API : https://churn-api-kynr.onrender.com
- Streamlit : https://projet-mlops-11yi.onrender.com

---

## Variables d'environnement

| Variable | Description | Défaut local |
|---|---|---|
| `PORT` | Port d'écoute | `8000` |
| `PYTHONPATH` | Chemin source | `src` |
| `MLFLOW_TRACKING_URI` | URL MLflow | `http://localhost:5000` |
| `API_URL` | URL de l'API | `http://localhost:8000` |
| `ARTIFACTS_DIR` | Dossier des artefacts | `src/mlops_tp/artifacts` |

Les secrets ne sont jamais versionnés — ils sont définis directement
dans les variables d'environnement de Render.

---

## Stack technique

Python 3.11 · scikit-learn · FastAPI · Streamlit · MLflow ·
Docker · GitHub Actions · Render · pytest · pandas · numpy
