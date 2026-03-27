 TP MLOps — Churn Prediction

Projet MLOps réalisé dans le cadre du cours IA et Génie Logiciel MLOps.  
L'objectif est d'industrialiser un modèle de machine learning : entraînement reproductible, exposition via une API REST, interface utilisateur, suivi des expérimentations avec MLflow et déploiement via Docker.



 Contexte et justification du dataset

Dataset : [Customer Churn Records](https://www.kaggle.com/datasets) — Kaggle  
Licence : Open Data (usage académique et personnel autorisé)

Le dataset modélise le comportement de clients d'une banque et cherche à prédire s'ils vont quitter l'établissement (churn). Ce problème est représentatif des enjeux réels en industrie financière : détecter les clients à risque avant qu'ils ne partent permet de mettre en place des actions de rétention ciblées.

| Propriété | Valeur |
|-----------|--------|
| Taille | 10 000 lignes × 18 colonnes |
| Tâche | Classification binaire (`Exited` : 0 = reste, 1 = quitte) |
| Variables | 13 features (numériques + catégorielles) |
| Déséquilibre | ~80% classe 0 / ~20% classe 1 → `class_weight='balanced'` |
| Valeurs manquantes | Aucune |

Défis anticipés : déséquilibre de classes important, variables catégorielles à encoder (Geography, Card Type, Gender), nécessité de normaliser les variables numériques pour le bon fonctionnement du modèle.


 Architecture du projet

TP_MLOps/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.streamlit
├── conftest.py
├── app_streamlit.py
├── data/
│   └── Customer-Churn-Records.csv
├── reports/
│   └── eda_report.html
├── src/
│   └── mlops_tp/
│       ├── __init__.py
│       ├── config.py          # chemins centralisés
│       ├── train.py           # entraînement + MLflow
│       ├── inference.py       # prétraitement + prédiction
│       ├── api.py             # API FastAPI
│       ├── eda_profiling.py   # rapport YData
│       └── artifacts/
│           ├── model.joblib
│           ├── metrics.json
│           ├── feature_schema.json
│           └── run_info.json
└── tests/
    ├── test_training.py
    ├── test_inference.py
    └── test_api.py


Installation et lancement

  Prérequis

- Python 3.11+
- Docker + Docker Compose

En local (sans Docker)

```bash
# Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Terminal 1 — MLflow
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2 — Entraîner le modèle
python -m src.mlops_tp.train

# Terminal 3 — API FastAPI
uvicorn mlops_tp.api:app --host 0.0.0.0 --port 8000 --app-dir src --reload

# Terminal 4 — Interface Streamlit
streamlit run app_streamlit.py
```

### Avec Docker Compose (recommandé)

```bash
docker compose up --build
```

Lance automatiquement les 3 services :

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| API FastAPI (Swagger) | http://localhost:8000/docs |
| Dashboard Streamlit | http://localhost:8501 |

Pour lancer un entraînement dans le conteneur :

```bash
docker compose exec churn-api python -m src.mlops_tp.train
```

---

## Modèle et pipeline

**Algorithme** : `RandomForestClassifier` (scikit-learn)

**Pipeline de prétraitement** :
1. Suppression des colonnes inutiles (`RowNumber`, `CustomerId`, `Surname`, `Complain`)
2. Encodage binaire de `Gender` via `LabelEncoder`
3. One-Hot Encoding de `Geography` et `Card Type` via `pd.get_dummies`
4. Normalisation via `StandardScaler`

**Split des données** :

| Ensemble | Proportion |
|----------|-----------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

La stratification (`stratify=y`) est appliquée pour préserver la proportion de churners dans chaque ensemble.

---

## Métriques de performance

Les métriques sont évaluées sur le jeu de validation et enregistrées dans `metrics.json` et MLflow.

| Métrique | Valeur (Run 1) | Justification |
|----------|---------------|---------------|
| Accuracy | 0.8427 | % global de bonnes prédictions |
| F1-Score | 0.6230 | Équilibre précision/rappel, adapté au déséquilibre |
| ROC-AUC  | 0.8683 | Capacité discriminante du modèle |

Le **F1-Score** est la métrique principale car le dataset est déséquilibré : l'accuracy seule peut être trompeuse (un modèle qui prédit toujours "reste" aurait 80% d'accuracy).

---

## API REST

L'API est exposée via FastAPI sur le port `8000`.

### Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Vérifie que l'API est opérationnelle |
| GET | `/metadata` | Informations sur le modèle et les features |
| GET | `/schema` | Schéma complet des variables attendues |
| POST | `/predict` | Prédit le churn pour un client |
| POST | `/predict/batch` | Prédit le churn pour un batch (max 1000) |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "CreditScore": 650,
      "Gender": "Male",
      "Age": 35,
      "Tenure": 5,
      "Balance": 75000.0,
      "NumOfProducts": 2,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 50000.0,
      "Satisfaction Score": 3,
      "Point Earned": 400,
      "Geography": "France",
      "Card Type": "GOLD"
    }
  }'
```

### Exemple de réponse

```json
{
  "prediction": "no",
  "task": "classification",
  "proba": { "yes": 0.12, "no": 0.88 },
  "model_version": "0.1.0",
  "latency_ms": 4.2
}
```

---

## Suivi des expérimentations avec MLflow

MLflow est intégré dans `train.py` pour tracer chaque run automatiquement.

**Ce qui est enregistré à chaque run :**

| Type | Éléments |
|------|----------|
| Paramètres | `n_estimators`, `max_depth`, `random_state`, `class_weight`, `test_size` |
| Métriques | `accuracy`, `f1_score`, `roc_auc` |
| Artefacts | Matrice de confusion (PNG), Courbe ROC (PNG), `feature_schema.json` |
| Modèle | `RandomForestClassifier` sérialisé |

**Comparaison des runs :**

| Run | n_estimators | max_depth | F1-Score | ROC-AUC |
|-----|-------------|-----------|----------|---------|
| Run 1 | 100 | 10 | 0.6230 | 0.8683 |
| Run 2 | 200 | 15 | — | — |
| Run 3 | 50  | 5  | — | — |

L'interface MLflow est accessible sur **http://localhost:5000**.

---

## Tests automatisés

```bash
# Lancer tous les tests
pytest tests/ -v

# Tester un fichier spécifique
pytest tests/test_inference.py -v
pytest tests/test_api.py -v
```

| Fichier | Ce qui est testé |
|---------|-----------------|
| `test_training.py` | Entraînement bout en bout, génération de `model.joblib` |
| `test_inference.py` | Prédictions (classe connue, probabilités entre 0 et 1, cohérence) |
| `test_api.py` | Codes HTTP 200/422, format des réponses JSON |

---

## Réponses aux questions du TP

### Questions préliminaires MLflow

**1. Qu'appelle-t-on une expérience dans MLflow ?**  
Une expérience est un espace de travail nommé qui regroupe plusieurs runs liés au même projet. Dans ce projet, l'expérience s'appelle `churn-prediction`.

**2. Qu'appelle-t-on un run ?**  
Un run est une exécution unique du script d'entraînement. Chaque run enregistre ses propres paramètres, métriques et artefacts, ce qui permet de comparer plusieurs essais.

**3. Quelle différence entre un paramètre, une métrique et un artefact ?**  
Un **paramètre** est une valeur de configuration fixée avant l'entraînement (ex : `n_estimators=100`). Une **métrique** est un résultat numérique mesuré après l'entraînement (ex : `f1_score=0.62`). Un **artefact** est un fichier produit pendant le run (ex : matrice de confusion en PNG).

**4. Exemples dans ce projet :**  
- Paramètres : `n_estimators`, `max_depth`, `class_weight`  
- Métriques : `accuracy`, `f1_score`, `roc_auc`  
- Artefacts : `confusion_matrix.png`, `roc_curve.png`

**5. Adresse de l'interface MLflow :** http://localhost:5000

**6. Interface avant le premier run :** L'expérience `churn-prediction` apparaît mais la liste des runs est vide — aucune donnée n'a encore été enregistrée.

**18. Artefact choisi :** Matrice de confusion (`confusion_matrix.png`)  
**19. Pourquoi ?** Elle visualise les vrais positifs, faux positifs, vrais négatifs et faux négatifs — indispensable pour évaluer un modèle sur données déséquilibrées.  
**20. Quand ?** Après l'évaluation sur le jeu de validation, dans le bloc `with mlflow.start_run()`.

---

## Technologies utilisées

| Technologie | Rôle |
|-------------|------|
| Python 3.11 | Langage principal |
| scikit-learn | Modèle et prétraitement |
| FastAPI | API REST |
| Streamlit | Dashboard interactif |
| MLflow | Suivi des expérimentations |
| Docker / Docker Compose | Conteneurisation |
| pytest | Tests automatisés |
| pandas / numpy | Manipulation des données |
| matplotlib / seaborn | Visualisation |
