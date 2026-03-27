import os
from pathlib import Path

# Racine — configurable via variable d'environnement
BASE_DIR      = Path(os.getenv("APP_BASE_DIR", Path(__file__).resolve().parents[2]))

# Dossiers
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))
REPORTS_DIR   = BASE_DIR / "reports"

# Fichiers
DATASET_PATH        = DATA_DIR / "Customer-Churn-Records.csv"
MODEL_PATH          = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH        = ARTIFACTS_DIR / "metrics.json"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
RUN_INFO_PATH       = ARTIFACTS_DIR / "run_info.json"

# Variables d'environnement
API_PORT            = int(os.getenv("PORT", 8000))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")