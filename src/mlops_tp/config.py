from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[2]

# Dossiers importants
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "src" / "mlops_tp" / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

# Fichiers
DATASET_PATH = DATA_DIR / "Customer-Churn-Records.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
RUN_INFO_PATH = ARTIFACTS_DIR / "run_info.json"