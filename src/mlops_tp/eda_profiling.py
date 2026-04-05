import pandas as pd
from ydata_profiling import ProfileReport
from pathlib import Path

# CHEMINS PROJET

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "Customer-Churn-Records.csv"
REPORT_PATH = BASE_DIR / "reports" / "eda_report.html"

# Créer le dossier reports si absent
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

try:
    print("Chargement du dataset...")
    df = pd.read_csv(DATA_PATH)

    # Nettoyage 
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    print(" Génération du rapport YData...")

    profile = ProfileReport(
        df,
        title="Analyse EDA — Customer Churn",
        explorative=True,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "cramers": {"calculate": True},
        }
    )

    profile.to_file(REPORT_PATH)
    print(f"Rapport sauvegardé → {REPORT_PATH}")

except Exception as e:
    print(f" Erreur : {e}")