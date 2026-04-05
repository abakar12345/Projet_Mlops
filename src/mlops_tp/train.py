import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
from datetime import datetime
import joblib
import mlflow
import mlflow.sklearn

from .config import (
    DATASET_PATH,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_SCHEMA_PATH,
    RUN_INFO_PATH,
    ARTIFACTS_DIR
)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


N_ESTIMATORS = 100
MAX_DEPTH    = 10
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
TEST_SIZE    = 0.30   


df = pd.read_csv(DATASET_PATH)
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

print(df.info())
print("\n--- Card type Distribution ---")
print(df['Card Type'].value_counts())

X = df.drop(columns=['Exited', 'Complain'])
y = df['Exited']


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"Train      : {X_train.shape} — {len(X_train)/len(df)*100:.1f}%")
print(f"Validation : {X_val.shape} — {len(X_val)/len(df)*100:.1f}%")
print(f"Test       : {X_test.shape} — {len(X_test)/len(df)*100:.1f}%")


run_info = {
    "dataset": "Customer-Churn-Records.csv",
    "shape": {"lignes": df.shape[0], "colonnes": df.shape[1]},
    "cible": "Exited",
    "split": {
        "train":      f"{len(X_train)/len(df)*100:.1f}%",
        "validation": f"{len(X_val)/len(df)*100:.1f}%",
        "test":       f"{len(X_test)/len(df)*100:.1f}%"
    },
    "random_state": RANDOM_STATE
}
with open(RUN_INFO_PATH, "w") as f:
    json.dump(run_info, f, indent=4)


le = LabelEncoder()
X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_val['Gender']   = le.transform(X_val['Gender'])
X_test['Gender']  = le.transform(X_test['Gender'])

X_train = pd.get_dummies(X_train, columns=['Geography', 'Card Type'], drop_first=False)
X_val   = pd.get_dummies(X_val,   columns=['Geography', 'Card Type'], drop_first=False)
X_test  = pd.get_dummies(X_test,  columns=['Geography', 'Card Type'], drop_first=False)

X_val  = X_val.reindex(columns=X_train.columns,  fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("Prétraitement terminé ")



from .config import MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  

mlflow.set_experiment("churn-prediction")       

with mlflow.start_run():

    
    mlflow.log_param("model_type",    "RandomForestClassifier")
    mlflow.log_param("n_estimators",  N_ESTIMATORS)
    mlflow.log_param("max_depth",     MAX_DEPTH)
    mlflow.log_param("random_state",  RANDOM_STATE)
    mlflow.log_param("class_weight",  CLASS_WEIGHT)
    mlflow.log_param("test_size",     TEST_SIZE)
    mlflow.log_param("n_features",    X_train_scaled.shape[1])

    
   

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT
    )
    rf.fit(X_train_scaled, y_train)


   

    y_pred_val  = rf.predict(X_val_scaled)
    y_proba_val = rf.predict_proba(X_val_scaled)[:, 1]

    accuracy = accuracy_score(y_val, y_pred_val)
    f1       = f1_score(y_val, y_pred_val)
    roc_auc  = roc_auc_score(y_val, y_proba_val)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    mlflow.log_metric("accuracy", round(accuracy, 4))
    mlflow.log_metric("f1_score", round(f1, 4))
    mlflow.log_metric("roc_auc",  round(roc_auc, 4))

    
    # matrice de confusion
    cm = confusion_matrix(y_val, y_pred_val)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Reste", "Quitte"],
        yticklabels=["Reste", "Quitte"]
    )
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    cm_path = str(ARTIFACTS_DIR / "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="figures")

  
    # courbe ROC
  
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_val, y_proba_val, ax=ax, name="RandomForest")
    ax.set_title("Courbe ROC")
    plt.tight_layout()
    roc_path = str(ARTIFACTS_DIR / "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path, artifact_path="figures")


    # feature_schema.json

    mlflow.log_artifact(str(FEATURE_SCHEMA_PATH), artifact_path="schema")

    
    # SAUVEGARDE MODÈLE via MLflow
    
    mlflow.sklearn.log_model(rf, "model")

    print("MLflow run enregistré ")


# SAUVEGARDE LOCALE 
pipeline = {
    "scaler":          scaler,
    "encoder_gender":  le,
    "model":           rf,
    "feature_columns": list(X_train.columns)
}
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

# SAUVEGARDE metrics.json 
metrics_history = {
    "runs": [
        {
            "model": "RandomForestClassifier",
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": {
                "n_estimators": N_ESTIMATORS,
                "max_depth":    MAX_DEPTH,
                "random_state": RANDOM_STATE,
                "class_weight": CLASS_WEIGHT
            },
            "metrics": {
                "accuracy": round(accuracy, 4),
                "f1_score": round(f1, 4),
                "roc_auc":  round(roc_auc, 4)
            },
            "dataset_split": "validation"
        }
    ]
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics_history, f, indent=4)


# SAUVEGARDE feature_schema.json 

feature_schema = {
    "features_originales": {
        "CreditScore":       {"type": "int",   "description": "Score de crédit du client"},
        "Gender":            {"type": "str",   "categories": ["Male", "Female"]},
        "Age":               {"type": "int",   "description": "Âge du client"},
        "Tenure":            {"type": "int",   "description": "Années en tant que client"},
        "Balance":           {"type": "float", "description": "Solde bancaire"},
        "NumOfProducts":     {"type": "int",   "description": "Nombre de produits souscrits"},
        "HasCrCard":         {"type": "int",   "categories": [0, 1]},
        "IsActiveMember":    {"type": "int",   "categories": [0, 1]},
        "EstimatedSalary":   {"type": "float", "description": "Salaire estimé"},
        "Satisfaction Score":{"type": "int",   "range": [1, 5]},
        "Point Earned":      {"type": "int",   "description": "Points de fidélité"},
        "Geography":         {"type": "str",   "categories": ["France", "Germany", "Spain"]},
        "Card Type":         {"type": "str",   "categories": ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]}
    },
    "features_apres_encodage": list(X_train.columns),
    "cible": {
        "Exited": {"type": "int", "categories": {0: "Reste", 1: "Quitte"}}
    },
    "features_exclues": ["Complain", "RowNumber", "CustomerId", "Surname"],
    "n_features": len(X_train.columns)
}
with open(FEATURE_SCHEMA_PATH, "w") as f:
    json.dump(feature_schema, f, indent=4, ensure_ascii=False)
print("feature_schema.json sauvegardé ")


print("\n=== Artefacts générés ===")
for f in ARTIFACTS_DIR.iterdir():
    print(f.name, "—", f.stat().st_size / 1024, "KB")