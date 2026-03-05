import os
import numpy as np 
import pandas as pd 
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datetime import datetime
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from .config import (
    DATASET_PATH,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_SCHEMA_PATH,
    RUN_INFO_PATH,
    ARTIFACTS_DIR
)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

# Nettoyage des colonnes inutiles
colonne_a_supprimer = ['RowNumber', 'CustomerId', 'Surname']
df.drop(columns=colonne_a_supprimer, inplace=True)

print(df.info())

print("\n--- Card type  Distribution ---")
print(df['Card Type'].value_counts())


# Définition de X et y
X = df.drop(columns=['Exited','Complain'])
y = df['Exited']

# --- SPLIT ---
# Étape 1 : Train 70% / Reste 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y
)

# Étape 2 : Reste 30% → Validation 15% / Test 15%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.50,   # 50% du reste = 15% du total
    random_state=42, 
    stratify=y_temp
)

# Vérification des proportions
print(f"Train      : {X_train.shape} — {len(X_train)/len(df)*100:.1f}%")
print(f"Validation : {X_val.shape} — {len(X_val)/len(df)*100:.1f}%")
print(f"Test       : {X_test.shape} — {len(X_test)/len(df)*100:.1f}%")

# Vérification de la stratification
print("\nProportions de la cible 'Exited' :")
print(f"  Original   : {y.value_counts(normalize=True).to_dict()}")
print(f"  Train      : {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Validation : {y_val.value_counts(normalize=True).to_dict()}")
print(f"  Test       : {y_test.value_counts(normalize=True).to_dict()}")

# --- TRAÇABILITÉ ---
run_info = {
    "dataset": "Customer-Churn-Records.csv",
    "shape": {
        "lignes": df.shape[0],
        "colonnes": df.shape[1]
    },
    "cible": "Exited",
    "split": {
        "train":      f"{len(X_train)/len(df)*100:.1f}%",
        "validation": f"{len(X_val)/len(df)*100:.1f}%",
        "test":       f"{len(X_test)/len(df)*100:.1f}%"
    },
    "random_state": 42
}

with open(RUN_INFO_PATH, "w") as f:
    json.dump(run_info, f, indent=4)



# --- ENCODAGE ---
# Gender : binaire → LabelEncoder (Male=1, Female=0)
le = LabelEncoder()
X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_val['Gender']   = le.transform(X_val['Gender'])
X_test['Gender']  = le.transform(X_test['Gender'])

# Geography (3 catégories) + Card Type (4 catégories) → One-Hot Encoding
X_train = pd.get_dummies(X_train, columns=['Geography', 'Card Type'], drop_first=False)
X_val   = pd.get_dummies(X_val,   columns=['Geography', 'Card Type'], drop_first=False)
X_test  = pd.get_dummies(X_test,  columns=['Geography', 'Card Type'], drop_first=False)

# Aligner les colonnes (sécurité si une catégorie manque dans val/test)
X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --- NORMALISATION ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("Prétraitement terminé ✓")
print(f"Shape X_train : {X_train_scaled.shape}")
print(f"Shape X_val   : {X_val_scaled.shape}")
print(f"Shape X_test  : {X_test_scaled.shape}")

# Aperçu des colonnes créées
print(f"\nColonnes après encodage ({len(X_train.columns)}) :")
print(list(X_train.columns))


# --- MODÈLE ---
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'   # <-- compense le déséquilibre 80/20
)

rf.fit(X_train_scaled, y_train)

# --- PRÉDICTIONS sur Validation ---
y_pred_val  = rf.predict(X_val_scaled)
y_proba_val = rf.predict_proba(X_val_scaled)[:, 1]  # pour ROC-AUC

# --- MÉTRIQUES ---
accuracy = accuracy_score(y_val, y_pred_val)
f1       = f1_score(y_val, y_pred_val)
roc_auc  = roc_auc_score(y_val, y_proba_val)

print(f"Accuracy  : {accuracy:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# --- SAUVEGARDE metrics.json ---
metrics = {
    "model": "RandomForestClassifier",
    "timestamp": datetime.now().isoformat(),
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "metrics": {
        "accuracy":  round(accuracy, 4),
        "f1_score":  round(f1, 4),
        "roc_auc":   round(roc_auc, 4)
    },
    "dataset_split": "validation"
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)


# ==========================================
# 1. SAUVEGARDE model.joblib
# ==========================================
# On sauvegarde un pipeline complet : prétraitement + modèle


# Reconstruire le pipeline proprement pour joblib
# (le scaler et le modèle déjà fittés suffisent ici)
pipeline = {
    "scaler": scaler,       # déjà fitté sur X_train
    "encoder_gender": le,   # déjà fitté sur X_train
    "model": rf,            # déjà fitté
    "feature_columns": list(X_train.columns)  # colonnes après encodage
}

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)


# ==========================================
# 2. SAUVEGARDE metrics.json (historique)
# ==========================================
metrics_history = {
    "runs": [
        {
            "model": "RandomForestClassifier",
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "class_weight": "balanced"
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


# ==========================================
# 3. SAUVEGARDE feature_schema.json
# ==========================================
feature_schema = {
    "features_originales": {
        "CreditScore":      {"type": "int",   "description": "Score de crédit du client"},
        "Gender":           {"type": "str",   "categories": ["Male", "Female"]},
        "Age":              {"type": "int",   "description": "Âge du client"},
        "Tenure":           {"type": "int",   "description": "Années en tant que client"},
        "Balance":          {"type": "float", "description": "Solde bancaire"},
        "NumOfProducts":    {"type": "int",   "description": "Nombre de produits souscrits"},
        "HasCrCard":        {"type": "int",   "categories": [0, 1]},
        "IsActiveMember":   {"type": "int",   "categories": [0, 1]},
        "EstimatedSalary":  {"type": "float", "description": "Salaire estimé"},
        "Satisfaction Score":{"type": "int",  "range": [1, 5]},
        "Point Earned":     {"type": "int",   "description": "Points de fidélité"},
        "Geography":        {"type": "str",   "categories": ["France", "Germany", "Spain"]},
        "Card Type":        {"type": "str",   "categories": ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]}
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
print("feature_schema.json sauvegardé ✓")

# --- VÉRIFICATION FINALE ---
print("\n=== Artefacts générés ===")
for f in ARTIFACTS_DIR.iterdir():
    print(f.name, "—", f.stat().st_size / 1024, "KB")