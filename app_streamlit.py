import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import streamlit.components.v1 as components
from pathlib import Path

from src.mlops_tp.config import (
    DATASET_PATH,
    ARTIFACTS_DIR,
    METRICS_PATH,
    REPORTS_DIR
)

# CONFIG
st.set_page_config(page_title="Churn Dashboard", layout="wide")

API_URL     = "http://churn-api:8000"
DATA_PATH   = DATASET_PATH
ARTIFACTS   = ARTIFACTS_DIR
REPORT_PATH = REPORTS_DIR / "eda_report.html"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    return df

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

df      = load_data()
metrics = load_metrics()

# SIDEBAR
st.sidebar.title("Churn Dashboard")
page = st.sidebar.radio("Navigation", [
    "Vue générale",
    "Analyse univariée",
    "Analyse bivariée",
    "Métriques du modèle",
    "Prédiction live",
    "Rapport YData"
])


# PAGE 1 : VUE GÉNÉRALE

if page == "Vue générale":
    st.title("Vue générale du dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de clients",  f"{len(df):,}")
    col2.metric("Variables",           df.shape[1])
    col3.metric("Taux de churn",       f"{df['Exited'].mean()*100:.1f}%")
    col4.metric("Valeurs manquantes",  df.isnull().sum().sum())

    st.subheader("Aperçu des données")
    st.dataframe(df.head(5), use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe(), use_container_width=True)

    churn_counts = df['Exited'].value_counts().reset_index()
    churn_counts.columns = ['Exited', 'Count']
    churn_counts['Label'] = churn_counts['Exited'].map({0: 'Reste', 1: 'Quitte'})
    fig = px.pie(churn_counts, values='Count', names='Label',
                 color_discrete_sequence=['#2ecc71', '#e74c3c'],
                 title="Répartition Churn vs Non-Churn")
    st.plotly_chart(fig, use_container_width=True)


# PAGE 2 : ANALYSE UNIVARIÉE

elif page == "Analyse univariée":
    st.title("Analyse univariée")

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    st.subheader("Variables numériques")
    col_selected = st.selectbox("Choisir une variable", num_cols)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=col_selected, nbins=40,
                           title=f"Distribution — {col_selected}",
                           color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, y=col_selected,
                     title=f"Boxplot — {col_selected}",
                     color_discrete_sequence=['#9b59b6'])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Variables catégorielles")
    cat_selected = st.selectbox("Choisir une variable catégorielle", cat_cols)
    fig = px.bar(df[cat_selected].value_counts().reset_index(),
                 x=cat_selected, y='count',
                 title=f"Fréquence — {cat_selected}",
                 color_discrete_sequence=['#e67e22'])
    st.plotly_chart(fig, use_container_width=True)


# PAGE 3 : ANALYSE BIVARIÉE

elif page == "Analyse bivariée":
    st.title("Analyse bivariée vs Churn")

    num_cols = [c for c in df.select_dtypes(include='number').columns if c != 'Exited']
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    st.subheader("Variable numérique vs Churn")
    num_var = st.selectbox("Variable numérique", num_cols)
    fig = px.box(df, x='Exited', y=num_var,
                 color='Exited',
                 color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                 labels={'Exited': 'Churn'},
                 title=f"{num_var} selon le statut de churn")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Variable catégorielle vs Churn")
    cat_var = st.selectbox("Variable catégorielle", cat_cols)
    fig = px.histogram(df, x=cat_var, color='Exited',
                       barmode='group',
                       color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                       title=f"{cat_var} vs Churn")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrice de corrélation")
    corr = df.select_dtypes(include='number').corr()
    fig  = px.imshow(corr, text_auto=True, aspect="auto",
                     color_continuous_scale='RdBu_r',
                     title="Corrélations (variables numériques)")
    st.plotly_chart(fig, use_container_width=True)


# PAGE 4 : MÉTRIQUES
elif page == "Métriques du modèle":
    st.title("Métriques du modèle")

    last_run = metrics["runs"][-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",  last_run["metrics"]["accuracy"])
    col2.metric("F1-Score",  last_run["metrics"]["f1_score"])
    col3.metric("ROC-AUC",   last_run["metrics"]["roc_auc"])

    st.subheader("Détails du run")
    st.json(last_run)

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = last_run["metrics"]["roc_auc"],
        title = {'text': "ROC-AUC"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar':  {'color': "#3498db"},
            'steps': [
                {'range': [0, 0.7],   'color': '#e74c3c'},
                {'range': [0.7, 0.85],'color': '#f39c12'},
                {'range': [0.85, 1],  'color': '#2ecc71'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)


# PAGE 5 : PRÉDICTION LIVE

elif page == "Prédiction live":
    st.title("Prédiction en live via l'API")

    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API en ligne - modèle chargé : {health['model_loaded']}")
    except:
        st.error("API hors ligne — lance `uvicorn mlops_tp.api:app` d'abord")
        st.stop()

    st.subheader("Renseigne les informations du client")

    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("CreditScore",       300, 850, 600)
        age          = st.slider("Age",                18,  92,  40)
        tenure       = st.slider("Tenure",             0,   10,  3)
        num_products = st.selectbox("NumOfProducts",   [1, 2, 3, 4])
    with col2:
        balance      = st.number_input("Balance",         0.0, 300000.0, 60000.0)
        salary       = st.number_input("EstimatedSalary", 0.0, 200000.0, 50000.0)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        points       = st.number_input("Point Earned", 0, 1000, 400)
    with col3:
        gender       = st.selectbox("Gender",        ["Male", "Female"])
        geography    = st.selectbox("Geography",     ["France", "Germany", "Spain"])
        card_type    = st.selectbox("Card Type",     ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])
        has_cr_card  = st.selectbox("HasCrCard",     [0, 1])
        is_active    = st.selectbox("IsActiveMember",[0, 1])

    if st.button("Prédire", type="primary"):
        payload = {"features": {         # ← faute de frappe corrigée (ayload → payload)
            "CreditScore":        credit_score,
            "Gender":             gender,
            "Age":                age,
            "Tenure":             tenure,
            "Balance":            balance,
            "NumOfProducts":      num_products,
            "HasCrCard":          has_cr_card,
            "IsActiveMember":     is_active,
            "EstimatedSalary":    salary,
            "Satisfaction Score": satisfaction,
            "Point Earned":       points,
            "Geography":          geography,
            "Card Type":          card_type
        }}

        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code != 200:
            st.error(f"Erreur API : {response.json()['detail']}")
            st.stop()
        response = response.json()

        st.divider()
        if response["prediction"] == "yes":
            st.error("Ce client risque de **quitter** la banque !")
        else:
            st.success("Ce client va probablement **rester**.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Prédiction",  response["prediction"].upper())
        col2.metric("Proba Churn", f"{response['proba']['yes']*100:.1f}%")
        col3.metric("Latence",     f"{response['latency_ms']} ms")

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = response['proba']['yes'] * 100,
            title = {'text': "Probabilité de Churn (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar':  {'color': "#e74c3c"},
                'steps': [
                    {'range': [0,  40],  'color': '#2ecc71'},
                    {'range': [40, 70],  'color': '#f39c12'},
                    {'range': [70, 100], 'color': '#e74c3c'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)


# PAGE 6 : RAPPORT YDATA
elif page == "Rapport YData":
    st.title("Rapport YData Profiling")

    if REPORT_PATH.exists():
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=900, scrolling=True)
    else:
        st.warning("Rapport non trouvé. Lance d'abord `python eda_profiling.py`")
        if st.button("Générer le rapport maintenant"):
            with st.spinner("Génération en cours..."):
                import subprocess
                subprocess.run(["python", "eda_profiling.py"])
                st.rerun()