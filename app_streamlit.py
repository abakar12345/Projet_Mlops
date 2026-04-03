import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import os
import streamlit.components.v1 as components
from pathlib import Path

from src.mlops_tp.config import (
    DATASET_PATH,
    ARTIFACTS_DIR,
    METRICS_PATH,
    REPORTS_DIR
)

# ── CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard — Prédiction de churn",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL     = os.getenv("API_URL", "http://churn-api:8000")
DATA_PATH   = DATASET_PATH
REPORT_PATH = REPORTS_DIR / "eda_report.html"

# ── STYLE GLOBAL ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #f5f6fa;
    color: #1a1d2e;
}

section[data-testid="stSidebar"] {
    background: #1a2744;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #c8d4f0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #243158 !important;
}

h1 { font-size: 26px !important; font-weight: 700 !important; color: #1a1d2e !important; letter-spacing: -0.5px; }
h2 { font-size: 17px !important; font-weight: 500 !important; color: #5a6080 !important; }
h3 { font-size: 14px !important; font-weight: 500 !important; color: #7a80a0 !important; }

.kpi-card {
    background: white;
    border: 1px solid #e8eaf2;
    border-top: 3px solid #1a3a8f;
    border-radius: 10px;
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(26,58,143,0.06);
}
.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #1a3a8f;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-label {
    font-size: 11px;
    color: #9095b0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 6px;
}

div.stButton > button {
    background: #1a3a8f;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-size: 14px;
    font-weight: 600;
    width: 100%;
    letter-spacing: 0.5px;
    transition: background 0.2s;
}
div.stButton > button:hover { background: #142d72; }

.result-churn {
    background: #fff5f5;
    border: 1px solid #fca5a5;
    border-left: 4px solid #dc2626;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 20px;
}
.result-safe {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 4px solid #16a34a;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 20px;
}
.result-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 6px;
    color: #1a1d2e;
}
.result-sub {
    font-size: 12px;
    color: #7a80a0;
    font-family: 'JetBrains Mono', monospace;
}

.stDataFrame { border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }

hr { border-color: #e8eaf2 !important; }

div[role="radiogroup"] label {
    padding: 8px 14px;
    border-radius: 8px;
    margin-bottom: 3px;
    color: #a0b0d0 !important;
    transition: background 0.15s;
}
div[role="radiogroup"] label:hover { background: #243158; }
</style>
""", unsafe_allow_html=True)


# ── DONNÉES ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    return df

@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"runs": [{"metrics": {"accuracy": 0, "f1_score": 0, "roc_auc": 0},
                          "timestamp": "N/A", "hyperparameters": {}}]}

df      = load_data()
metrics = load_metrics()


# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 ChurnGuard")
    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio("Navigation", [
        "Vue générale",
        "Métriques du modèle",
        "Prédiction live",
        "Rapport EDA"
    ], label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#7a90c0; line-height:1.8;'>
    Modèle : RandomForest<br>
    Dataset : 10 000 clients<br>
    Tâche : Classification binaire
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 1 — VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════
if page == "Vue générale":
    st.markdown("# Vue générale")
    st.markdown("### Dataset — Customer Churn Records")
    st.markdown("<hr>", unsafe_allow_html=True)

    churn_rate = df['Exited'].mean() * 100
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        (f"{len(df):,}", "Clients"),
        (f"{df.shape[1]}", "Variables"),
        (f"{churn_rate:.1f}%", "Taux de churn"),
        (f"{df.isnull().sum().sum()}", "Valeurs manquantes"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.markdown("#### Aperçu des données")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    with col_right:
        churn_counts = df['Exited'].value_counts().reset_index()
        churn_counts.columns = ['Exited', 'Count']
        churn_counts['Label'] = churn_counts['Exited'].map({0: 'Reste', 1: 'Quitte'})
        fig = px.pie(
            churn_counts, values='Count', names='Label',
            color_discrete_sequence=['#1a3a8f', '#dc2626'],
            hole=0.55
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#5a6080',
            legend=dict(font=dict(color='#5a6080')),
            margin=dict(t=20, b=20)
        )
        fig.update_traces(textfont_color='white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Distributions clés")
    c1, c2 = st.columns(2)
    for col, var, color in [(c1, 'Age', '#1a3a8f'), (c2, 'Balance', '#dc2626')]:
        fig = px.histogram(df, x=var, nbins=40, color_discrete_sequence=[color])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#5a6080',
            xaxis=dict(gridcolor='#e8eaf2'),
            yaxis=dict(gridcolor='#e8eaf2'),
            margin=dict(t=10, b=10),
            showlegend=False
        )
        col.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — MÉTRIQUES DU MODÈLE
# ══════════════════════════════════════════════════════════
elif page == "Métriques du modèle":
    st.markdown("# Métriques du modèle")
    st.markdown("<hr>", unsafe_allow_html=True)

    last_run = metrics["runs"][-1]
    acc = last_run["metrics"]["accuracy"]
    f1  = last_run["metrics"]["f1_score"]
    auc = last_run["metrics"]["roc_auc"]

    col1, col2, col3 = st.columns(3)
    for col, val, label in [
        (col1, f"{acc:.4f}", "Accuracy"),
        (col2, f"{f1:.4f}", "F1-Score"),
        (col3, f"{auc:.4f}", "ROC-AUC"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_gauge, col_info = st.columns([1, 1])
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=auc,
            number={'font': {'color': '#1a3a8f', 'size': 48}},
            title={'text': "ROC-AUC", 'font': {'color': '#5a6080', 'size': 14}},
            gauge={
                'axis': {'range': [0, 1], 'tickcolor': '#9095b0'},
                'bar': {'color': '#1a3a8f'},
                'bgcolor': '#f5f6fa',
                'bordercolor': '#e8eaf2',
                'steps': [
                    {'range': [0, 0.7],    'color': '#fff0f0'},
                    {'range': [0.7, 0.85], 'color': '#fffbea'},
                    {'range': [0.85, 1],   'color': '#f0fdf4'}
                ],
                'threshold': {
                    'line': {'color': '#1a3a8f', 'width': 2},
                    'thickness': 0.75,
                    'value': auc
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#5a6080',
            height=280,
            margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("#### Détails du run")
        hp = last_run.get("hyperparameters", {})
        st.markdown(f"""
        <div style='background:white; border:1px solid #e8eaf2; border-radius:10px;
                    padding:24px; font-family: JetBrains Mono, monospace;
                    font-size:13px; line-height:2.2; color:#5a6080;
                    box-shadow: 0 2px 8px rgba(26,58,143,0.06);'>
        <b style='color:#1a3a8f'>Modèle</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; RandomForestClassifier<br>
        <b style='color:#1a3a8f'>n_estimators</b> &nbsp; {hp.get('n_estimators', 100)}<br>
        <b style='color:#1a3a8f'>max_depth</b> &nbsp;&nbsp;&nbsp;&nbsp; {hp.get('max_depth', 10)}<br>
        <b style='color:#1a3a8f'>class_weight</b> &nbsp; {hp.get('class_weight', 'balanced')}<br>
        <b style='color:#1a3a8f'>Timestamp</b> &nbsp;&nbsp;&nbsp;&nbsp; {last_run.get('timestamp', 'N/A')[:19]}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — PRÉDICTION LIVE
# ══════════════════════════════════════════════════════════
elif page == "Prédiction live":
    st.markdown("# Prédiction live")
    st.markdown("<hr>", unsafe_allow_html=True)

    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        status_color = "#16a34a" if health.get("model_loaded") else "#f59e0b"
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:20px;
                    background:white; border:1px solid #e8eaf2; border-radius:8px;
                    padding:12px 18px; box-shadow:0 1px 4px rgba(0,0,0,0.05);'>
            <div style='width:9px; height:9px; border-radius:50%; background:{status_color};'></div>
            <span style='font-size:13px; color:#5a6080;'>
                API connectée — modèle chargé : {health.get("model_loaded")}
            </span>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.error("API hors ligne")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Profil client**")
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age          = st.slider("Âge", 18, 92, 38)
        tenure       = st.slider("Ancienneté (années)", 0, 10, 4)
        num_products = st.selectbox("Nombre de produits", [1, 2, 3, 4])

    with col2:
        st.markdown("**Finances**")
        balance      = st.number_input("Solde (€)", 0.0, 300000.0, 65000.0, step=1000.0)
        salary       = st.number_input("Salaire estimé (€)", 0.0, 200000.0, 55000.0, step=1000.0)
        satisfaction = st.slider("Score de satisfaction", 1, 5, 3)
        points       = st.number_input("Points fidélité", 0, 1000, 420)

    with col3:
        st.markdown("**Informations**")
        gender      = st.selectbox("Genre", ["Male", "Female"])
        geography   = st.selectbox("Pays", ["France", "Germany", "Spain"])
        card_type   = st.selectbox("Type de carte", ["GOLD", "SILVER", "PLATINUM", "DIAMOND"])
        has_cr_card = st.selectbox("Carte de crédit", [1, 0], format_func=lambda x: "Oui" if x else "Non")
        is_active   = st.selectbox("Membre actif", [1, 0], format_func=lambda x: "Oui" if x else "Non")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Lancer la prédiction", type="primary")

    if predict_btn:
        payload = {"features": {
            "CreditScore":        credit_score,
            "Gender":             gender,
            "Age":                age,
            "Tenure":             tenure,
            "Balance":            float(balance),
            "NumOfProducts":      num_products,
            "HasCrCard":          has_cr_card,
            "IsActiveMember":     is_active,
            "EstimatedSalary":    float(salary),
            "Satisfaction Score": satisfaction,
            "Point Earned":       int(points),
            "Geography":          geography,
            "Card Type":          card_type
        }}

        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                data = response.json()
            except Exception as e:
                st.error(f"Erreur réseau : {e}")
                st.stop()

        if response.status_code != 200:
            st.error(f"Erreur API : {data.get('detail', 'Inconnue')}")
            st.stop()

        is_churn    = data["prediction"] == "yes"
        proba_churn = data["proba"]["yes"] * 100
        proba_reste = data["proba"]["no"]  * 100
        css_class   = "result-churn" if is_churn else "result-safe"
        icon        = "⚠️" if is_churn else "✅"
        verdict     = "Ce client risque de quitter la banque" if is_churn else "Ce client va probablement rester"
        color       = "#dc2626" if is_churn else "#16a34a"

        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-title">{icon} {verdict}</div>
            <div class="result-sub">
                Probabilité de churn : <b style='color:{color}'>{proba_churn:.1f}%</b>
                &nbsp;·&nbsp; Latence : {data['latency_ms']} ms
                &nbsp;·&nbsp; Modèle : {data['model_version']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_churn,
            number={'suffix': '%', 'font': {'color': color, 'size': 44}},
            title={'text': "Probabilité de churn", 'font': {'color': '#5a6080', 'size': 13}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#9095b0'},
                'bar': {'color': color},
                'bgcolor': '#f5f6fa',
                'bordercolor': '#e8eaf2',
                'steps': [
                    {'range': [0,  40],  'color': '#f0fdf4'},
                    {'range': [40, 70],  'color': '#fffbea'},
                    {'range': [70, 100], 'color': '#fff5f5'}
                ],
                'threshold': {
                    'line': {'color': '#1a1d2e', 'width': 2},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#5a6080',
            height=260,
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — RAPPORT EDA
# ══════════════════════════════════════════════════════════
elif page == "Rapport EDA":
    st.markdown("# Analyse EDA")
    st.markdown("<hr>", unsafe_allow_html=True)

    if REPORT_PATH.exists():
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=900, scrolling=True)
    else:
        st.warning("Rapport non trouvé. Lance d'abord `python -m src.mlops_tp.eda_profiling`")