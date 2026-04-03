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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fond général */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1e2230;
}
section[data-testid="stSidebar"] * {
    color: #c8cad4 !important;
}

/* Titres */
h1 { font-size: 28px !important; font-weight: 600 !important; color: #f0f2f8 !important; letter-spacing: -0.5px; }
h2 { font-size: 18px !important; font-weight: 500 !important; color: #a0a4b8 !important; }
h3 { font-size: 15px !important; font-weight: 500 !important; color: #8085a0 !important; }

/* Cards KPI */
.kpi-card {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.kpi-value {
    font-size: 32px;
    font-weight: 600;
    color: #4fc3f7;
    font-family: 'DM Mono', monospace;
}
.kpi-label {
    font-size: 12px;
    color: #606480;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Bouton predict */
div.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-size: 15px;
    font-weight: 600;
    width: 100%;
    letter-spacing: 0.3px;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }

/* Résultat churn */
.result-churn {
    background: #1a0a0a;
    border: 1px solid #c62828;
    border-left: 4px solid #ef5350;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 20px;
}
.result-safe {
    background: #071a0f;
    border: 1px solid #1b5e20;
    border-left: 4px solid #66bb6a;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 20px;
}
.result-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
}
.result-sub {
    font-size: 13px;
    color: #8085a0;
    font-family: 'DM Mono', monospace;
}

/* Inputs */
.stSlider > div, .stNumberInput, .stSelectbox {
    background: transparent !important;
}
input, select {
    background: #13161e !important;
    color: #e8eaf0 !important;
    border-color: #1e2230 !important;
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Séparateur */
hr { border-color: #1e2230 !important; }

/* Radio sidebar */
div[role="radiogroup"] label {
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 2px;
    transition: background 0.15s;
}
div[role="radiogroup"] label:hover { background: #1e2230; }

/* Success / error */
.stSuccess { background: #071a0f !important; border-color: #1b5e20 !important; }
.stError   { background: #1a0a0a !important; border-color: #c62828 !important; }
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
    <div style='font-size:11px; color:#404460; line-height:1.8;'>
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

    # KPIs
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
        st.dataframe(
            df.head(8),
            use_container_width=True,
            hide_index=True
        )
    with col_right:
        churn_counts = df['Exited'].value_counts().reset_index()
        churn_counts.columns = ['Exited', 'Count']
        churn_counts['Label'] = churn_counts['Exited'].map({0: 'Reste', 1: 'Quitte'})
        fig = px.pie(
            churn_counts, values='Count', names='Label',
            color_discrete_sequence=['#1565c0', '#ef5350'],
            hole=0.55
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a4b8',
            legend=dict(font=dict(color='#a0a4b8')),
            margin=dict(t=20, b=20)
        )
        fig.update_traces(textfont_color='white')
        st.plotly_chart(fig, use_container_width=True)

    # Distribution Age et Balance
    st.markdown("#### Distributions clés")
    c1, c2 = st.columns(2)
    for col, var, color in [(c1, 'Age', '#1565c0'), (c2, 'Balance', '#ef5350')]:
        fig = px.histogram(
            df, x=var, nbins=40,
            color_discrete_sequence=[color]
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a4b8',
            xaxis=dict(gridcolor='#1e2230'),
            yaxis=dict(gridcolor='#1e2230'),
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
    acc  = last_run["metrics"]["accuracy"]
    f1   = last_run["metrics"]["f1_score"]
    auc  = last_run["metrics"]["roc_auc"]

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

    # Gauge ROC-AUC
    col_gauge, col_info = st.columns([1, 1])
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=auc,
            number={'font': {'color': '#4fc3f7', 'size': 48}},
            title={'text': "ROC-AUC", 'font': {'color': '#a0a4b8', 'size': 14}},
            gauge={
                'axis': {'range': [0, 1], 'tickcolor': '#404460'},
                'bar': {'color': '#1565c0'},
                'bgcolor': '#13161e',
                'bordercolor': '#1e2230',
                'steps': [
                    {'range': [0, 0.7],    'color': '#1a0a0a'},
                    {'range': [0.7, 0.85], 'color': '#1a1200'},
                    {'range': [0.85, 1],   'color': '#071a0f'}
                ],
                'threshold': {
                    'line': {'color': '#4fc3f7', 'width': 2},
                    'thickness': 0.75,
                    'value': auc
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a4b8',
            height=280,
            margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("#### Détails du run")
        hp = last_run.get("hyperparameters", {})
        st.markdown(f"""
        <div style='background:#13161e; border:1px solid #1e2230; border-radius:10px; padding:20px; font-family: DM Mono, monospace; font-size:13px; line-height:2; color:#a0a4b8;'>
        <b style='color:#4fc3f7'>Modèle</b> &nbsp;&nbsp;&nbsp; RandomForestClassifier<br>
        <b style='color:#4fc3f7'>n_estimators</b> &nbsp; {hp.get('n_estimators', 100)}<br>
        <b style='color:#4fc3f7'>max_depth</b> &nbsp;&nbsp;&nbsp; {hp.get('max_depth', 10)}<br>
        <b style='color:#4fc3f7'>class_weight</b> &nbsp; {hp.get('class_weight', 'balanced')}<br>
        <b style='color:#4fc3f7'>Timestamp</b> &nbsp;&nbsp;&nbsp; {last_run.get('timestamp', 'N/A')[:19]}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — PRÉDICTION LIVE
# ══════════════════════════════════════════════════════════
elif page == "Prédiction live":
    st.markdown("# Prédiction live")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Statut API
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        status_color = "#66bb6a" if health.get("model_loaded") else "#ffa726"
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:20px;'>
            <div style='width:10px; height:10px; border-radius:50%; background:{status_color};'></div>
            <span style='font-size:13px; color:#a0a4b8;'>API connectée — modèle chargé : {health.get("model_loaded")}</span>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.error("API hors ligne")
        st.stop()

    # Formulaire
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Profil client**")
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age          = st.slider("Âge", 18, 92, 38)
        tenure       = st.slider("Ancienneté (années)", 0, 10, 4)
        num_products = st.selectbox("Nombre de produits", [1, 2, 3, 4])

    with col2:
        st.markdown("**Finances**")
        balance  = st.number_input("Solde (€)", 0.0, 300000.0, 65000.0, step=1000.0)
        salary   = st.number_input("Salaire estimé (€)", 0.0, 200000.0, 55000.0, step=1000.0)
        satisfaction = st.slider("Score de satisfaction", 1, 5, 3)
        points   = st.number_input("Points fidélité", 0, 1000, 420)

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
        color       = "#ef5350" if is_churn else "#66bb6a"

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

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_churn,
            number={'suffix': '%', 'font': {'color': color, 'size': 44}},
            title={'text': "Probabilité de churn", 'font': {'color': '#a0a4b8', 'size': 13}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#404460'},
                'bar': {'color': color},
                'bgcolor': '#13161e',
                'bordercolor': '#1e2230',
                'steps': [
                    {'range': [0,  40],  'color': '#071a0f'},
                    {'range': [40, 70],  'color': '#1a1200'},
                    {'range': [70, 100], 'color': '#1a0a0a'}
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 2},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a4b8',
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