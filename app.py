import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).parent

DATA_PATH = APP_DIR / "student_lifestyle_100k.csv"
RESULTS_PATH = APP_DIR / "model_results.csv"

MODEL_FILES = {
    "Random Forest (Best Model)": "random_forest_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Logistic Regression + PCA": "logistic_pca_model.pkl",
}

NUMERIC_OPTIONS = [
    "Age",
    "CGPA",
    "Sleep_Duration",
    "Study_Hours",
    "Social_Media_Hours",
    "Physical_Activity_Hours",
    "Stress_Level",
    "Lifestyle_Balance",
    "Productive_Hours",
    "Stress_Study",
]

COLOR_MAP = {
    False: "#2563EB",
    True: "#F97316",
    "False": "#2563EB",
    "True": "#F97316",
}


st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
    }

    .hero {
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%);
        border-radius: 22px;
        padding: 38px 36px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 28px rgba(15, 118, 110, 0.20);
    }

    .hero h1 {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: white !important;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        max-width: 760px;
        margin: 0;
        color: #F8FAFC !important;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0.8rem 0 1rem 0;
        color: #0F172A;
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .card h4 {
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.05rem;
        color: #0F172A;
    }

    .card p {
        color: #475569;
        font-size: 0.93rem;
        margin-bottom: 0;
        line-height: 1.55;
    }

    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .badge-best {
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #22C55E;
    }

    .result-box {
        border-radius: 18px;
        padding: 24px;
        margin-top: 16px;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    .healthy {
        background: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
    }

    .risk {
        background: #FFF7ED;
        border: 1px solid #F97316;
        color: #9A3412;
    }

    .disclaimer {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 8px;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .stButton > button {
        background-color: #0F766E;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.65rem 1.2rem;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #115E59;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["Physical_Activity_Hours"] = df["Physical_Activity"] / 60

    df["Lifestyle_Balance"] = (
        df["Study_Hours"]
        + df["Physical_Activity_Hours"]
        - df["Social_Media_Hours"]
    )

    df["Productive_Hours"] = (
        df["Study_Hours"]
        + df["Physical_Activity_Hours"]
    )

    df["Stress_Study"] = (
        df["Stress_Level"]
        * df["Study_Hours"]
    )

    return df


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)

    return pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Logistic Regression + PCA"
        ],
        "Accuracy": [0.618, 0.735, 0.618],
        "Precision": [0.162, 0.217, 0.162],
       
