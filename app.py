import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "student_lifestyle_100k.csv"
METRICS_PATH = APP_DIR / "model_metrics.json"

MODEL_FILES = {
    "Random Forest (Best Model)": "random_forest_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Logistic Regression + PCA": "logistic_pca_model.pkl",
}

# --------------------------------------------------------------------------
# Color palette — reused everywhere (CSS + Plotly) so the app feels unified
# --------------------------------------------------------------------------
PRIMARY = "#6C5CE7"      # indigo/violet
PRIMARY_DARK = "#4834D4"
ACCENT = "#00CEC9"       # teal accent
HEALTHY = "#00B894"
RISK = "#FF7675"
TEXT_DARK = "#2D2A4A"
MUTED = "#8B8FB8"

CATEGORICAL_PALETTE = ["#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E", "#74B9FF", "#55EFC4"]
DEPRESSION_COLOR_MAP = {0: HEALTHY, 1: RISK, "False": HEALTHY, "True": RISK,
                         False: HEALTHY, True: RISK}

st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.color_discrete_sequence = CATEGORICAL_PALETTE
px.defaults.template = "plotly_white"

# --------------------------------------------------------------------------
# Global styling
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 0% 0%, #F3F1FF 0%, #F7FBFF 45%, #FFFFFF 100%);
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(195deg, {PRIMARY_DARK} 0%, {PRIMARY} 65%, #8E7CFF 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.25);
    }}
    section[data-testid="stSidebar"] .stRadio > div {{
        gap: 6px;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        background: rgba(255,255,255,0.08);
        padding: 10px 14px;
        border-radius: 12px;
        width: 100%;
        margin-bottom: 2px;
        transition: 0.15s ease-in-out;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    section[data-testid="stSidebar"] .stRadio label:hover {{
        background: rgba(255,255,255,0.18);
    }}
    .sidebar-brand {{
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }}
    .sidebar-tag {{
        font-size: 0.8rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }}

    /* ---------- Page header banner ---------- */
    .page-banner {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #8E7CFF 55%, {ACCENT} 130%);
        border-radius: 22px;
        padding: 2rem 2.2rem;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 14px 30px rgba(108, 92, 231, 0.28);
    }}
    .page-banner h1 {{
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
    }}
    .page-banner p {{
        font-size: 1.02rem;
        opacity: 0.95;
        margin: 0;
        max-width: 760px;
    }}

    /* ---------- Generic content cards ---------- */
    .main-title {{
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: {TEXT_DARK};
    }}
    .subtitle {{
        color: {MUTED};
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background: #FFFFFF;
        border: 1px solid #ECEBFB;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(108, 92, 231, 0.08);
    }}
    .section-card {{
        background: #FFFFFF;
        border: 1px solid #ECEBFB;
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 8px 24px rgba(45, 42, 74, 0.06);
        margin-bottom: 1.4rem;
    }}
    .section-card h3, .section-card h2 {{
        color: {TEXT_DARK};
        margin-top: 0;
    }}

    /* ---------- Result boxes ---------- */
    .result-box {{
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-top: 12px;
        font-size: 1.05rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }}
    .healthy {{
        background: linear-gradient(120deg, #E9FBF4 0%, #DFFAEE 100%);
        border: 1px solid {HEALTHY};
        color: #0B6E52;
    }}
    .risk {{
        background: linear-gradient(120deg, #FFF1F0 0%, #FFE6E3 100%);
        border: 1px solid {RISK};
        color: #B3322B;
    }}

    /* ---------- Pills / badges ---------- */
    .pill {{
        display: inline-block;
        background: #F1EEFF;
        color: {PRIMARY_DARK};
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button, .stFormSubmitButton > button {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #8E7CFF 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        font-weight: 700;
        font-size: 1.0rem;
        width: 100%;
        box-shadow: 0 10px 22px rgba(108, 92, 231, 0.30);
        transition: 0.18s ease-in-out;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 14px 26px rgba(108, 92, 231, 0.4);
    }}

    /* ---------- Metrics (st.metric) ---------- */
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #ECEBFB;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: 0 6px 16px rgba(108, 92, 231, 0.08);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT_DARK};
    }}

    /* ---------- Dataframe / tables ---------- */
    .stDataFrame {{
        border-radius: 14px;
        overflow: hidden;
    }}

    footer, #MainMenu {{visibility: hidden;}}

    .footnote {{
        text-align: center;
        color: {MUTED};
        font-size: 0.82rem;
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Physical_Activity_Hours"] = df["Physical_Activity"] / 60
    df["Lifestyle_Balance"] = (
        df["Study_Hours"] + df["Physical_Activity_Hours"] - df["Social_Media_Hours"]
    )
    df["Productive_Hours"] = df["Study_Hours"] + df["Physical_Activity_Hours"]
    df["Stress_Study"] = df["Stress_Level"] * df["Study_Hours"]
    return df


@st.cache_resource
def load_model(model_file: str):
    return joblib.load(APP_DIR / model_file)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return pd.DataFrame(metrics).drop(columns=["Confusion Matrix"], errors="ignore")
    return pd.DataFrame()


def make_input_row(
    age: int,
    gender: str,
    department: str,
    cgpa: float,
    sleep_duration: float,
    study_hours: float,
    social_media_hours: float,
    physical_activity_minutes: float,
    stress_level: int,
) -> pd.DataFrame:
    physical_activity_hours = physical_activity_minutes / 60
    lifestyle_balance = study_hours + physical_activity_hours - social_media_hours
    productive_hours = study_hours + physical_activity_hours
    stress_study = stress_level * study_hours

    return pd.DataFrame(
        [{
            "Age": age,
            "Gender": gender,
            "Department": department,
            "CGPA": cgpa,
            "Sleep_Duration": sleep_duration,
            "Study_Hours": study_hours,
            "Social_Media_Hours": social_media_hours,
            "Stress_Level": stress_level,
            "Physical_Activity_Hours": physical_activity_hours,
            "Lifestyle_Balance": lifestyle_balance,
            "Productive_Hours": productive_hours,
            "Stress_Study": stress_study,
        }]
    )


def prediction_label(prediction: int) -> str:
    return "Probable Depression" if prediction == 1 else "Healthy"


def page_banner(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="page-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_chart(probability: float):
    color = RISK if probability >= 0.5 else HEALTHY
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36, "color": TEXT_DARK}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#E9FBF4"},
                    {"range": [50, 100], "color": "#FFF1F0"},
                ],
                "threshold": {
                    "line": {"color": TEXT_DARK, "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Poppins"},
    )
    return fig


df = load_data()
metrics_df = load_metrics()

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🎓 StudentMind AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tag">Depression Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown("---")
    page_choice = st.radio(
        "Go to",
        ["🏠 Home", "📊 Dataset Overview", "📈 EDA", "🤖 Model Results", "🔮 Prediction", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("⚠️ Educational tool only — not a medical diagnosis.")

page = page_choice.split(" ", 1)[1]  # strip emoji for logic below

# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
if page == "Home":
    page_banner(
        "🎓 Student Depression Prediction",
        "An interactive machine learning app for predicting probable student depression "
        "based on lifestyle and academic factors.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Target", "Depression")
    c4.metric("Best Model", "Random Forest")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧭 Project Workflow")
    st.write(
        "The project includes data loading, EDA, feature engineering, preprocessing, "
        "PCA, model training, evaluation, and deployment using Streamlit."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🤖 Models Available")
    st.write("Users can choose one of three trained models in the Prediction page:")
    st.markdown(
        """
        <span class="pill">🌲 Random Forest — Best Model</span>
        <span class="pill">📐 Logistic Regression</span>
        <span class="pill">🧬 Logistic Regression + PCA</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Dataset Overview":
    page_banner("📊 Dataset Overview", "Preview of the dataset after feature engineering.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Dataset Shape")
        st.write(f"Rows: **{df.shape[0]:,}**")
        st.write(f"Columns: **{df.shape[1]:,}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Missing Values")
        missing = df.isnull().sum().sum()
        st.write(f"Total missing values: **{missing}**")
        st.write(f"Duplicated rows: **{df.duplicated().sum()}**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "EDA":
    page_banner("📈 Exploratory Data Analysis", "Visual patterns in student lifestyle and academic data.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Target Distribution")
    fig = px.histogram(
        df, x="Depression", color="Depression",
        title="Distribution of Depression Status",
        color_discrete_map=DEPRESSION_COLOR_MAP,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "The target variable is imbalanced, so model evaluation should rely on "
        "Recall, F1-score, and ROC-AUC in addition to Accuracy."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Feature Distribution")
    numeric_options = [
        "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
        "Physical_Activity_Hours", "Stress_Level", "Lifestyle_Balance",
        "Productive_Hours", "Stress_Study",
    ]
    selected_feature = st.selectbox("Choose a numerical feature", numeric_options)
    fig = px.histogram(
        df, x=selected_feature, nbins=30, marginal="box",
        title=f"Distribution of {selected_feature}",
        color_discrete_sequence=[PRIMARY],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Feature by Depression Status")
    compare_feature = st.selectbox("Choose feature to compare with Depression", numeric_options, index=1)
    fig = px.box(
        df, x="Depression", y=compare_feature, color="Depression",
        title=f"{compare_feature} by Depression Status",
        color_discrete_map=DEPRESSION_COLOR_MAP,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Categorical Analysis")
    cat_feature = st.selectbox("Choose categorical feature", ["Gender", "Department"])
    fig = px.histogram(
        df, x=cat_feature, color="Depression", barmode="group",
        title=f"Depression Status by {cat_feature}",
        color_discrete_map=DEPRESSION_COLOR_MAP,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    corr_cols = [
        "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
        "Physical_Activity_Hours", "Stress_Level", "Depression", "Lifestyle_Balance",
        "Productive_Hours", "Stress_Study",
    ]
    corr_df = df[corr_cols].copy()
    corr_df["Depression"] = corr_df["Depression"].astype(int)
    fig = px.imshow(
        corr_df.corr(), text_auto=".2f", aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale=[ACCENT, "#FFFFFF", PRIMARY],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Results":
    page_banner("🤖 Model Results", "Comparing the performance of all trained models.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("The table below summarizes the performance of the trained models.")

    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)

        melted = metrics_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            melted, x="Model", y="Score", color="Metric", barmode="group",
            title="Model Comparison",
            color_discrete_sequence=CATEGORICAL_PALETTE,
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.success(
        "🏆 Random Forest was selected as the final model because it achieved the best "
        "overall balance, including the highest F1-score and ROC-AUC in the final comparison."
    )

elif page == "Prediction":
    page_banner("🔮 Student Depression Prediction", "Enter student information and choose a model to generate a prediction.")

    selected_model_name = st.selectbox("Choose Prediction Model", list(MODEL_FILES.keys()))
    model = load_model(MODEL_FILES[selected_model_name])

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 24, 21)
            gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
            department = st.selectbox("Department", sorted(df["Department"].unique()))
        with col2:
            cgpa = st.slider("CGPA", 1.5, 4.0, 3.0, 0.01)
            sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.1)
            study_hours = st.slider("Study Hours", 0.0, 13.0, 4.5, 0.1)
        with col3:
            social_media_hours = st.slider("Social Media Hours", 0.0, 10.0, 3.5, 0.1)
            physical_activity_minutes = st.slider("Physical Activity (minutes)", 0, 150, 75)
            stress_level = st.slider("Stress Level", 1, 10, 4)

        submitted = st.form_submit_button("🔮 Predict")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        input_data = make_input_row(
            age, gender, department, cgpa, sleep_duration, study_hours,
            social_media_hours, physical_activity_minutes, stress_level,
        )

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        label = prediction_label(prediction)

        st.subheader("Prediction Result")
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            st.metric("Selected Model", selected_model_name)
            st.metric("Probability of Probable Depression", f"{probability:.2%}")

            if prediction == 1:
                st.markdown(
                    f'<div class="result-box risk"><b>⚠️ Result:</b> {label}<br>'
                    f"This prediction suggests possible depression risk. This app is for "
                    f"educational purposes only and is not a medical diagnosis.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-box healthy"><b>✅ Result:</b> {label}<br>'
                    f"The model predicts a lower probability of depression risk. This app is "
                    f"for educational purposes only and is not a medical diagnosis.</div>",
                    unsafe_allow_html=True,
                )

        with gauge_col:
            st.plotly_chart(gauge_chart(probability), use_container_width=True)

        with st.expander("📄 Show input features used by the model"):
            st.dataframe(input_data, use_container_width=True)

elif page == "About":
    page_banner("ℹ️ About This App", "Background on the project and how this app was built.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write(
        "This Streamlit application was built from the Student Depression Prediction machine "
        "learning project. It allows users to explore the dataset, view model results, and "
        "make predictions using one of three trained models."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning("⚠️ This application is for educational use only and should not be used as a medical diagnosis tool.")

st.markdown(
    """
    <div class="footnote">
        Built with ❤️ using Streamlit · Student Depression Prediction Project
    </div>
    """,
    unsafe_allow_html=True,
)
