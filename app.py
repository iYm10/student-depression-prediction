import json
import time
from pathlib import Path

import joblib
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

st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Professional CSS Styling
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #111827 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.96);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(37,99,235,0.25), rgba(6,182,212,0.18));
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 28px;
        padding: 42px 36px;
        margin-bottom: 28px;
        box-shadow: 0 20px 55px rgba(0,0,0,0.30);
        text-align: center;
    }

    .main-title {
        font-size: 3.3rem;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 0.45rem;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #cbd5e1;
        font-size: 1.18rem;
        max-width: 900px;
        margin: 0 auto;
        line-height: 1.7;
    }

    .section-title {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 1.4rem;
        margin-bottom: 0.8rem;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 14px 35px rgba(0,0,0,0.24);
        margin-bottom: 18px;
    }

    .small-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 22px;
        min-height: 145px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.20);
        transition: transform 0.2s ease, border 0.2s ease;
    }

    .small-card:hover {
        transform: translateY(-3px);
        border: 1px solid rgba(56, 189, 248, 0.45);
    }

    .card-icon {
        font-size: 2.1rem;
        margin-bottom: 8px;
    }

    .card-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .card-text {
        color: #cbd5e1;
        font-size: 0.96rem;
        line-height: 1.55;
    }

    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.20);
        padding: 18px;
        border-radius: 20px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 900;
    }

    div[data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.82);
        padding: 30px;
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 16px 40px rgba(0,0,0,0.24);
    }

    .stButton > button, div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        height: 54px;
        border-radius: 16px;
        font-size: 18px;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        color: white;
        border: none;
        box-shadow: 0 10px 25px rgba(37,99,235,0.28);
    }

    .stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(90deg, #1d4ed8, #0891b2);
        color: white;
        border: none;
    }

    .result-box {
        border-radius: 24px;
        padding: 26px;
        margin-top: 18px;
        font-size: 1.05rem;
        box-shadow: 0 16px 40px rgba(0,0,0,0.25);
    }

    .healthy {
        background: linear-gradient(135deg, rgba(16,185,129,0.22), rgba(5,150,105,0.12));
        border: 1px solid rgba(16,185,129,0.55);
        color: #d1fae5;
    }

    .risk {
        background: linear-gradient(135deg, rgba(249,115,22,0.24), rgba(239,68,68,0.14));
        border: 1px solid rgba(249,115,22,0.58);
        color: #ffedd5;
    }

    .result-title {
        font-size: 1.7rem;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 35px;
        padding-top: 18px;
        border-top: 1px solid rgba(148,163,184,0.18);
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: inherit;
    }

    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }


    .home-metric-card {
        background: rgba(30, 41, 59, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 20px;
        padding: 22px 18px;
        min-height: 150px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
        overflow: visible !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .home-metric-title {
        color: #cbd5e1 !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 14px;
        white-space: nowrap;
    }

    .home-metric-value {
        color: #ffffff !important;
        font-size: clamp(24px, 2.2vw, 36px);
        font-weight: 900;
        line-height: 1.15;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: normal;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Helper Functions
# =========================
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


def style_plot(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", size=14),
        title_font=dict(size=22, color="#ffffff"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def card(icon: str, title: str, text: str):
    st.markdown(
        f"""
        <div class="small-card">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <div class="card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def home_metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="home-metric-card">
            <div class="home-metric-title">{title}</div>
            <div class="home-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="footer">
            Developed by Yahya | Data Science & AI Diploma | Streamlit + Scikit-learn
        </div>
        """,
        unsafe_allow_html=True,
    )


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


# =========================
# Load Data
# =========================
df = load_data()
metrics_df = load_metrics()

# =========================
# Sidebar
# =========================
st.sidebar.markdown("# 🎓 Student AI")
st.sidebar.markdown("A professional ML dashboard for student depression prediction.")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Dataset Overview", "📈 EDA", "🤖 Model Results", "🔮 Prediction", "ℹ️ About"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, Plotly and Scikit-learn")

# =========================
# Home Page
# =========================
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero-card">
            <div class="main-title">🎓 Student Depression Prediction</div>
            <div class="subtitle">
                A professional machine learning dashboard for analyzing student lifestyle and
                predicting depression risk through interactive data visualization and AI models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        home_metric_card("Rows", f"{df.shape[0]:,}")
    with c2:
        home_metric_card("Columns", f"{df.shape[1]}")
    with c3:
        home_metric_card("Target", "Depression")
    with c4:
        home_metric_card("Best Model", "Random Forest")

    st.markdown('<div class="section-title">Project Highlights</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        card("📊", "Data Exploration", "Explore dataset shape, missing values, summary statistics, and engineered features.")
    with h2:
        card("🤖", "Machine Learning", "Compare Random Forest, Logistic Regression, and Logistic Regression with PCA.")
    with h3:
        card("🔮", "Smart Prediction", "Enter student information and generate a prediction with probability score.")

    st.markdown('<div class="section-title">Project Workflow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            The project includes data loading, EDA, feature engineering, preprocessing,
            PCA, model training, evaluation, and deployment using Streamlit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Models Available</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        card("🌲", "Random Forest", "Selected as the best final model based on overall performance.")
    with m2:
        card("📉", "Logistic Regression", "A simple and interpretable baseline classification model.")
    with m3:
        card("🧩", "Logistic + PCA", "Uses dimensionality reduction before classification.")

    footer()

# =========================
# Dataset Overview Page
# =========================
elif page == "📊 Dataset Overview":
    st.markdown('<div class="main-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Preview and summary of the dataset after feature engineering.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    col4.metric("Duplicated Rows", f"{df.duplicated().sum():,}")

    tab1, tab2 = st.tabs(["Dataset Preview", "Summary Statistics"])
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

    footer()

# =========================
# EDA Page
# =========================
elif page == "📈 EDA":
    st.markdown('<div class="main-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive visual analysis of student lifestyle and academic features.</div>', unsafe_allow_html=True)

    numeric_options = [
        "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
        "Physical_Activity_Hours", "Stress_Level", "Lifestyle_Balance",
        "Productive_Hours", "Stress_Study",
    ]

    tab1, tab2, tab3, tab4 = st.tabs([
        "Target Distribution",
        "Feature Distribution",
        "Feature Comparison",
        "Correlation Heatmap",
    ])

    with tab1:
        fig = px.histogram(
            df,
            x="Depression",
            color="Depression",
            title="Distribution of Depression Status",
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)
        st.info("The target variable is imbalanced, so evaluation should include Recall, F1-score, and ROC-AUC in addition to Accuracy.")

    with tab2:
        selected_feature = st.selectbox("Choose a numerical feature", numeric_options)
        fig = px.histogram(
            df,
            x=selected_feature,
            nbins=30,
            title=f"Distribution of {selected_feature}",
            marginal="box",
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with tab3:
        compare_feature = st.selectbox("Choose feature to compare with Depression", numeric_options, index=1)
        fig = px.box(
            df,
            x="Depression",
            y=compare_feature,
            color="Depression",
            title=f"{compare_feature} by Depression Status",
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

        cat_feature = st.selectbox("Choose categorical feature", ["Gender", "Department"])
        fig = px.histogram(
            df,
            x=cat_feature,
            color="Depression",
            barmode="group",
            title=f"Depression Status by {cat_feature}",
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with tab4:
        corr_cols = [
            "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
            "Physical_Activity_Hours", "Stress_Level", "Depression", "Lifestyle_Balance",
            "Productive_Hours", "Stress_Study",
        ]
        corr_df = df[corr_cols].copy()
        corr_df["Depression"] = corr_df["Depression"].astype(int)
        fig = px.imshow(
            corr_df.corr(),
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap",
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    footer()

# =========================
# Model Results Page
# =========================
elif page == "🤖 Model Results":
    st.markdown('<div class="main-title">🤖 Model Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Performance comparison of the trained machine learning models.</div>', unsafe_allow_html=True)

    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)

        melted = metrics_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            melted,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Model Comparison",
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)
    else:
        st.warning("model_metrics.json was not found. Please make sure the metrics file exists in the app folder.")

    st.markdown(
        """
        <div class="glass-card">
            ✅ <b>Final Model:</b> Random Forest was selected as the final model because it achieved
            the best overall balance, including strong F1-score and ROC-AUC performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    footer()

# =========================
# Prediction Page
# =========================
elif page == "🔮 Prediction":
    st.markdown('<div class="main-title">🔮 Student Depression Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter student information and choose a model to generate a prediction.</div>', unsafe_allow_html=True)

    selected_model_name = st.selectbox("Choose Prediction Model", list(MODEL_FILES.keys()))
    model = load_model(MODEL_FILES[selected_model_name])

    with st.form("prediction_form"):
        st.markdown("### Student Information")
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

        submitted = st.form_submit_button("Predict")

    if submitted:
        with st.spinner("Generating prediction..."):
            time.sleep(0.6)

        input_data = make_input_row(
            age,
            gender,
            department,
            cgpa,
            sleep_duration,
            study_hours,
            social_media_hours,
            physical_activity_minutes,
            stress_level,
        )

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        label = prediction_label(prediction)

        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1])
        with r1:
            st.metric("Selected Model", selected_model_name)
            st.metric("Probability of Probable Depression", f"{probability:.2%}")

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-box risk">
                        <div class="result-title">🟠 Result: {label}</div>
                        This prediction suggests a possible depression risk.<br><br>
                        <b>Important:</b> This app is for educational purposes only and is not a medical diagnosis.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box healthy">
                        <div class="result-title">🟢 Result: {label}</div>
                        The model predicts a lower probability of depression risk.<br><br>
                        <b>Important:</b> This app is for educational purposes only and is not a medical diagnosis.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with r2:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={"text": "Depression Risk (%)", "font": {"size": 22, "color": "#ffffff"}},
                    number={"suffix": "%", "font": {"size": 42, "color": "#ffffff"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#e5e7eb"},
                        "bar": {"color": "#38bdf8"},
                        "bgcolor": "rgba(30,41,59,0.5)",
                        "borderwidth": 1,
                        "bordercolor": "rgba(148,163,184,0.3)",
                        "steps": [
                            {"range": [0, 40], "color": "rgba(16,185,129,0.35)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,0.35)"},
                            {"range": [70, 100], "color": "rgba(239,68,68,0.35)"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=60, b=20),
                height=360,
            )
            st.plotly_chart(gauge, use_container_width=True)

        with st.expander("Show input features used by the model"):
            st.dataframe(input_data, use_container_width=True)

    footer()

# =========================
# About Page
# =========================
elif page == "ℹ️ About":
    st.markdown('<div class="main-title">ℹ️ About This App</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            This Streamlit application was built from the Student Depression Prediction
            machine learning project. It allows users to explore the dataset, view model results,
            and make predictions using one of three trained models.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        card("🎯", "Goal", "Support educational understanding of ML-based prediction workflows.")
    with a2:
        card("🧠", "Technology", "Built using Python, Streamlit, Plotly, Pandas, and Scikit-learn.")
    with a3:
        card("⚠️", "Disclaimer", "This application is for educational use only and is not a medical diagnosis tool.")

    st.warning("This application is for educational use only and should not be used as a medical diagnosis tool.")

    footer()
