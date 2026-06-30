import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
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

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
    }
    .result-box {
        border-radius: 16px;
        padding: 18px;
        margin-top: 12px;
        font-size: 1.05rem;
    }
    .healthy {
        background: #ecfdf5;
        border: 1px solid #10b981;
        color: #065f46;
    }
    .risk {
        background: #fff7ed;
        border: 1px solid #f97316;
        color: #9a3412;
    }
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


df = load_data()
metrics_df = load_metrics()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Dataset Overview", "EDA", "Model Results", "Prediction", "About"],
)

if page == "Home":
    st.markdown('<div class="main-title">🎓 Student Depression Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">An interactive machine learning app for predicting probable student depression based on lifestyle and academic factors.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Target", "Depression")
    c4.metric("Best Model", "Random Forest")

    st.subheader("Project Workflow")
    st.write(
        "The project includes data loading, EDA, feature engineering, preprocessing, PCA, model training, evaluation, and deployment using Streamlit."
    )

    st.subheader("Models Available")
    st.write("Users can choose one of three trained models in the Prediction page:")
    st.markdown("- Random Forest (selected as the best model)\n- Logistic Regression\n- Logistic Regression + PCA")

elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("Preview of the dataset after feature engineering.")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: **{df.shape[0]:,}**")
        st.write(f"Columns: **{df.shape[1]:,}**")
    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum().sum()
        st.write(f"Total missing values: **{missing}**")
        st.write(f"Duplicated rows: **{df.duplicated().sum()}**")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("Target Distribution")
    fig = px.histogram(df, x="Depression", color="Depression", title="Distribution of Depression Status")
    st.plotly_chart(fig, use_container_width=True)
    st.info("The target variable is imbalanced, so model evaluation should rely on Recall, F1-score, and ROC-AUC in addition to Accuracy.")

    st.subheader("Feature Distribution")
    numeric_options = [
        "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
        "Physical_Activity_Hours", "Stress_Level", "Lifestyle_Balance",
        "Productive_Hours", "Stress_Study",
    ]
    selected_feature = st.selectbox("Choose a numerical feature", numeric_options)
    fig = px.histogram(df, x=selected_feature, nbins=30, title=f"Distribution of {selected_feature}", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature by Depression Status")
    compare_feature = st.selectbox("Choose feature to compare with Depression", numeric_options, index=1)
    fig = px.box(df, x="Depression", y=compare_feature, color="Depression", title=f"{compare_feature} by Depression Status")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Analysis")
    cat_feature = st.selectbox("Choose categorical feature", ["Gender", "Department"])
    fig = px.histogram(df, x=cat_feature, color="Depression", barmode="group", title=f"Depression Status by {cat_feature}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_cols = [
        "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
        "Physical_Activity_Hours", "Stress_Level", "Depression", "Lifestyle_Balance",
        "Productive_Hours", "Stress_Study",
    ]
    corr_df = df[corr_cols].copy()
    corr_df["Depression"] = corr_df["Depression"].astype(int)
    fig = px.imshow(corr_df.corr(), text_auto=".2f", aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Results":
    st.title("🤖 Model Results")
    st.write("The table below summarizes the performance of the trained models.")

    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)

        melted = metrics_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", title="Model Comparison")
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Random Forest was selected as the final model because it achieved the best overall balance, including the highest F1-score and ROC-AUC in the final comparison."
    )

elif page == "Prediction":
    st.title("🔮 Student Depression Prediction")
    st.write("Enter student information and choose a model to generate a prediction.")

    selected_model_name = st.selectbox("Choose Prediction Model", list(MODEL_FILES.keys()))
    model = load_model(MODEL_FILES[selected_model_name])

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

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = make_input_row(
            age, gender, department, cgpa, sleep_duration, study_hours,
            social_media_hours, physical_activity_minutes, stress_level,
        )

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        label = prediction_label(prediction)

        st.subheader("Prediction Result")
        st.metric("Selected Model", selected_model_name)
        st.metric("Probability of Probable Depression", f"{probability:.2%}")

        if prediction == 1:
            st.markdown(
                f'<div class="result-box risk"><b>Result:</b> {label}<br>This prediction suggests possible depression risk. This app is for educational purposes only and is not a medical diagnosis.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-box healthy"><b>Result:</b> {label}<br>The model predicts a lower probability of depression risk. This app is for educational purposes only and is not a medical diagnosis.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Show input features used by the model"):
            st.dataframe(input_data, use_container_width=True)

elif page == "About":
    st.title("ℹ️ About This App")
    st.write(
        "This Streamlit application was built from the Student Depression Prediction machine learning project. "
        "It allows users to explore the dataset, view model results, and make predictions using one of three trained models."
    )
    st.warning("This application is for educational use only and should not be used as a medical diagnosis tool.")
