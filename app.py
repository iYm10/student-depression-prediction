"""
Student Depression Prediction — Streamlit App
Tuwaiq Academy | Machine Learning Project

This app loads the trained models (Logistic Regression, Random Forest,
Logistic Regression + PCA) produced in the project notebook and lets a
user enter a student's lifestyle/academic profile to estimate the
probability of probable depression.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Global styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(180deg, #F7F9FC 0%, #EEF2F9 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #5B6CFF 0%, #7C5CFF 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.12);
        color: #FFFFFF;
        border-radius: 10px;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(120deg, #5B6CFF 0%, #8C6CFF 60%, #B06CFF 100%);
        padding: 2.4rem 2.2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 28px rgba(91, 108, 255, 0.25);
    }
    .hero h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .hero p {
        font-size: 1.02rem;
        opacity: 0.92;
        margin: 0;
    }

    /* Section cards */
    .card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.6rem 1.7rem;
        box-shadow: 0 6px 18px rgba(40, 50, 90, 0.07);
        border: 1px solid rgba(91, 108, 255, 0.07);
        margin-bottom: 1.2rem;
    }
    .card h3 {
        margin-top: 0;
        color: #2B2B45;
        font-weight: 600;
    }

    /* Result banners */
    .result-healthy {
        background: linear-gradient(120deg, #E5FBF1 0%, #D2F7E6 100%);
        border: 1px solid #34C77B;
        color: #16693F;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        text-align: center;
    }
    .result-risk {
        background: linear-gradient(120deg, #FFF1EE 0%, #FFE3DD 100%);
        border: 1px solid #FF6B4A;
        color: #9C3217;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        text-align: center;
    }
    .result-healthy h2, .result-risk h2 {
        margin: 0 0 0.3rem 0;
        font-size: 1.6rem;
    }

    /* Metric pills */
    .pill {
        display: inline-block;
        background: #F0F1FF;
        color: #4A4FE0;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(120deg, #5B6CFF 0%, #8C6CFF 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        font-weight: 600;
        font-size: 1.0rem;
        width: 100%;
        box-shadow: 0 8px 18px rgba(91, 108, 255, 0.30);
        transition: 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(91, 108, 255, 0.38);
    }

    footer, #MainMenu {visibility: hidden;}

    .footnote {
        text-align: center;
        color: #8A8FA8;
        font-size: 0.82rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Constants — must mirror the feature engineering done in the notebook
# --------------------------------------------------------------------------
MODEL_FILES = {
    "Random Forest (Recommended)": "random_forest_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Logistic Regression + PCA": "logistic_pca_model.pkl",
}

GENDER_OPTIONS = ["Male", "Female"]
DEPARTMENT_OPTIONS = ["Engineering", "Science", "Medical", "Arts", "Business"]


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def build_features(age, gender, department, cgpa, sleep_duration,
                    study_hours, social_media_hours, physical_activity_minutes,
                    stress_level):
    """Recreate the exact feature engineering pipeline used in training."""
    physical_activity_hours = physical_activity_minutes / 60

    lifestyle_balance = study_hours + physical_activity_hours - social_media_hours
    productive_hours = study_hours + physical_activity_hours
    stress_study = stress_level * study_hours

    row = pd.DataFrame([{
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
    }])
    return row


def gauge_chart(probability: float):
    color = "#FF6B4A" if probability >= 0.5 else "#34C77B"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 38, "color": "#2B2B45"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#C7CBE8"},
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#E5FBF1"},
                    {"range": [50, 100], "color": "#FFF1EE"},
                ],
                "threshold": {
                    "line": {"color": "#2B2B45", "width": 3},
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


# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Student Depression Risk Predictor</h1>
        <p>An ML-powered tool that estimates a student's probability of probable depression
        from lifestyle and academic factors — built on a Random Forest model trained on
        100,000 student records.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_choice = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

    st.markdown("---")
    st.markdown("### 📊 About the project")
    st.markdown(
        """
        This app is part of a Tuwaiq Academy machine learning project that
        analyzes student lifestyle and academic data to flag students who
        may be at risk of depression.

        **Pipeline:** EDA → Feature Engineering → Preprocessing
        (StandardScaler + OneHotEncoder) → PCA → Classification
        """
    )
    st.markdown("### 🏆 Model performance")
    st.markdown(
        """
        | Model | Accuracy | F1 | ROC-AUC |
        |---|---|---|---|
        | Logistic Regression | 0.618 | 0.260 | 0.679 |
        | **Random Forest** | **0.735** | **0.322** | **0.702** |
        | LogReg + PCA | 0.618 | 0.261 | 0.680 |
        """
    )
    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and is **not** a clinical diagnosis.")

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
left, right = st.columns([1.05, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📝 Student Profile")

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", min_value=18, max_value=24, value=20)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        department = st.selectbox("Department", DEPARTMENT_OPTIONS)
        cgpa = st.slider("CGPA", min_value=1.50, max_value=4.00, value=3.00, step=0.01)
    with c2:
        sleep_duration = st.slider("Sleep Duration (hours/day)", 3.0, 12.0, 7.0, 0.1)
        study_hours = st.slider("Study Hours (hours/day)", 0.0, 13.0, 4.5, 0.1)
        social_media_hours = st.slider("Social Media Usage (hours/day)", 0.0, 10.0, 3.5, 0.1)
        stress_level = st.slider("Stress Level (1 = low, 10 = high)", 1, 10, 4)

    physical_activity_minutes = st.slider(
        "Physical Activity (minutes/day)", min_value=0, max_value=150, value=75, step=5
    )

    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("🔮 Predict Depression Risk", use_container_width=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧮 Engineered Features (auto-calculated)")

    physical_activity_hours = physical_activity_minutes / 60
    lifestyle_balance = study_hours + physical_activity_hours - social_media_hours
    productive_hours = study_hours + physical_activity_hours
    stress_study = stress_level * study_hours

    m1, m2 = st.columns(2)
    m1.metric("Physical Activity (hrs)", f"{physical_activity_hours:.2f}")
    m2.metric("Productive Hours", f"{productive_hours:.2f}")
    m1.metric("Lifestyle Balance", f"{lifestyle_balance:.2f}")
    m2.metric("Stress × Study", f"{stress_study:.1f}")

    st.markdown(
        """
        <span class="pill">Lifestyle_Balance = Study + Activity − Social Media</span>
        <span class="pill">Productive_Hours = Study + Activity</span>
        <span class="pill">Stress_Study = Stress × Study</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if predict_clicked:
    model_path = MODEL_FILES[model_choice]
    model = load_model(model_path)

    if model is None:
        st.error(
            f"⚠️ Couldn't find **{model_path}**. Make sure this file is in the same "
            "folder as `app.py` (it's produced at the end of the training notebook)."
        )
    else:
        features = build_features(
            age, gender, department, cgpa, sleep_duration,
            study_hours, social_media_hours, physical_activity_minutes,
            stress_level,
        )

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        st.markdown("### 🔍 Prediction Result")
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-risk">
                        <h2>⚠️ Probable Depression</h2>
                        <p>The model estimates a <b>{probability*100:.1f}%</b> probability
                        of probable depression based on the provided profile.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-healthy">
                        <h2>✅ Healthy</h2>
                        <p>The model estimates a <b>{probability*100:.1f}%</b> probability
                        of probable depression — within the healthy range.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(f"Model used: **{model_choice}**")

        with gauge_col:
            st.plotly_chart(gauge_chart(probability), use_container_width=True)

        with st.expander("📄 View input data sent to the model"):
            st.dataframe(features, use_container_width=True)

        st.info(
            "💡 This prediction reflects statistical patterns learned from data and "
            "should never replace professional medical or psychological advice. "
            "If you or someone you know is struggling, please reach out to a counselor "
            "or mental health professional."
        )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="footnote">
        Built with ❤️ using Streamlit · Student Depression Prediction Project · Tuwaiq Academy
    </div>
    """,
    unsafe_allow_html=True,
)
