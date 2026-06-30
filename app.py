import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "student_lifestyle_100k.csv"
METRICS_PATH = APP_DIR / "model_metrics.json"

MODEL_FILES = {
    "Random Forest (Best Model)": "random_forest_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Logistic Regression + PCA": "logistic_pca_model.pkl",
}

NUMERIC_OPTIONS = [
    "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
    "Physical_Activity_Hours", "Stress_Level", "Lifestyle_Balance",
    "Productive_Hours", "Stress_Study",
]

# ----------------------------------------------------------------------------
# Unified color palette - تصميم جديد ومريح
# ----------------------------------------------------------------------------
COLORS = {
    "primary": "#5B6ABF",          # أزرق/نيلي ناعم
    "primary_dark": "#3F4A8F",
    "primary_light": "#E8ECF8",
    "accent": "#F9A8A4",           # مرجاني دافئ
    "accent_light": "#FEF0EE",
    "success": "#34D399",          # زمردي
    "success_light": "#ECFDF5",
    "warning": "#FBBF24",          # كهرماني
    "warning_light": "#FEF3C7",
    "bg": "#F8FAFC",               # رمادي فاتح جداً
    "surface": "#FFFFFF",
    "border": "#E2E8F0",
    "text": "#1E293B",
    "text_muted": "#64748B",
}

# سلاسل ألوان متوافقة مع التصميم الجديد
CHART_SEQUENCE = ["#5B6ABF", "#F9A8A4", "#FBBF24", "#34D399", "#A78BFA", "#F472B6"]
CHART_DIVERGING = [[0, "#34D399"], [0.5, "#F8FAFC"], [1, "#5B6ABF"]]

px.defaults.color_discrete_sequence = CHART_SEQUENCE
px.defaults.template = "plotly_white"

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling - كود CSS المحدّث
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --primary: {COLORS["primary"]};
        --primary-dark: {COLORS["primary_dark"]};
        --primary-light: {COLORS["primary_light"]};
        --accent: {COLORS["accent"]};
        --accent-light: {COLORS["accent_light"]};
        --success: {COLORS["success"]};
        --success-light: {COLORS["success_light"]};
        --warning: {COLORS["warning"]};
        --warning-light: {COLORS["warning_light"]};
        --bg: {COLORS["bg"]};
        --surface: {COLORS["surface"]};
        --border: {COLORS["border"]};
        --text: {COLORS["text"]};
        --text-muted: {COLORS["text_muted"]};
    }}

    /* خلفية التطبيق */
    .stApp {{
        background-color: var(--bg);
    }}

    /* النصوص */
    h1, h2, h3, h4, h5, p, span, label, div {{
        color: var(--text);
    }}

    /* الهيدر الرئيسي - تدرج ناعم */
    .hero {{
        background: linear-gradient(145deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 24px;
        padding: 48px 40px;
        color: white !important;
        margin-bottom: 2rem;
        box-shadow: 0 16px 40px rgba(91, 106, 191, 0.25);
        transition: box-shadow 0.3s ease;
    }}
    .hero:hover {{
        box-shadow: 0 20px 48px rgba(91, 106, 191, 0.35);
    }}
    .hero h1, .hero p {{
        color: white !important;
    }}
    .hero h1 {{
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}
    .hero p {{
        font-size: 1.1rem;
        opacity: 0.92;
        max-width: 720px;
        margin: 0;
        line-height: 1.6;
    }}

    /* عناوين الأقسام */
    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.8rem 0 1.2rem 0;
        color: var(--text);
        border-left: 6px solid var(--primary);
        padding-left: 14px;
    }}

    /* البطاقات - ظلال ناعمة وزوايا دائرية */
    .card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
        height: 100%;
        transition: all 0.2s ease;
    }}
    .card:hover {{
        box-shadow: 0 12px 28px rgba(91, 106, 191, 0.10);
        transform: translateY(-3px);
        border-color: var(--primary-light);
    }}
    .card h4 {{
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.1rem;
        color: var(--primary-dark);
    }}
    .card p {{
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-bottom: 0;
        line-height: 1.5;
    }}

    /* الشارات */
    .badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 10px;
        background: var(--primary-light);
        color: var(--primary-dark);
    }}
    .badge-best {{
        background: var(--success-light);
        color: #0B7B3E;
        border: 1px solid var(--success);
    }}

    /* مربعات النتيجة */
    .result-box {{
        border-radius: 20px;
        padding: 24px;
        margin-top: 20px;
        font-size: 1.05rem;
        line-height: 1.6;
    }}
    .healthy {{
        background: var(--success-light);
        border: 1px solid var(--success);
        color: #0B7B3E;
    }}
    .risk {{
        background: var(--warning-light);
        border: 1px solid var(--warning);
        color: #92400E;
    }}
    .disclaimer {{
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 8px;
    }}

    /* بطاقات المقاييس الأصلية في Streamlit */
    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        transition: box-shadow 0.2s;
    }}
    div[data-testid="stMetric"]:hover {{
        box-shadow: 0 6px 16px rgba(91, 106, 191, 0.08);
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--primary-dark);
        font-weight: 600;
    }}

    /* التبويبات */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--primary-light);
        border-radius: 12px 12px 0 0;
        color: var(--primary-dark);
        font-weight: 600;
        padding: 10px 20px;
        border: 1px solid transparent;
        border-bottom: none;
        transition: all 0.2s;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: #D6DEF5;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--primary) !important;
        color: white !important;
        border-color: var(--primary);
    }}

    /* الأزرار */
    .stButton > button, .stFormSubmitButton > button {{
        background: linear-gradient(145deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(91, 106, 191, 0.25);
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        opacity: 0.92;
        transform: scale(1.02);
        box-shadow: 0 6px 18px rgba(91, 106, 191, 0.35);
        color: white;
    }}

    /* الشريط الجانبي - تدرج غامق ناعم */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #2D3A6B 0%, #1E2A4F 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }}
    section[data-testid="stSidebar"] * {{
        color: #EDF2F7 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.12);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        border-radius: 12px;
        padding: 8px 14px;
        margin-bottom: 4px;
        transition: background 0.15s ease;
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: rgba(255,255,255,0.10);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {{
        background: rgba(255,255,255,0.15);
        box-shadow: inset 0 0 0 2px var(--accent);
    }}

    /* جداول البيانات */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Data / model loading (with error handling)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Physical_Activity_Hours"] = df["Physical_Activity"] / 60
    df["Lifestyle_Balance"] = (
        df["Study_Hours"] + df["Physical_Activity_Hours"] - df["Social_Media_Hours"]
    )
    df["Productive_Hours"] = df["Study_Hours"] + df["Physical_Activity_Hours"]
    df["Stress_Study"] = df["Stress_Level"] * df["Study_Hours"]
    return df


@st.cache_resource(show_spinner="Loading model...")
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


def style_fig(fig: go.Figure) -> go.Figure:
    """Apply a consistent, eye-friendly look to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"], "family": "sans-serif"},
        title={"font": {"color": COLORS["text"], "size": 16}},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        margin=dict(t=55, b=30, l=10, r=10),
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    return fig


def gauge_figure(probability: float, prediction: int) -> go.Figure:
    bar_color = COLORS["warning"] if prediction == 1 else COLORS["accent"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36, "color": COLORS["text"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_muted"]},
                "bar": {"color": bar_color},
                "bgcolor": COLORS["bg"],
                "borderwidth": 1,
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0, 40], "color": COLORS["success_light"]},
                    {"range": [40, 70], "color": COLORS["primary_light"]},
                    {"range": [70, 100], "color": COLORS["warning_light"]},
                ],
            },
            title={"text": "Probability of Probable Depression", "font": {"color": COLORS["text_muted"], "size": 14}},
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
    )
    return fig


# ----------------------------------------------------------------------------
# Guard: make sure required files exist before rendering the app
# ----------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(
        f"Dataset file not found: `{DATA_PATH.name}`. "
        "Please place it in the same folder as this app and reload."
    )
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

metrics_df = load_metrics()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Student Depression")
    st.caption("ML-powered risk screening app")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Dataset Overview", "📈 EDA", "🤖 Model Results", "🔮 Prediction", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Streamlit · scikit-learn · Plotly")
    st.caption("⚠️ Educational use only — not a medical tool.")

page = page.split(" ", 1)[1]  # strip emoji prefix

# ----------------------------------------------------------------------------
# HOME
# ----------------------------------------------------------------------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>🎓 Student Depression Prediction</h1>
            <p>An interactive machine learning app that estimates probable depression
            risk in students based on academic and lifestyle factors — built for
            education and awareness, not diagnosis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Target", "Depression")
    c4.metric("Best Model", "Random Forest")

    st.markdown('<div class="section-title">Project Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("1. Data Loading", "Importing and validating the raw student lifestyle dataset."),
        ("2. EDA", "Exploring distributions, correlations, and group comparisons."),
        ("3. Feature Engineering", "Deriving Lifestyle Balance, Productive Hours, Stress × Study."),
        ("4. Preprocessing & PCA", "Encoding, scaling, and optional dimensionality reduction."),
        ("5. Model Training", "Random Forest and Logistic Regression variants."),
        ("6. Deployment", "Interactive prediction served through this Streamlit app."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f'<div class="card"><h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
            st.write("")

    st.markdown('<div class="section-title">Available Models</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            '<div class="card"><span class="badge badge-best">BEST MODEL</span>'
            "<h4>Random Forest</h4><p>Highest F1-score and ROC-AUC overall.</p></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="card"><h4>Logistic Regression</h4>'
            "<p>Simple, interpretable linear baseline.</p></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="card"><h4>Logistic Regression + PCA</h4>'
            "<p>Dimensionality-reduced variant for comparison.</p></div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# DATASET OVERVIEW
# ----------------------------------------------------------------------------
elif page == "Dataset Overview":
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    st.caption("Preview of the dataset after feature engineering.")

    st.dataframe(df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    col3.metric("Duplicated Rows", f"{df.duplicated().sum():,}")

    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

    with st.expander("Column data types"):
        st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)

# ----------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------
elif page == "EDA":
    st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Target", "Feature Distributions", "Categorical", "Correlation"]
    )

    with tab1:
        fig = px.histogram(df, x="Depression", color="Depression", title="Distribution of Depression Status")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.info(
            "The target variable is imbalanced, so model evaluation should rely on "
            "Recall, F1-score, and ROC-AUC in addition to Accuracy."
        )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            selected_feature = st.selectbox("Choose a numerical feature", NUMERIC_OPTIONS, key="dist_feat")
            fig = px.histogram(
                df, x=selected_feature, nbins=30,
                title=f"Distribution of {selected_feature}", marginal="box",
                color_discrete_sequence=[COLORS["primary"]],
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            compare_feature = st.selectbox(
                "Choose feature to compare with Depression", NUMERIC_OPTIONS, index=1, key="cmp_feat"
            )
            fig = px.box(
                df, x="Depression", y=compare_feature, color="Depression",
                title=f"{compare_feature} by Depression Status",
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with tab3:
        cat_feature = st.selectbox("Choose categorical feature", ["Gender", "Department"])
        fig = px.histogram(
            df, x=cat_feature, color="Depression", barmode="group",
            title=f"Depression Status by {cat_feature}",
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tab4:
        corr_cols = NUMERIC_OPTIONS + ["Depression"]
        corr_df = df[corr_cols].copy()
        corr_df["Depression"] = corr_df["Depression"].astype(int)
        fig = px.imshow(
            corr_df.corr(), text_auto=".2f", aspect="auto", title="Correlation Heatmap",
            color_continuous_scale=CHART_DIVERGING,
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ----------------------------------------------------------------------------
# MODEL RESULTS
# ----------------------------------------------------------------------------
elif page == "Model Results":
    st.markdown('<div class="section-title">🤖 Model Results</div>', unsafe_allow_html=True)
    st.caption("Performance comparison of the trained models on the held-out test set.")

    if not metrics_df.empty:
        st.dataframe(
            metrics_df.style.highlight_max(
                subset=[c for c in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"] if c in metrics_df.columns],
                color=COLORS["success_light"],
            ),
            use_container_width=True,
        )

        melted = metrics_df.melt(
            id_vars="Model",
            value_vars=[c for c in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"] if c in metrics_df.columns],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            melted, x="Model", y="Score", color="Metric", barmode="group",
            title="Model Comparison", range_y=[0, 1],
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
    else:
        st.warning(f"Metrics file not found: `{METRICS_PATH.name}`.")

    st.success(
        "Random Forest was selected as the final model because it achieved the best "
        "overall balance, including the highest F1-score and ROC-AUC in the final comparison."
    )

    # Optional feature importance, only shown if the RF model & file are available
    rf_path = APP_DIR / MODEL_FILES["Random Forest (Best Model)"]
    if rf_path.exists():
        try:
            rf_model = load_model(MODEL_FILES["Random Forest (Best Model)"])
            estimator = rf_model.named_steps["model"] if hasattr(rf_model, "named_steps") else rf_model
            if hasattr(estimator, "feature_importances_"):
                with st.expander("Feature Importance (Random Forest)"):
                    importances = pd.Series(
                        estimator.feature_importances_,
                        index=getattr(estimator, "feature_names_in_", range(len(estimator.feature_importances_))),
                    ).sort_values(ascending=True)
                    fig = px.bar(
                        importances, orientation="h", title="Feature Importance",
                        color_discrete_sequence=[COLORS["primary"]],
                    )
                    st.plotly_chart(style_fig(fig), use_container_width=True)
        except Exception:
            pass  # Importance display is a bonus, never block the page on it

# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
elif page == "Prediction":
    st.markdown('<div class="section-title">🔮 Student Depression Prediction</div>', unsafe_allow_html=True)
    st.caption("Enter student information and choose a model to generate a prediction.")

    available_models = {name: f for name, f in MODEL_FILES.items() if (APP_DIR / f).exists()}
    if not available_models:
        st.error("No trained model files were found next to this app.")
        st.stop()

    selected_model_name = st.selectbox("Choose Prediction Model", list(available_models.keys()))

    try:
        model = load_model(available_models[selected_model_name])
    except Exception as e:
        st.error(f"Failed to load model `{selected_model_name}`: {e}")
        st.stop()

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

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        input_data = make_input_row(
            age, gender, department, cgpa, sleep_duration, study_hours,
            social_media_hours, physical_activity_minutes, stress_level,
        )

        try:
            prediction = int(model.predict(input_data)[0])
            probability = float(model.predict_proba(input_data)[0][1])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        label = prediction_label(prediction)

        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        rcol1, rcol2 = st.columns([1, 1])
        with rcol1:
            st.plotly_chart(gauge_figure(probability, prediction), use_container_width=True)
        with rcol2:
            st.metric("Selected Model", selected_model_name)
            st.metric("Predicted Class", label)
            box_class = "risk" if prediction == 1 else "healthy"
            message = (
                "This prediction suggests possible depression risk."
                if prediction == 1
                else "The model predicts a lower probability of depression risk."
            )
            st.markdown(
                f'<div class="result-box {box_class}"><b>Result:</b> {label}<br>{message}'
                '<div class="disclaimer">This app is for educational purposes only and is not a medical diagnosis.</div></div>',
                unsafe_allow_html=True,
            )

        with st.expander("Show input features used by the model"):
            st.dataframe(input_data, use_container_width=True)

# ----------------------------------------------------------------------------
# ABOUT
# ----------------------------------------------------------------------------
elif page == "About":
    st.markdown('<div class="section-title">ℹ️ About This App</div>', unsafe_allow_html=True)
    st.write(
        "This Streamlit application was built from the Student Depression Prediction "
        "machine learning project. It allows users to explore the dataset, view model "
        "results, and make predictions using one of three trained models."
    )
    st.markdown(
        """
        **Tech stack**
        - Data processing: `pandas`, `numpy`
        - Visualization: `plotly`
        - Modeling: `scikit-learn`
        - Interface: `streamlit`
        """
    )
    st.warning("This application is for educational use only and should not be used as a medical diagnosis tool.")
