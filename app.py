import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CSS STYLE
# =========================
st.markdown("""
<style>
    .main {
        background-color: #F7F9FC;
    }

    h1, h2, h3 {
        color: #1F4E79;
        font-family: 'Segoe UI', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF4FF;
    }

    .metric-card {
        background: linear-gradient(135deg, #EAF4FF, #FFFFFF);
        padding: 20px;
        border-radius: 16px;
        border-left: 6px solid #1F77B4;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }

    .insight-card {
        background: linear-gradient(135deg, #EAF4FF, #FFFFFF);
        padding: 20px;
        border-radius: 16px;
        border-left: 6px solid #1F77B4;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-top: 18px;
        margin-bottom: 18px;
    }

    .success-card {
        background: linear-gradient(135deg, #E9F7EF, #FFFFFF);
        padding: 20px;
        border-radius: 16px;
        border-left: 6px solid #2E8B57;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-top: 18px;
    }

    .warning-card {
        background: linear-gradient(135deg, #FFF4E5, #FFFFFF);
        padding: 20px;
        border-radius: 16px;
        border-left: 6px solid #F39C12;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-top: 18px;
    }

    .stButton > button {
        background-color: #1F77B4;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #145A86;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD FILES
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("student_lifestyle_100k.csv")


@st.cache_data
def load_results():
    return pd.read_csv("model_results.csv")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


df = load_data()
results = load_results()

models = {
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Logistic Regression + PCA": "logistic_pca_model.pkl"
}


# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧠 Student Depression")
st.sidebar.markdown("Machine Learning App")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Overview",
        "📈 EDA",
        "🤖 Model Results",
        "🔮 Prediction",
        "ℹ️ About"
    ]
)


# =========================
# HOME
# =========================
if page == "🏠 Home":
    st.title("🧠 Student Depression Prediction")
    st.markdown("""
    This application predicts **probable student depression** based on lifestyle and academic factors.

    The project includes:
    - Data analysis and visualization
    - Feature engineering
    - PCA dimensionality reduction
    - Machine learning model comparison
    - Interactive prediction using Streamlit
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Rows</h3>
            <h2>{df.shape[0]:,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Columns</h3>
            <h2>{df.shape[1]}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Best Model</h3>
            <h2>Random Forest</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        <h4>Project Goal</h4>
        <p>
        The goal is to support early identification of students who may be at risk of depression
        using machine learning models trained on academic and lifestyle features.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# DATA OVERVIEW
# =========================
elif page == "📊 Data Overview":
    st.title("📊 Data Overview")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: **{df.shape[0]:,}**")
    st.write(f"Columns: **{df.shape[1]}**")

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}
    ), use_container_width=True)

    st.subheader("Data Types")
    st.dataframe(df.dtypes.reset_index().rename(
        columns={"index": "Column", 0: "Data Type"}
    ), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)


# =========================
# EDA
# =========================
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    chart = st.selectbox(
        "Choose Visualization",
        [
            "Depression Distribution",
            "Age Distribution",
            "CGPA Distribution",
            "Sleep Duration Distribution",
            "Study Hours Distribution",
            "Gender Distribution",
            "Department Distribution",
            "CGPA by Depression",
            "Sleep Duration by Depression",
            "Social Media Hours by Depression",
            "Stress Level by Depression",
            "Correlation Heatmap"
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if chart == "Depression Distribution":
        sns.countplot(data=df, x="Depression", ax=ax)
        ax.set_title("Distribution of Depression Status")
        ax.set_xlabel("Depression")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.info("The target variable is imbalanced, so evaluation should not depend on accuracy alone.")

    elif chart == "Age Distribution":
        sns.countplot(data=df, x="Age", ax=ax)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.info("The dataset includes university-aged students with a relatively balanced age distribution.")

    elif chart == "CGPA Distribution":
        sns.histplot(df["CGPA"], bins=20, kde=True, ax=ax)
        ax.set_title("CGPA Distribution")
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.info("CGPA values cover a wide range of academic performance levels.")

    elif chart == "Sleep Duration Distribution":
        sns.histplot(df["Sleep_Duration"], bins=20, kde=True, ax=ax)
        ax.set_title("Sleep Duration Distribution")
        ax.set_xlabel("Sleep Duration")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.info("Most students sleep around moderate daily sleep durations.")

    elif chart == "Study Hours Distribution":
        sns.histplot(df["Study_Hours"], bins=20, kde=True, ax=ax)
        ax.set_title("Study Hours Distribution")
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.info("Most students have moderate study hours.")

    elif chart == "Gender Distribution":
        sns.countplot(data=df, x="Gender", ax=ax)
        ax.set_title("Gender Distribution")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.info("Gender distribution is nearly balanced.")

    elif chart == "Department Distribution":
        sns.countplot(data=df, x="Department", ax=ax)
        ax.set_title("Department Distribution")
        ax.set_xlabel("Department")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        st.info("Students are distributed across different academic departments.")

    elif chart == "CGPA by Depression":
        sns.boxplot(data=df, x="Depression", y="CGPA", ax=ax)
        ax.set_title("CGPA by Depression Status")
        st.pyplot(fig)
        st.info("Students with probable depression tend to have lower CGPA compared with healthy students.")

    elif chart == "Sleep Duration by Depression":
        sns.boxplot(data=df, x="Depression", y="Sleep_Duration", ax=ax)
        ax.set_title("Sleep Duration by Depression Status")
        st.pyplot(fig)
        st.info("Sleep duration shows some difference between groups, but the distributions overlap.")

    elif chart == "Social Media Hours by Depression":
        sns.boxplot(data=df, x="Depression", y="Social_Media_Hours", ax=ax)
        ax.set_title("Social Media Hours by Depression Status")
        st.pyplot(fig)
        st.info("Students with probable depression show slightly higher social media usage.")

    elif chart == "Stress Level by Depression":
        sns.boxplot(data=df, x="Depression", y="Stress_Level", ax=ax)
        ax.set_title("Stress Level by Depression Status")
        st.pyplot(fig)
        st.info("Stress level distributions overlap, but some high-stress outliers appear.")

    elif chart == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=["int64", "float64", "bool"])
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        st.info("Most original features show weak linear correlation with Depression.")


# =========================
# MODEL RESULTS
# =========================
elif page == "🤖 Model Results":
    st.title("🤖 Model Results")

    st.markdown("""
    This section summarizes the performance of the three trained machine learning models.
    The models were evaluated using **Accuracy, Precision, Recall, F1-score, and ROC-AUC**.
    """)

    st.dataframe(results, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <h4>🏆 Final Model Selection</h4>
        <p>
        <b>Random Forest</b> was selected as the final model because it achieved the best overall balance
        across the evaluation metrics. It obtained the highest F1-score and ROC-AUC, making it the most
        suitable model for predicting probable student depression.
        </p>
    </div>
    """, unsafe_allow_html=True)

    results_melted = results.melt(
        id_vars="Model",
        value_vars=["F1-score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model Comparison Based on F1-score and ROC-AUC")
    plt.xticks(rotation=20)
    st.pyplot(fig)


# =========================
# PREDICTION
# =========================
elif page == "🔮 Prediction":
    st.title("🔮 Student Depression Prediction")

    st.markdown("""
    Fill in the student information below, choose a machine learning model, and click **Predict**.
    """)

    selected_model = st.selectbox(
        "Choose Prediction Model",
        ["Random Forest", "Logistic Regression", "Logistic Regression + PCA"]
    )

    model = load_model(models[selected_model])

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 24, 21)
        gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
        department = st.selectbox("Department", sorted(df["Department"].unique()))
        cgpa = st.slider("CGPA", 1.5, 4.0, 3.0, 0.01)
        sleep_duration = st.slider("Sleep Duration", 2.0, 12.0, 7.0, 0.1)

    with col2:
        study_hours = st.slider("Study Hours", 0.0, 13.0, 4.0, 0.1)
        social_media_hours = st.slider("Social Media Hours", 0.0, 10.0, 3.0, 0.1)
        physical_activity_hours = st.slider("Physical Activity Hours", 0.0, 2.5, 1.2, 0.1)
        stress_level = st.slider("Stress Level", 1, 10, 4)

    lifestyle_balance = study_hours + physical_activity_hours - social_media_hours
    productive_hours = study_hours + physical_activity_hours
    stress_study = stress_level * study_hours

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Department": [department],
        "CGPA": [cgpa],
        "Sleep_Duration": [sleep_duration],
        "Study_Hours": [study_hours],
        "Social_Media_Hours": [social_media_hours],
        "Stress_Level": [stress_level],
        "Physical_Activity_Hours": [physical_activity_hours],
        "Lifestyle_Balance": [lifestyle_balance],
        "Productive_Hours": [productive_hours],
        "Stress_Study": [stress_study]
    })

    st.subheader("Input Summary")
    st.dataframe(input_data, use_container_width=True)

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="warning-card">
                <h3>⚠️ Probable Depression</h3>
                <p>The selected model predicts that this student may be at risk of depression.</p>
                <p><b>Probability:</b> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-card">
                <h3>✅ Healthy</h3>
                <p>The selected model predicts that this student is unlikely to have probable depression.</p>
                <p><b>Probability of probable depression:</b> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption("This prediction is for educational purposes only and should not be used as a medical diagnosis.")


# =========================
# ABOUT
# =========================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    This project was developed as a machine learning application to predict probable student depression
    using lifestyle and academic factors.

    ### Project Workflow
    - Data loading and understanding
    - Exploratory Data Analysis
    - Feature Engineering
    - Data Preprocessing
    - PCA
    - Machine Learning Modeling
    - Model Evaluation
    - Streamlit Deployment

    ### Models Used
    - Logistic Regression
    - Random Forest
    - Logistic Regression with PCA

    ### Final Model
    Random Forest was selected as the final model because it achieved the best overall performance
    based on the model comparison results.
    """)

    st.warning("This app is for educational purposes only and is not a medical diagnostic tool.")
