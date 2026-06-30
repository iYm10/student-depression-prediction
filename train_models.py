import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("student_lifestyle_100k.csv")

df = pd.read_csv(DATA_PATH)
df["Physical_Activity_Hours"] = df["Physical_Activity"] / 60
df["Lifestyle_Balance"] = df["Study_Hours"] + df["Physical_Activity_Hours"] - df["Social_Media_Hours"]
df["Productive_Hours"] = df["Study_Hours"] + df["Physical_Activity_Hours"]
df["Stress_Study"] = df["Stress_Level"] * df["Study_Hours"]

df_ml = df.drop_duplicates().drop(columns=["Student_ID", "Physical_Activity"])
X = df_ml.drop("Depression", axis=1)
y = df_ml["Depression"].astype(int)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )),
    ]),
    "Logistic Regression + PCA": Pipeline([
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=0.90)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
}

file_names = {
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression + PCA": "logistic_pca_model.pkl",
}

metrics = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    })
    joblib.dump(model, file_names[name])

joblib.dump(models["Random Forest"], "best_student_depression_model.pkl")
pd.DataFrame(metrics).drop(columns=["Confusion Matrix"]).to_csv("model_results.csv", index=False)
with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Models trained and saved successfully.")
