import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("ML Assignment 2 – Classification Models & Evaluation")

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

TARGET = "default.payment.next.month"

MODELS = {
    "Logistic Regression": MODEL_DIR / "Logistic Regression.pkl",
    "Decision Tree": MODEL_DIR / "Decision Tree.pkl",
    "KNN": MODEL_DIR / "KNN.pkl",
    "Naive Bayes": MODEL_DIR / "Naive Bayes.pkl",
    "Random Forest": MODEL_DIR / "Random Forest.pkl",
    "XGBoost": MODEL_DIR / "XGBoost.pkl"
}

SCALER = MODEL_DIR / "scaler.pkl"

uploaded = st.file_uploader("Upload CSV Dataset (must contain target column)", type="csv")

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if TARGET not in df.columns:
        st.error("Dataset must contain target column: default.payment.next.month")
        st.stop()

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Cleaning
    X = X.replace("?", np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    scaler = joblib.load(SCALER)
    expected = scaler.n_features_in_

    X = X.to_numpy()

    if X.shape[1] > expected:
        X = X[:, :expected]
    elif X.shape[1] < expected:
        pad = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad))])

    X_scaled = scaler.transform(X)

    metrics = []

    st.subheader("Model Evaluation Metrics")

    for name, path in MODELS.items():

        model = joblib.load(path)
        preds = model.predict(X_scaled)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, probs)
        else:
            auc = "NA"

        metrics.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y, preds), 4),
            "AUC": round(auc, 4) if auc != "NA" else "NA",
            "Precision": round(precision_score(y, preds), 4),
            "Recall": round(recall_score(y, preds), 4),
            "F1 Score": round(f1_score(y, preds), 4),
            "MCC": round(matthews_corrcoef(y, preds), 4)
        })

    metrics_df = pd.DataFrame(metrics)

    st.dataframe(metrics_df)

    st.download_button(
        "Download Metrics CSV",
        metrics_df.to_csv(index=False).encode("utf-8"),
        "metrics.csv",
        "text/csv"
    )

    st.success("All 6 models evaluated on same dataset successfully.")
