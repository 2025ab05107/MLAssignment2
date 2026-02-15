import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 – Credit Card Default Prediction")

# -----------------------------------
# Absolute project paths (CRITICAL)
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

TARGET = "default.payment.next.month"

MODELS = {
    "Logistic Regression": MODEL_DIR / "Logistic Regression.pkl",
    "Decision Tree": MODEL_DIR / "Decision Tree.pkl",
    "KNN": MODEL_DIR / "KNN.pkl",
    "Naive Bayes": MODEL_DIR / "Naive Bayes.pkl",
    "XGBoost": MODEL_DIR / "XGBoost.pkl"
}

SCALER_PATH = MODEL_DIR / "scaler.pkl"

# -----------------------------------
model_choice = st.selectbox("Select Model", MODELS.keys())

uploaded = st.file_uploader("Upload CSV file", type="csv")

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Target detection
    if TARGET in df.columns:
        y = df[TARGET]
        X = df.drop(columns=[TARGET])
        st.success("Target column detected – evaluation enabled.")
    else:
        y = None
        X = df
        st.info("Prediction mode (no target column).")

    # Cleaning
    X = X.replace("?", np.nan)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0)

    # -----------------------------------
    # Load scaler safely
    # -----------------------------------
    scaler = joblib.load(SCALER_PATH)
    expected_features = scaler.n_features_in_

    X = X.to_numpy()

    # Feature alignment
    if X.shape[1] > expected_features:
        X = X[:, :expected_features]
    elif X.shape[1] < expected_features:
        pad = expected_features - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad))])

    X_scaled = scaler.transform(X)

    # -----------------------------------
    # Load selected model
    # -----------------------------------
    model = joblib.load(MODELS[model_choice])

    # -----------------------------
    # Predictions + Probabilities
    # -----------------------------
    preds = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
       probs = model.predict_proba(X_scaled)[:, 1]
    else:
       probs = None

    result_df = df.copy()
    result_df["Prediction"] = preds

    if probs is not None:
       result_df["Probability"] = probs.round(3)

    st.subheader("Prediction Results")
    st.dataframe(result_df.head(50))

    # Download button
    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )

    # Summary counts
    st.subheader("Prediction Summary")
    st.write(result_df["Prediction"].value_counts())

    # -----------------------------------
    # Metrics (if target exists)
    # -----------------------------------
    if y is not None:
        st.subheader("Classification Report")
        st.text(classification_report(y, preds))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, preds))
