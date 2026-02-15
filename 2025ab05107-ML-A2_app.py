import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Credit Card Default Prediction – ML Assignment 2")

# -----------------------------
# Configuration
# -----------------------------
TARGET_COL = "default.payment.next.month"

MODEL_MAP = {
    "Logistic Regression": "Logistic Regression.pkl",
    "Decision Tree": "Decision Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive Bayes.pkl",
    "Random Forest": "Random Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

# -----------------------------
# Model selection
# -----------------------------
model_name = st.selectbox("Select Model", list(MODEL_MAP.keys()))

uploaded_file = st.file_uploader("Upload Test CSV File", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # -----------------------------
    # Separate target if present
    # -----------------------------
    if TARGET_COL in data.columns:
        y_true = data[TARGET_COL]
        X = data.drop(columns=[TARGET_COL])
        st.success("Target column found. Evaluation mode enabled.")
    else:
        y_true = None
        X = data
        st.info("Target column not found. Prediction-only mode enabled.")

    # -----------------------------
    # Data cleaning
    # -----------------------------
    X = X.replace("?", np.nan)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    # -----------------------------
    # Load scaler
    # -----------------------------
    scaler = joblib.load("model/scaler.pkl")
    required_features = scaler.n_features_in_

    X = X.to_numpy(dtype=float)

    # Fix feature mismatch safely
    if X.shape[1] > required_features:
        X = X[:, :required_features]
    elif X.shape[1] < required_features:
        pad = required_features - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad))])

    X_scaled = scaler.transform(X)

    # -----------------------------
    # Load model
    # -----------------------------
    model = joblib.load(f"model/{MODEL_MAP[model_name]}")

    preds = model.predict(X_scaled)

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Predictions")
    output_df = pd.DataFrame({"Prediction": preds})
    st.dataframe(output_df)

    # -----------------------------
    # Evaluation (only if target exists)
    # -----------------------------
    if y_true is not None:
        st.subheader("Classification Report")
        st.text(classification_report(y_true, preds))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, preds))
