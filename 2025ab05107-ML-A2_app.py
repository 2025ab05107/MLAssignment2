import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("ML Assignment 2 – Classification Models & Evaluation")

# ---------------- Paths ----------------
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

uploaded = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    has_target = TARGET in df.columns

    if has_target:
        y = df[TARGET]
        X = df.drop(columns=[TARGET])
        st.success("Evaluation Mode Enabled")
    else:
        X = df
        y = None
        st.info("Prediction Mode Enabled")

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
        X = np.hstack([X, np.zeros((X.shape[0], expected - X.shape[1]))])

    X_scaled = scaler.transform(X)

    # ---------------- Prediction Mode ----------------
    if not has_target:

        model_name = st.selectbox("Select Model for Prediction", MODELS.keys())
        model = joblib.load(MODELS[model_name])

        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        result = df.copy()
        result["Prediction"] = preds

        if probs is not None:
            result["Probability"] = probs.round(3)

        st.subheader("Prediction Results")
        st.dataframe(result.head(50))

        st.download_button(
            "Download Predictions",
            result.to_csv(index=False).encode(),
            "predictions.csv"
        )

        st.write("Prediction Summary")
        st.write(result["Prediction"].value_counts())

    # ---------------- Evaluation Mode ----------------
    else:

        metrics = {}
        conf_mats = {}

        for name, path in MODELS.items():

            model = joblib.load(path)
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

            metrics[name] = {
                "Accuracy": accuracy_score(y, preds),
                "AUC": roc_auc_score(y, probs) if probs is not None else 0,
                "Precision": precision_score(y, preds),
                "Recall": recall_score(y, preds),
                "F1": f1_score(y, preds),
                "MCC": matthews_corrcoef(y, preds)
            }

            conf_mats[name] = confusion_matrix(y, preds)

        metrics_df = pd.DataFrame(metrics).T.round(4)

        # -------- Metrics Table --------
        st.subheader("📊 Model Evaluation Metrics")
        st.dataframe(metrics_df)

        # -------- Best Model --------
        best_model = metrics_df["F1"].idxmax()
        st.success(f"🏆 Best Performing Model (by F1 Score): {best_model}")

        # -------- Detailed Metrics --------
        st.subheader("📌 Detailed Metrics Per Model")

        for model_name, row in metrics_df.iterrows():
            st.markdown(f"### 🔹 {model_name}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", row["Accuracy"])
            c1.metric("AUC", row["AUC"])
            c2.metric("Precision", row["Precision"])
            c2.metric("Recall", row["Recall"])
            c3.metric("F1 Score", row["F1"])
            c3.metric("MCC", row["MCC"])

        # -------- Confusion Matrices --------
        st.subheader("🧩 Confusion Matrices")

        for name, cm in conf_mats.items():
            st.write(f"### {name}")
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center")
            st.pyplot(fig)

        st.download_button(
            "Download Metrics CSV",
            metrics_df.to_csv().encode(),
            "metrics.csv"
        )

        st.success("Evaluation completed successfully for all models.")
