import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("ML Assignment 2 – Credit Card Default Prediction")

TARGET = "default.payment.next.month"

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

model_choice = st.selectbox("Select Model", model_names)


uploaded = st.file_uploader("Upload CSV file for prediction", type="csv")

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Check target column
    if TARGET in df.columns:
        y = df[TARGET]
        X = df.drop(columns=[TARGET])
        st.success("Target column detected. Evaluation enabled.")
    else:
        y = None
        X = df
        st.info("Target column not found. Prediction mode.")

    # Cleaning
    X = X.replace("?", np.nan)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0)

    # Load scaler
    scaler = joblib.load("model/scaler.pkl")
    expected_features = scaler.n_features_in_

    X = X.to_numpy()

    # Feature alignment
    if X.shape[1] > expected_features:
        X = X[:, :expected_features]
    elif X.shape[1] < expected_features:
        diff = expected_features - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], diff))])

    X_scaled = scaler.transform(X)

    # Load model
    
    model = joblib.load(f"model/{model_choice}.pkl")
    preds = model.predict(X_scaled)

    st.subheader("Predictions")
    st.dataframe(pd.DataFrame({"Prediction": preds}))

    if y is not None:
        st.subheader("Classification Report")
        st.text(classification_report(y, preds))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, preds))
