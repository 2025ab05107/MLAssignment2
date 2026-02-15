import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

st.title("Heart Disease Classification – ML Assignment 2")

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

model_choice = st.selectbox("Select Model", model_names)

uploaded = st.file_uploader("Upload Test CSV", type="csv")

if uploaded:
    data = pd.read_csv(uploaded)

    # Replace ?
    data.replace("?", np.nan, inplace=True)

    # Convert all to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Fill NaN with 0
    data = data.fillna(0)

    y = data["num"]
    X = data.drop("num", axis=1)

    scaler = joblib.load("model/scaler.pkl")

    required_features = scaler.n_features_in_

    X = X.to_numpy(dtype=float)

    # Fix feature size exactly
    if X.shape[1] > required_features:
        X = X[:, :required_features]
    elif X.shape[1] < required_features:
        pad = required_features - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad))])

    X = scaler.transform(X)

    model = joblib.load(f"model/{model_choice}.pkl")

    preds = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, preds))
