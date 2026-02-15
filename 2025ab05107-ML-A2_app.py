import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

st.title("ML Assignment 2 – Classification App")

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

    X = data.drop("target", axis=1)
    y = data["target"]

    scaler = joblib.load("model/scaler.pkl")
    X = scaler.transform(X)

    model = joblib.load(f"model/{model_choice}.pkl")

    preds = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, preds))
