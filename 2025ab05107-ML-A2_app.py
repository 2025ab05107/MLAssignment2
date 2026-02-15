import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Credit Card Default Prediction")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/UCI_Credit_Card.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show columns (VERY IMPORTANT for debugging)
st.subheader("Available Columns")
st.write(list(df.columns))

# -----------------------------
# Detect Target Column Safely
# -----------------------------
possible_targets = [
    "default.payment.next.month",
    "default payment next month",
    "DEFAULT_PAYMENT_NEXT_MONTH"
]

target = None
for col in df.columns:
    if "default" in col.lower():
        target = col
        break

if target is None:
    st.error("Target column not found!")
    st.stop()

st.success(f"Target column detected: {target}")

X = df.drop(columns=[target])
y = df[target]

# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Feature Values")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(col, float(X[col].mean()))

input_df = pd.DataFrame([input_data])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("Customer WILL default next month")
    else:
        st.success("Customer will NOT default")

    st.write(f"Default Probability: {prob:.2f}")
