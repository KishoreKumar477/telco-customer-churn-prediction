import streamlit as st
import pandas as pd
import joblib

st.title("📊 Telco Customer Churn Prediction")

# Load trained model
model = joblib.load("models/trained/LogisticRegression_best.pkl")

st.header("Customer Information")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

st.subheader("Additional Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

phone_service = st.selectbox("Phone Service", ["Yes", "No"])

multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

if st.button("Predict Churn"):

    data = pd.DataFrame({
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f"⚠ Customer likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer likely to stay (Probability: {1-probability:.2f})")