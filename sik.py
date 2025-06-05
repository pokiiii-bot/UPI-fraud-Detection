import streamlit as st
import joblib

# Load the trained model
model = joblib.load("upi_fraud_model.pkl")  # Make sure this file is in the same folder

# Encoding maps (update as per your dataset)
upi_type_map = {'P2P': 0, 'P2M': 1, 'P2B': 2}
device_type_map = {'Mobile': 0, 'Web': 1, 'POS': 2}

st.title("🚦 UPI Fraud Detection Dashboard")

# Input fields
amount = st.number_input("Amount", min_value=0.0, step=0.01)
merchant_id = st.number_input("Merchant ID (numeric)", min_value=0, step=1)
time = st.number_input("Time (e.g., 14.5)", min_value=0.0, max_value=24.0, step=0.1)
upi_type = st.selectbox("UPI Type", options=list(upi_type_map.keys()))
device_type = st.selectbox("Device Type", options=list(device_type_map.keys()))
txn_day = st.slider("Transaction Day (0=Mon, 6=Sun)", min_value=0, max_value=6, step=1)
txn_hour = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, step=1)
customer_age = st.number_input("Customer Age", min_value=0, step=1)
geo_distance = st.number_input("Geo Distance", min_value=0.0, step=0.1)
txn_velocity = st.number_input("Transaction Velocity", min_value=0.0, step=0.1)

if st.button("Predict Fraud"):
    features = [[
        amount,
        merchant_id,
        time,
        upi_type_map[upi_type],
        device_type_map[device_type],
        txn_day,
        txn_hour,
        customer_age,
        geo_distance,
        txn_velocity
    ]]

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {prob:.2f}")
