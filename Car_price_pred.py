import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import gdown

# --- Load preprocessing objects ---
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("y_scaler.pkl", "rb") as f:
    y_scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# --- Load model from Google Drive ---
url = "https://drive.google.com/uc?id=1MzonX_w_KCmIgqcmFWqtxFXgvoZ0GRYn"
output = "model.pkl"
gdown.download(url, output, quiet=False)
model = joblib.load(output)

# --- Streamlit UI ---
st.title("Car Price Prediction Application")
st.header("Please complete the details below")

vehicle_age = st.number_input("Car age (years)", 0, 30, value=5)
transmission_type = st.selectbox("Transmission type", ["Automatic", "Manual"])
mileage = st.number_input("Car mileage (km/L)", 1, 40, value=15)
engine = st.number_input("Car engine capacity (cc)", 100, 10000, value=1500)
max_power = st.number_input("Maximum power (kW)", 10, 700, value=100)

# --- Prepare user input ---
user_input = pd.DataFrame({
    "vehicle_age": [vehicle_age],
    "transmission_type": [transmission_type],
    "mileage": [mileage],
    "engine": [engine],
    "max_power": [max_power],
})

numeric_cols = ["vehicle_age", "mileage", "engine", "max_power"]

# Apply log(x+1)
user_input[numeric_cols] = np.log1p(user_input[numeric_cols])

# Scale numeric columns
user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])

# Encode categorical column
user_input["transmission_type"] = encoder.transform(user_input["transmission_type"].values)

st.write("Processed input ready for model:")
st.dataframe(user_input)

# --- Prediction ---
if st.button("Predict"):
    result = model.predict(user_input)
    result_transformed = y_scaler.inverse_transform(result.reshape(-1, 1))
    final_prediction = np.expm1(result_transformed).flatten()[0]
    st.success(f"The predicted car price is Rs {final_prediction:,.2f}")
















