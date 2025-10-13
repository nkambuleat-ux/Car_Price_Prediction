import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import requests
import io


# --- Load your fitted scaler, y_scaler, encoder and model ---
scaler = joblib.load("scaler.pkl")          # StandardScaler fitted on training numeric columns
y_scaler = joblib.load("y_scaler.pkl")      # Scaler fitted on target variable
encoder = joblib.load("encoder.pkl")        # LabelEncoder fitted on transmission type

#load the model
url = "https://drive.google.com/uc?export=download&id=1EI0zukjBbhKcxTclqNQdUODSiV1fs7NB"
response = requests.get(url)
model = joblib.load(io.BytesIO(response.content))


# Columns
with open('dummy_columns.pkl', 'rb') as f:
    dummy_columns = pickle.load(f)
numeric_cols = ['vehicle_age', 'mileage', 'engine', 'max_power']


# --- Streamlit UI ---
st.title("Car Price Prediction Application")
st.header("Please complete the details below")

# Feature inputs
fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
vehicle_age = st.number_input("Car age (years)", 0, 30, value=5)
transmission_type = st.selectbox("Transmission type", ["Automatic", "Manual"])
mileage = st.number_input("Car mileage (km/L)", 1, 40, value=15)
engine = st.number_input("Car engine capacity (cc)", 100, 10000, value=1500)
max_power = st.number_input("Maximum power (kW)", 10, 700, value=100)


# Create a dataframe with user input
user_input = pd.DataFrame({
    'vehicle_age': [vehicle_age],
    'transmission_type': [transmission_type],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
})

# --- Preprocessing ---
# Apply log( x + 1 ) to numeric columns to avoid log(0)
user_input[numeric_cols] = np.log1p(user_input[numeric_cols])

# Transform numeric features using the **already fitted** scaler
user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])

# Encode categorical feature using the fitted encoder
user_input['transmission_type'] = encoder.transform(user_input['transmission_type'])

# Add other categorical features (fuel_type) to user_input
user_input['fuel_type'] = fuel_type

# Apply get_dummies to handle one-hot encoding
user_input = pd.get_dummies(user_input)


# Align with training dummy columns
for col in dummy_columns:
    if col not in user_input:
        user_input[col] = 0


# # Ensure column order matches training
# user_input = user_input[dummy_columns]

st.write("Processed input ready for model:")
st.dataframe(user_input)

# --- Prediction ---
if st.button("Predict"):
    result = model.predict(user_input)
    
    # Inverse transform the predicted (scaled) value back to original
    result_transformed = y_scaler.inverse_transform(result.reshape(-1, 1))
    
    # If the target was also log-transformed during training, reverse it
    final_prediction = np.expm1(result_transformed).flatten()[0]
    
    st.success(f"The predicted car price is Rs {final_prediction:,.2f}")









