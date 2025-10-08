import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib

# --- Load your fitted scaler and encoder ---
scaler = joblib.load("scaler.pkl")           # StandardScaler fitted on your training data
y_scaler = joblib.load("y_scaler.pkl")           # StandardScaler fitted on your training data
encoder = joblib.load("encoder.pkl")   # LabelEncoder fitted on your training data

# Columns
numeric_cols = ['vehicle_age', 'mileage', 'engine', 'max_power']
categorical_cols = ['transmission_type']

#Streamlit App
st.title("Car Price Prediction Application")
st.header("Please complete the details")

#Features used to train the model
vehicle_age = st.number_input("Car age (years)", 0, 30)
transmission_type = st.selectbox("Transmission type", ["Automatic", "Manual"])
mileage = st.number_input("Car mileage (km/L)", 1, 40)
engine = st.number_input("Car engine capacity (cc)", 100, 10000)
max_power = st.number_input("Maximum power (700kW)", 10, 700)


# Create dataframe
user_input = pd.DataFrame({
    'vehicle_age': [vehicle_age],
    'transmission_type': [transmission_type],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],    
})

# --- Preprocessing ---
# Scale numeric columns
user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])

# Encode categorical columns
user_input[categorical_cols] = user_input[categorical_cols].apply(lambda col: encoder.transform(col))

st.write("Processed input ready for model:")
st.dataframe(user_input)


# # --- Now you can feed it to your model ---
# model = joblib.load("best_rf_model.pkl")
# prediction = model.predict(user_input)
# st.write(f"The predicted car price is Rs{prediction[0]}")

#Create a button to predict the output
prediction = st.button("Predict")
if prediction == True:
    model = joblib.load(open("best_rf_model.pkl", "rb"))
    # data = np.array([['vehicle_age', 'transmission_type', 'mileage', 'engine', 'max_power']])
    result = model.predict(user_input)
    result_transformed = y_scaler.inverse_transform(result.reshape(1, -1))
    st.success(f"The predicted car price is Rs{result_transformed}")











