import streamlit as st 
import pickle 
import joblib 
import numpy as np 
import pandas as pd 
import requests 
import io 

# Open the pickle file in read-binary mode 
with open("scaler_rev2.pkl", "rb") as file: 
    scaler = pickle.load(file) 
# StandardScaler fitted on training numeric columns 
with open("y_scaler.pkl", "rb") as file: 
    y_scaler = pickle.load(file) 
    
# Scaler fitted on target variable 
with open("encoder.pkl", "rb") as file: 
    encoder = pickle.load(file) 

# LabelEncoder fitted on transmission type 
# Load the model from Google Drive 

url = "https://drive.google.com/uc?export=download&id=1MzonX_w_KCmIgqcmFWqtxFXgvoZ0GRYn" 

response = requests.get(url) 
model = joblib.load(io.BytesIO(response.content)) 

# Streamlit UI 
st.title("Car Price Prediction Application") 
st.header("Please complete the details below") 

# Feature inputs 
vehicle_age = st.number_input("Car age (years)", 0, 30, value=5) 
transmission_type = st.selectbox("Transmission type", ["Automatic", "Manual"]) 
mileage = st.number_input("Car mileage (km/L)", 1, 40, value=15) 
engine = st.number_input("Car engine capacity (cc)", 100, 10000, value=1500) 
max_power = st.number_input("Maximum power (kW)", 10, 700, value=100) 

# Create a dataframe with user input 
user_input = pd.DataFrame({ 'vehicle_age': [vehicle_age], 'transmission_type': [transmission_type], 'mileage': [mileage], 'engine': [engine], 'max_power': [max_power], }) 

# --- Preprocessing --- 
# Apply log( x + 1 ) to numeric columns to avoid log(0) 
numeric_cols = ["vehicle_age", "mileage", "engine", "max_power"] 
user_input[numeric_cols] = np.log1p(user_input[numeric_cols]) 

# Transform numeric features using the fitted scaler 
user_input[numeric_cols] = scaler.transform(user_input[numeric_cols].values) 

# Encode categorical feature using the fitted encoder 
user_input['transmission_type'] = encoder.transform(user_input['transmission_type'].values) 

# Ensure column order matches training 
user_input = user_input["transmission_type"] 
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



