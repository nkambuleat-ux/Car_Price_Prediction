import streamlit as st
import pickleimport numpy as np

st.title("Car Price Prediction Application")
st.header("Please complete the details")

#Features used to train the model
vehicle_age = st.number_input("Car age (years)", 0, 30)
transmission_type = st.selectbox("Transmission type", "Automatic", "Manual")
mileage = st.number_input("Car mileage (km/L)", 1, 40)
engine = st.number_input("Car engine capacity (cc)", 100, 10000)
max_power = st.number_input("Maximum power (700kW)", 10, 700)

#Create a button to predict the output
prediction = st.button("Predict")
if prediction == True:
    model = pickle.load(open("C:\Users\ankambule\VS_Code\best_rf_model.pkl", "rb"))
    data = np.array([['vehicle_age', 'transmission_type', 'mileage', 'engine', 'max_power']])
    result = model.predict(data)
    st.success(f"The predicted car price is Rs{result[0]:,.2f}")python -m venvvenv_streamlit