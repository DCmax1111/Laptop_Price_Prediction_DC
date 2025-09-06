import streamlit as st
import pandas as pd
import joblib
from utils import model, feature_names, preprocess_input

# Load the trained model + scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Laptop Price Prediction App")
st.write("Enter your laptop specifications to predict the price.")

# User inputs
company = st.selectbox("company", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "MSI", "Other"])
typename = st.selectbox("Type", ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible", "Workstation"])
cpu = st.selectbox("CPU Brand", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD", "Other"])
gpu = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD", "Other"])
opsys = st.selectbox("Operating System", ["Windows 10", "Windows 7", "MacOS", "Linux", "No OS/Other"])

ram = st.slider("RAM (GB)", 2, 64, 8, 128)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
ssd = st.number_input("SSD size (GB)", min_value=0, max_value=2000, value=0, step=128)
hdd = st.number_input("HDD size (GB)", min_value=0, max_value=6000, value=0, step=500)
flash = st.number_input("HDD size (GB)", min_value=0, max_value=64, value=0, step=16)
hybrid = st.number_input("Hybrid storage (GB)", min_value=0, max_value=2000, value=0, step=128)
inches = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6, 20.0)

# Predict Button
if st.button("Predict Price"):
    try:
        # Build input DataFrame
        received_data = pd.DataFrame([{
            "Company": company,
            "TypeName": typename,
            "Inches": inches,
            "Ram": ram,
            "Weight": weight,
            "OpSys": opsys,
            "SSD": ssd,
            "HDD": hdd,
            "Hybrid": hybrid,
            "Flash_Storage": flash,
            "Cpu_brand": cpu,
            "Gpu_brand": gpu
        }])

        # NOTE: Applying the same preprocessing (target encoding + scaling) here.
        input_data = preprocess_input(received_data, feature_names)
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Price: ₤{prediction:.4f}")
    except Exception as e:
        st.error("Something went wrong while generating the prediction. Please check your inputs and try again.")