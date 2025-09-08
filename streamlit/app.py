import streamlit as st
import pandas as pd
import joblib
from utils import model, feature_names, preprocess_input, predict_price, log_event

# Load the trained model + scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
LOG_FILE = "logs/input_errors.log"

st.title("Laptop Price Prediction App")
st.write("Enter your laptop specifications to predict the price.")

# User inputs
company = st.selectbox("Company", [
    "Dell", 
    "HP", 
    "Lenovo", 
    "Apple", 
    "Asus", 
    "Acer", 
    "MSI",
    "Samsung",
    "Toshiba",
    "Huawei",
    "Microsoft",
    "Other"
    ])
typename = st.selectbox("Type", [
    "Ultrabook", 
    "Notebook", 
    "Gaming", 
    "2 in 1 Convertible", 
    "Workstation",
    "Netbook"
    ])
cpu = st.selectbox("CPU Brand", [
    "Intel",
    "AMD", 
    "Other"
    ])
gpu = st.selectbox("GPU Brand", [
    "Intel", 
    "Nvidia", 
    "AMD", 
    "Other"
    ])
opsys = st.selectbox("Operating System", [
    "Windows 11",
    "Windows 10", 
    "Windows 7", 
    "MacOS", 
    "Linux", 
    "No OS"
    ])

ram = st.slider("RAM (GB)", min_value=2, max_value=128, step=2, value=8)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
ssd = st.number_input("SSD size (GB)", min_value=0, max_value=4000, value=512, step=128)
hdd = st.number_input("HDD size (GB)", min_value=0, max_value=6000, value=0, step=500)
flash = st.number_input("Flash Storage (GB)", min_value=0, max_value=1000, value=0, step=128)
hybrid = st.number_input("Hybrid storage (GB)", min_value=0, max_value=2000, value=0, step=500)
inches = st.slider("Screen Size (inches)", min_value=10.0, max_value=20.0, step=0.1, value=12.0)

# Normalizing Company Casing
if company == "HP": company = "Hp"
if company == "MSI": company = "Msi"

# Normalize OS
if opsys == "Windows 11": opsys = "Windows 10"

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

        st.write("Encoded sample shape:", input_data.shape)
        st.write("Model expects:", len(feature_names), "features")
        missing = set(feature_names) - set(input_data.columns)
        extra = set(input_data.columns) - set(feature_names)
        st.write("Missing features:", missing)
        st.write("Unexpected extra features:", extra)

        with st.spinner("Calculating..."):
            prediction = predict_price(model, input_data)
            # prediction = model.predict(input_data)[0]
        
        # Sanity bounds (based on your dataset: ₤100 - ₤6000)
        if prediction < 100 or prediction > 6000:
            log_event("warn", "Prediction", str(received_data.to_dict()), f"Unrealistic prediction: {prediction:.2f}")
            st.warning(f"Prediction seems unrealistic (₤{prediction:.2f}). Please re-check your inputs.")
            
        st.toast("Prediction ready!")
        st.success(f"Estimated Price: ₤{prediction:.4f}")
    except Exception as e:
        log_event("error", "StreamlitApp", str(received_data.to_dict()), f"Prediction failed: {e}")
        st.error(f"Something went wrong while generating the prediction. Please check your inputs and try again.")

