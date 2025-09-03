# utils.py

import pandas as pd
import joblib
from datetime import datetime

# Paths
MODEL_PATH = "models/laptop_price_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"
LOG_FILE = "logs/input_errors.log"

# load model + features
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

def log_error(field, value, message):
    """Save invalid user attempts to a log file."""
    with open(LOG_FILE, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Field: {field}, value: '{value}', Error: {message}\n")


def safe_numeric(prompt, cast_type=float, default=None, min_val=None, max_val=None, field=""):
    """Ask for numeric input safely with range checks + logging."""
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = cast_type(raw)
            if min_val is not None and val < min_val:
                msg = f"Too low (min {min_val})"
                print(f"⚠️ {msg}. Try again.")
                log_error(field, raw, msg)
                continue
            if max_val is not None and val > max_val:
                msg = f"Too high (max {max_val})"
                print(f"⚠️ {msg}. Try again.")
                log_error(field, raw, msg)
                continue
            return val
        except:
            msg = f"Invalid {cast_type.__name__}"
            print(f"⚠️ Please enter a valid {cast_type.__name__}.")
            log_error(field, raw, msg)


def safe_choice(prompt, options, default=None, field=""):
    """Ask for input safely from a set of options, with logging."""
    options_str = ", ".join(options)
    while True:
        raw = input(f"{prompt} ({options_str}): ").strip().title()
        if raw == "" and default is not None:
            return default
        if raw in options:
            return raw
        msg = "Invalid choice"
        print("⚠️ Invalid choice. Please pick from the list.")
        log_error(field, raw, msg)


def get_user_inputs():
    """Collect laptop specifications safely with validations."""
    print("💻 Laptop Price Prediction Demo")
    print("Please enter your laptop specifications.\n")

    companies = ["Dell", "Apple", "Hp", "Lenovo", "Acer", "Asus", "Msi"]
    types = ["Notebook", "Ultrabook", "Gaming", "2 In 1 Convertible", "Workstation"]
    oss = ["Windows 10", "Windows 7", "Macos", "Linux", "No Os"]

    return {
        "Company": safe_choice("Company", companies, default="Dell", field="Company"),
        "TypeName": safe_choice("Type", types, default="Notebook", field="TypeName"),
        "Inches": safe_numeric("Screen Size (inches): ", float, default=15.6, min_val=10, max_val=20, field="Inches"),
        "Ram": safe_numeric("RAM (GB): ", int, default=8, min_val=2, max_val=128, field="Ram"),
        "Weight": safe_numeric("Weight (kg): ", float, default=2.0, min_val=0.5, max_val=5.0, field="Weight"),
        "OpSys": safe_choice("Operating System", oss, default="Windows 10", field="OpSys"),
        "SSD": safe_numeric("SSD size (GB, 0 if none): ", int, default=0, min_val=0, field="SSD"),
        "HDD": safe_numeric("HDD size (GB, 0 if none): ", int, default=0, min_val=0, field="HDD"),
        "Hybrid": safe_numeric("Hybrid storage (GB, 0 if none): ", int, default=0, min_val=0, field="Hybrid"),
        "Flash_Storage": safe_numeric("Flash storage (GB, 0 if none): ", int, default=0, min_val=0, field="Flash_Storage"),
        "Cpu": input("CPU (e.g., Intel Core i5, AMD Ryzen 7): ").strip(),
        "Gpu": input("GPU (e.g., Nvidia GeForce GTX 1050, Intel HD Graphics): ").strip()
    }


def predict_price(sample_dict):
    """Predict laptop price given specs."""
    sample = pd.DataFrame([sample_dict])
    sample_encoded = pd.get_dummies(sample)
    
    # Predict
    sample_encoded = sample_encoded.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(sample_encoded)[0]
    return round(prediction, 2)

