<<<<<<< HEAD
# Laptop_Price_Prediction_DC

Laptop price prediction using **Machine Learning**. This project predicts the price of laptops based on their specifications using Linear Regression and Random Forest (scikit-learn project).


---

## Project Structure
Laptop_Price_Prediction/
│── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned data
│── models/ # Saved ML models
│── notebooks/ # Jupyter Notebooks
│ ├── 01_data_cleaning.ipynb
│ ├── 02_exploratory_analysis.ipynb
│ └── 03_model_training.ipynb
│── src/
│ ├── utils.py # Helper functions
│ └── predict.py # CLI prediction script
│── logs/ # Invalid user input logs
│── requirements.txt # Project dependencies
│── README.md # Project description

---

## Features
- Data cleaning & preprocessing  
- Exploratory data analysis (EDA)  
- Model training (Linear Regression, Random Forest)  
- Model evaluation (MAE, RMSE, R²)  
- CLI script (`predict.py`) to predict laptop prices  
- Input validation & logging of invalid attempts  

---

## How to Run
1. Clone the repository  
   ```bash
   git clone <repo_url>
   cd Laptop_Price_Prediction

2. Install dependencies 
  - (pip install -r requirements.txt)

3. Run Jupyter notebooks (optional)
  - jupyter notebook

4. Run the prediction script
  - python src/predict.py

---

## Laptop Price Prediction Demo
Please enter your laptop specifications.

Company (e.g., Dell, Apple, HP): Dell
Type (e.g., Gaming, Ultrabook, Notebook): Gaming
Screen Size (inches): 15.6
RAM (GB): 16
Weight (kg): 2.5
Operating System: Windows 10
SSD size (GB, 0 if none): 512
HDD size (GB, 0 if none): 0
Hybrid storage (GB, 0 if none): 0
Flash storage (GB, 0 if none): 0
CPU: Intel Core i7
GPU: Nvidia GeForce GTX 1050

Predicted Laptop Price: €1234.56

---

## Contributors
- Akpan Idongesit Moses        | 23/EG/CO/086 | @
- Sendan Abasiofon Uduak       | 23/EG/CO/087 | @abasiofon-sendan
- Usanga Ofonmbuk Mfon         | 23/EG/CO/092 | @
- Mbikan Gracious              | 23/EG/CO/093 | @
- Okon Godspower Aniedi        | 23/EG/CO/094 | @kompanyokon-hub
- Okechukwu Victory            | 23/EG/CO/095 | @
- Clinton Dominion Chinecherem | 23/EG/CO/096 | @DCmax111
- Usoro Ekemini Sunday         | 23/EG/CO/098 | @
- Aaron Shedrach Puyaighe      | 23/EG/CO/099 | @Shaddie-B
- Israel Emana Jimmy           | 23/EG/CO/101 | @Gmie72
