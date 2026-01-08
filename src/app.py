from fastapi import FastAPI
import joblib
import numpy as np
from pathlib import Path
from schemas.employees import EmployeeInput

app = FastAPI(title="Employee Attrition Prediction API")

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models/logistic_model.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.post("/predict")
def predict_employee(data: EmployeeInput):
    #ML models need 2D numeric arrays.
    input_data = np.array([[
        data.age,
        data.salary,
        data.years_at_company,
        data.job_satisfaction
    ]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    
    return {
        "prediction": int(prediction),
        "result": "Likely to Leave" if prediction == 1 else "Likely to Stay"
    }