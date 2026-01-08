import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] 

MODEL_PATH = BASE_DIR / "models/logistic_model.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

new_employee = np.array([[35, 60000, 0, 3]])

new_employee_scaled = scaler.transform(new_employee)
prediction = model.predict(new_employee_scaled)

if prediction[0] == 1:
    print("Employee is likely to leave")
else:
    print("Employee is likely to stay")
