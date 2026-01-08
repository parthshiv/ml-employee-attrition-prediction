import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] 

MODEL_PATH = BASE_DIR / "models/logistic_model.pkl"
model = joblib.load(MODEL_PATH)

features = ['age', 'salary', 'years_at_company', 'job_satisfaction']

coefficient = model.coef_[0]

importance = pd.DataFrame({
    'features': features,
    'coefficient': coefficient
})

importance = importance.sort_values(by="coefficient", ascending=False)

print(importance)