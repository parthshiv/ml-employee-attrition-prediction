# Employee Attrition Prediction System

An end-to-end machine learning project that predicts employee attrition using real-world HR data.  
The project focuses on **business-aware ML**, **proper evaluation**, and **production-readiness**, not just model accuracy.

---

## Problem Statement

Employee attrition is costly for organizations.  
The goal of this project is to **identify employees at risk of leaving early**, allowing HR teams to take proactive retention actions.

Key business priorities:

- Missing a leaving employee is very costly
- False alarms are acceptable
- Model decisions must be explainable
- Performance must be stable, not accidental

---

## Project Features

- Feature engineering based on real HR logic
- Business-aware evaluation (recall-focused)
- Logistic Regression baseline with pipeline
- Random Forest with overfitting control
- Confusion matrix and classification report
- Cross-validation for model stability
- Comparison of champion vs challenger models
- Production-style project structure

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI (for inference API)
- Git & GitHub

---

## Project Structure

```note
employee-attrition-ml/
│
├── data/
│ └── employees.csv
│
├── src/
│ ├── train_model.py
│ ├── predict_employee.py
│ ├── interpret_model.py
│ └── app.py
│
├── models/
│ ├── logistic_pipeline.pkl
│ └── random_forest_model.pkl
│
├── requirements.txt
└── README.md
```

---

## Feature Engineering

The project uses engineered features such as:

- Salary ratio (relative compensation)
- Tenure buckets (new, mid, senior)
- Job satisfaction signals

These features reflect **employee behavior**, not just raw values.

---

## Model Training

Two models are trained and compared:

### Logistic Regression (Pipeline)

- Uses StandardScaler + Logistic Regression
- Highly interpretable
- Strong recall for employees who leave
- Suitable for regulated HR environments

### Random Forest

- Captures complex patterns
- Regularized to avoid overfitting
- Evaluated using cross-validation

---

## Evaluation Strategy

Accuracy is **not trusted alone**.

The project emphasizes:

- Confusion Matrix
- Precision & Recall
- Recall for leavers (Class 1) as the primary metric
- Cross-validation to ensure stability

---

## Key Results

- Logistic Regression achieved ~96% accuracy with perfect recall for leavers
- Random Forest achieved perfect test accuracy and ~98% recall in cross-validation
- No missed leavers in final evaluation
- Business risk minimized

---

## Business Interpretation

- False negatives (missed leavers) are the highest risk
- False positives are acceptable
- Logistic Regression is suitable as a production "champion"
- Random Forest can act as a "challenger" model

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python src/train_model.py
```

Run prediction API:

```bash
uvicorn src.app:app --reload
```

Open API docs:

```bash
http://127.0.0.1:8000/docs
```

## Future Improvements

- Threshold tuning based on business cost
- Model monitoring and drift detection
- Automated retraining pipelines
- LLM-powered explanations for HR users

## Key Learning Outcomes

- Business-aware ML design
- Proper model evaluation
- Overfitting control
- Cross-validation for stability
- Production-ready ML mindset

## Disclaimer

This project is for educational and demonstration purposes and uses synthetic or sample HR data.
