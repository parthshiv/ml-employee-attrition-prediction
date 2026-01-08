import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "employees.csv"
data = pd.read_csv(CSV_PATH)

X = data[['age', 'salary', 'years_at_company', 'job_satisfaction']]
Y = data['left']
"""
X (Features): These are your 'clues'. You are looking at an employee's age, how much they earn, how long they've stayed, and how happy they are.

Y (Target): This is the 'answer' you want to predict. In this case, did they leave (1) or stay (0)?
"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
"""
test_size=0.2: means you are using 80% of your data. You take 80% of your data and give it to the AI to "study" (Training set). You hide the remaining 20% to "test" it later (Testing set).

random_state=42: This is just a seed number. It ensures that every time you run the code, you get the same random split, making your results consistent.
"""

model = LogisticRegression() #Logistic Regression is a "Probability Machine." Despite the name "Regression," it is used for Classification (Yes/No, Stay/Leave). It calculates the odds. If the probability of an employee leaving is higher than 50%, it classifies them as "Left."

model.fit(X_train, y_train) 
"""
****** This is where the magic happens. ******

The model looks at the 80% of data (the training set) and starts noticing patterns.

Example: It might notice that "People with low salary + low job satisfaction usually = Left."

The word 'fit' means the model is adjusting its internal math to match the historical data as closely as possible.
"""


predections = model.predict(X_test)
# predict: Now, you show the model the 20% of data it has never seen before (X_test). You ask: "Based on what you learned, do you think these employees left?"

accuracy = accuracy_score(y_test, predections)
# accuracy_score: You compare the model's guesses (predections) against the actual truth (Y_test).

print(f"Model Accuracy: {accuracy}")

MODEL_PATH = BASE_DIR / "models/attrition_model.pkl" 
joblib.dump(model, MODEL_PATH)