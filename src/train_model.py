import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression # its a "Probability Machine."
from sklearn.ensemble import RandomForestClassifier #Advanced classifier than LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler #Imports scaling tool, Normalize numeric features
from sklearn.pipeline import Pipeline # Combines steps, Production safety
from sklearn.metrics import confusion_matrix, classification_report
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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

"""
test_size=0.2: means you are using 80% of your data. You take 80% of your data and give it to the AI to "study" (Training set). You hide the remaining 20% to "test" it later (Testing set). For getting better accuracy, always use more data to train the model, less data means less accuracy.

random_state=42: This is just a seed number. It ensures that every time you run the code, you get the same random split, making your results consistent. Why 42? its a comminty standard but we can use any number to get the results consitency, if we don't use a random_state, then result will shuffle, like first time we get 85% accuracy, next time it can be 88% or 82.5% etc. so always use any number.

stratify=Y
Without stratify:
1. Confusion matrix becomes useless
2. Recall becomes meaningless
3. Business decisions become wrong
"""

################ Logistic Regression with scaling #################

# ****** Feature Scaling: **********

""" 
scaling is very important in ML. without it our model quality can't be reliable. 
StandardScaler Transforms values so that: Mean = 0, Standard deviation = 1
"""

scaler = StandardScaler()

# ******* not used below 2 lines becoz now we use Pipeline ******
# X_train_scaled = scaler.fit_transform(X_train) # use 'fit_transform' to train  data
# X_test_scaled = scaler.transform(X_test) #Never fit on test data, Always use 'transform' for test data

# ******* Train the Model: *********

"""
Logistic Regression is a "Probability Machine." Despite the name "Regression," it is used for Classification (Yes/No, Stay/Leave). It calculates the odds. If the probability of an employee leaving is higher than 50%, it classifies them as "Left."
"""

model = LogisticRegression() 

# model.fit(X_train, y_train) # used non-scaled data
# now that we have scaled data, will use scaled data only
# model.fit(X_train_scaled, y_train) 
# ******* not used above 1 lines becoz now we use Pipeline ******


"""
****** This is where the magic happens. ******

The model looks at the 80% of data (the training set) and starts noticing patterns.

Example: It might notice that "People with low salary + low job satisfaction usually = Left."

The word 'fit' means the model is adjusting its internal math to match the historical data as closely as possible.
"""


# ******** Prediction *******

# predections = model.predict(X_test) # used non-scaled test for prediction
# ******* not used below 1 lines becoz now we use Pipeline ******
# predections = model.predict(X_test_scaled) # now that we have scaled test data, will use scaled test data only
""" 
predict: Now, you show the model the 20% of data it has never seen before (X_test). You ask: "Based on what you learned, do you think these employees left?"
"""
# ******** Evaluate *********

# accuracy = accuracy_score(y_test, predections) # used predictions with non-scaled data
# ******* not used below 1 lines becoz now use Pipeline ******
# accuracy = accuracy_score(y_test, predections) # now will use scaled data for predictions
""" 
accuracy_score: You compare the model's guesses (predections) against the actual truth (Y_test).
"""

# print(f"Model Accuracy: {accuracy}")


# *********** Use Logistic Regression Pipeline ***********

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train) #Scales data, Trains model, One call only
predections = pipeline.predict(X_test) # Auto-scales

cm = confusion_matrix(y_test, predections)
cr = classification_report(y_test, predections)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:", cr)

accuracy = accuracy_score(y_test, predections)

print("Logistic Pipeline Accuracy:", accuracy)




# *********** Random Forest(no need of scaling because its an advance library) **************

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
rf_model.fit(X_train, y_train) 
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print("Random Forest Accuracy:", rf_acc)


rf_cm = confusion_matrix(y_test, rf_preds)
print("Random Forest Confusion Matrix:\n", rf_cm)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds))

"""
Random Forest sometime overfitting, i.e gives 1 from all tests because its a voting based algorithm and also its memorizing the data, it means instead of using new data, its using data from its memory and showing results which may faulcy and dangerous.
In this case we use     n_estimators=100, max_depth=5, min_samples_leaf=5 for making it accurate.

And for safer side, we use Cross-Validation to Prevents overfitting, Gives confidence before deployment, Reduces business risk , Industry standard.

Anytime we can use Cross-Validation, its a good practice to make the results validated
"""

# ********************** Cross-Validation Use *************************
rf_cv_scores = cross_val_score(
    rf_model,
    X,
    Y,
    cv=5,
    scoring="recall"
)

print("Random Forest CV Recall scores:", rf_cv_scores)
print("Average CV Recall:", rf_cv_scores.mean())

# save Model and Scaler and Random Forest

# ******* not used below 2 lines becoz now we use Pipeline ******
# MODEL_PATH = BASE_DIR / "models/logistic_model.pkl" 
# SCALER_PATH = BASE_DIR / "models/scaler.pkl" 

PIPELINE_PATH = BASE_DIR / "models/logistic_pipeline.pkl" #
RANDOM_FOREST_PATH = BASE_DIR / "models/random_forest_model.pkl" 

# ******* not used below 2 lines becoz now we use Pipeline ******
# joblib.dump(model, MODEL_PATH)
# joblib.dump(scaler, SCALER_PATH)

joblib.dump(pipeline, PIPELINE_PATH) #dum pipeline itself
joblib.dump(rf_model, RANDOM_FOREST_PATH)


# run the command in CLI 'python src/train_model.py' to train and save the models