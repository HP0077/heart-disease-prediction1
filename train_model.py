import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset.csv")

# Remove unnecessary columns (if present)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Ensure the target column exists
if "TenYearCHD" not in df.columns:
    raise ValueError("Error: 'TenYearCHD' column not found in dataset!")

# Define features and target
X = df.drop(columns=["TenYearCHD"])  # Features
y = df["TenYearCHD"]  # Target variable

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for use in the Flask app
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate model performance
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model training complete. Best parameters:", grid_search.best_params_)
print("Model and scaler saved successfully.")

