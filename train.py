"""
Training script for Predict whether an employee will leave the company using demographics, salary, tenure, education, an...

This script trains a classification model using scikit-learn.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score

# Load dataset
# TODO: Update this path to your dataset
df = pd.read_csv('data.csv')

# Set target column
# TODO: Update this to your target column name
target_column = 'Attrition'

# Prepare data
X = df.drop(columns=[target_column])
y = df[target_column]

# Handle categorical features
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.mean())

# Encode target for classification
if 'classification' == 'classification' and y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
# TODO: Choose your model (RandomForest, GradientBoosting, LogisticRegression, etc.)
if 'classification' == 'classification':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train_scaled, y_train)

# Evaluate
if 'classification' == 'classification':
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {score:.3f}")
else:
    y_pred = model.predict(X_test_scaled)
    score = r2_score(y_test, y_pred)
    print(f"R² Score: {score:.3f}")

# Save model and scaler
with open('classification_model_20261030_164534.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'label_encoder': le if 'classification' == 'classification' and 'le' in locals() else None
    }, f)

print(f"Model trained and saved to classification_model_20261030_164534.pkl")
