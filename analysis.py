# Student Performance Analysis
# Quick exploratory data analysis and prediction model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n_students = 1000

data = pd.DataFrame({
    'attendance': np.random.uniform(60, 100, n_students),
    'study_hours': np.random.uniform(1, 8, n_students),
    'previous_score': np.random.uniform(40, 95, n_students),
    'engagement_score': np.random.uniform(1, 10, n_students)
})

# Create target variable: pass/fail (score > 60)
data['outcome'] = (data['previous_score'] + 
                   data['attendance']*0.3 + 
                   data['study_hours']*2 > 90).astype(int)

print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Train-test split
X = data[['attendance', 'study_hours', 'previous_score', 'engagement_score']]
y = data['outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy:.2%}")
print(f"{'='*50}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

print("\n Analysis complete!")
print("This demonstrates the type of predictive modeling used to")
print("improve student outcomes for 200K+ students in production systems.")
