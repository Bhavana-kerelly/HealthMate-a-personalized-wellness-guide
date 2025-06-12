import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset/nutritional_dataset.csv")

# Encode categorical features
label_encoders = {}
categorical_columns = ['Gender', 'Physical Activity Level', 'Eating Habits', 'Medical Conditions', 'Medication Usage', 'Stress Level', 'Deficiency']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and target
X = df[['Age', 'Gender', 'BMI', 'Daily Water Intake (L)', 'Physical Activity Level', 'Eating Habits', 'Medical Conditions', 'Medication Usage', 'Sleep Hours', 'Stress Level']]
y_svm = df['Deficiency_Detected'].map({'Yes': 1, 'No': 0})  # Binary classification
y_xgb = df['Deficiency']  # Multi-class classification

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled, y_svm, test_size=0.2, random_state=42)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_scaled, y_xgb, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_svm, y_train_svm)

# Train XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train_xgb, y_train_xgb)

# Save models
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(xgb_model, "models/xgboost_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

# Testing and evaluation
svm_predictions = svm_model.predict(X_test_svm)
xgb_predictions = xgb_model.predict(X_test_xgb)

print("SVM Model Accuracy:", accuracy_score(y_test_svm, svm_predictions))
print("SVM Classification Report:\n", classification_report(y_test_svm, svm_predictions))

print("XGBoost Model Accuracy:", accuracy_score(y_test_xgb, xgb_predictions))
print("XGBoost Classification Report:\n", classification_report(y_test_xgb, xgb_predictions))
