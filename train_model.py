import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("credit_data.csv")  # or your correct path

# 🔧 Define your target and features
target = 'Target'  # replace with actual target column
X = df.drop(columns=[target])
y = df[target]

# 💡 Optional: one-hot encode categorical features
X = pd.get_dummies(X)

# 🚀 Save feature names for Streamlit compatibility
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

# 💾 Save model
joblib.dump(model, "best_credit_risk_model.pkl")

# 💾 Optionally save the scaler if you’ll use it in your Streamlit app
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and feature_names.pkl saved.")
