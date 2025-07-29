# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import numpy as np

# Load model and data structure
model = joblib.load('best_credit_risk_model.pkl')

st.set_page_config(layout="wide", page_title="Credit Risk Dashboard")

st.title("🏦 Credit Risk Model Dashboard")
st.markdown("This dashboard allows banks and lenders to assess customer credit risk based on recent payment behavior.")

# File Upload
st.sidebar.header("📂 Upload Customer Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load sample/test data
@st.cache_data
def load_sample():
    return pd.read_csv("credit_data.csv").dropna()

df = uploaded_file if uploaded_file else load_sample()

# --- Feature Engineering ---
def preprocess(df):
    # Risk label
    def compute_risk(row):
        missed_count = sum(
            1 for col in ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
            if str(row[col]).strip().lower() == 'missed'
        )
        return 1 if missed_count >= 3 else 0

    df['risk'] = df.apply(compute_risk, axis=1)
    df = df.dropna()
    X = df.drop(columns=['Customer_ID', 'risk'])
    X = pd.get_dummies(X)
    y = df['risk']
    return X, y, df

X, y, raw_df = preprocess(df)

# Show dataframe
if st.checkbox("🔍 Show Preprocessed Data"):
    st.dataframe(X.head())

# --- Risk Prediction ---
st.subheader("📊 Risk Prediction")
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]
df_results = raw_df.copy()
df_results["Predicted Risk"] = y_pred
df_results["Probability"] = y_proba

st.dataframe(df_results[["Customer_ID", "Predicted Risk", "Probability"]].head())

# --- Evaluation ---
st.subheader("📈 Model Evaluation")
col1, col2 = st.columns(2)

with col1:
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)

with col2:
    roc_auc = roc_auc_score(y, y_proba)
    st.metric("ROC-AUC Score", f"{roc_auc:.3f}")

    fpr, tpr, _ = roc_curve(y, y_proba)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label="ROC curve")
    ax2.plot([0, 1], [0, 1], linestyle='--')
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    st.pyplot(fig2)

# --- SHAP Explainability ---
st.subheader("🧠 Feature Importance with SHAP")
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Display SHAP summary plot
fig_shap, ax_shap = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig_shap)
