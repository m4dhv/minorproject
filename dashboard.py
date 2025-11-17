import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Classification Model Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

DATA_PATH = "tables/pd_prediction.xlsx"   # <-- fixed actual file
df = load_data(DATA_PATH)

st.title("📊 Classification Model Monitoring Dashboard")

# --------------------------
# SIDEBAR SETTINGS
# --------------------------
st.sidebar.header("🔧 Settings")

actual_col = "loan_status"   # FIXED ground truth column
pred_cols = [c for c in df.columns if "pred" in c.lower()]  # auto-detect model prediction cols

pred_col = st.sidebar.selectbox("Select Prediction Column", pred_cols)

# Numeric feature for trend
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_for_trend = st.sidebar.selectbox("Numeric Feature (Trend Chart)", num_cols)

# Categorical feature
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_feature = st.sidebar.selectbox("Categorical Feature (Distribution)", cat_cols)

# --------------------------
# METRICS
# --------------------------
y_true = df[actual_col]
y_pred = df[pred_col]

# KPI Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# KPI Cards Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy*100:.2f}%")
c2.metric("Precision", f"{precision*100:.2f}%")
c3.metric("Recall", f"{recall*100:.2f}%")
c4.metric("F1 Score", f"{f1*100:.2f}%")

# --------------------------
# CLASSIFICATION REPORT
# --------------------------
with st.expander("📄 Classification Report"):
    st.text(classification_report(y_true, y_pred, zero_division=0))

# --------------------------
# CONFUSION MATRIX
# --------------------------
st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# --------------------------
# FEATURE TREND CHART
# --------------------------
st.subheader("📈 Feature Trend (Actual vs Predicted)")

tmp = df.copy()
tmp["bin"] = pd.qcut(tmp[feature_for_trend], q=20, duplicates="drop")

trend = tmp.groupby("bin").agg(
    actual_mean=(actual_col, "mean"),
    pred_mean=(pred_col, "mean")
).reset_index()

fig2 = px.line(trend, x=trend.index, y=["actual_mean", "pred_mean"],
               labels={"value": "Mean", "index": "Feature Bins"})
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# CATEGORICAL DISTRIBUTION
# --------------------------
st.subheader("📊 Categorical Distribution by Actual Class")

fig3 = px.histogram(df, x=cat_feature, color=actual_col, barmode="group")
st.plotly_chart(fig3, use_container_width=True)

# --------------------------
# RAW DATA
# --------------------------
with st.expander("📂 Raw Data Preview"):
    st.dataframe(df)
