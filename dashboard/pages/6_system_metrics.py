import streamlit as st
import pandas as pd
from components.api_client import check_api_health
from components.styles import load_css

load_css()

st.title("⚙ System Metrics")

# --------------------------------
# API Health
# --------------------------------

st.subheader("API Status")

health = check_api_health()

if health.get("status") == "ok":
    st.success("API Server Running")
else:
    st.error("API Offline")

# --------------------------------
# Dataset Stats
# --------------------------------

df = pd.read_csv("data/cnc_features.csv")

st.subheader("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", len(df))
col2.metric("Total Features", len(df.columns))
col3.metric("Failures", int(df["failure"].sum()))

# --------------------------------
# Feature Overview
# --------------------------------

st.subheader("Dataset Columns")

st.dataframe(pd.DataFrame({
    "Column": df.columns
}))

# --------------------------------
# Model Info
# --------------------------------

st.subheader("Model Information")

st.info(
"""
Model Type: RandomForest / XGBoost  
Purpose: Predict CNC machine failure probability  
Input Features:
- temperature
- vibration
- speed
- energy
- machine_stress
"""
)