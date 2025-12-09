👉 Predict if the CNC machine will fail soon (based on last 30–60 seconds of sensor readings)

This uses supervised classification because we already created failure_label in Phase 2.

📌 1. Import Libraries & Load Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cnc_features.csv")
# ⭐ 2. Train-Test Split
# Choose features and target variable for modeling.

feature_cols = [
    "temperature", "vibration", "speed", "energy",
    "temp_roll_mean_30s", "vib_roll_mean_30s",
    "temp_diff", "speed_pct_change",
    "tool_wear_ind"
]

X = df[feature_cols]
y = df["failure_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42