# 👉 Predict if the CNC machine will fail soon (based on last 30–60 seconds of sensor readings)

# This uses supervised classification because we already created failure_label in Phase 2.

# 📌 1. Import Libraries & Load Dataset

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
)
# ⭐ 3. Train Models
# Model A — Logistic Regression (Baseline Model)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)

# Model B — Random Forest (Most Important Model)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

# Model C — XGBoost (Optional but strong)
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

# 📌 Evaluation Function
def evaluate_model(name, y_true, y_pred, y_prob):
    print("\n====================")
    print("Model:", name)
    print("====================")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_true, y_prob))
# ✔ Evaluate Models
evaluate_model("Logistic Regression", y_test, y_pred_lr, log_reg.predict_proba(X_test)[:,1])
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)

# ⭐ 5. Confusion Matrix (Must include in Report)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()