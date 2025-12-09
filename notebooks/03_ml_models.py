# 👉 Predict if the CNC machine will fail soon (based on last 30–60 seconds of sensor readings)
# This uses supervised classification because we already created failure_label in Phase 2.

# 📌 1. Import Libraries & Load Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# 1) Quick checks — run these first
print("\n--- 1. Data Inspection ---")
df = pd.read_csv("data/cnc_features.csv", parse_dates=["timestamp"]).sort_values("timestamp")
print("Rows,Cols:", df.shape)
print("Columns:", df.columns.tolist())
print("Nulls:\n", df.isna().sum())
print(df.head())

# 2) Create a predictive target to avoid leakage & Fix tool_usage
print("\n--- 2. Feature Engineering & Target Creation ---")
# If the original raw file has tool_usage, merge it:
raw = pd.read_csv("data/cnc_data_raw.csv", parse_dates=["timestamp"]).sort_values("timestamp")

# Merge tool_usage if missing or just to be safe/consistent, using merge_asof
if "tool_usage" in raw.columns:
    print("Merging tool_usage from raw data...")
    df = pd.merge_asof(df, raw[["timestamp","tool_usage"]], on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=5))
    df["tool_usage"] = df["tool_usage"].ffill().fillna(0).astype(int)

print("Original Failure Label Distribution:", df["failure_label"].value_counts(normalize=True))


# Create forward-looking label: failure within next 10 minutes
N_seconds = 10 * 60  # 10 minutes
df = df.sort_values("timestamp").reset_index(drop=True)

# build a boolean series: is there any failure_label==1 within next N_seconds for each row
future_window = pd.Timedelta(seconds=N_seconds)
# "failure_label" is the CURRENT failure status (from Phase 2)
failure_times = df.loc[df["failure_label"]==1, "timestamp"].reset_index(drop=True)

# Efficient approach: rolling forward using pandas merge_asof
df["failure_in_next_10min"] = 0
if len(failure_times) > 0:
    # for each row find first failure at or after timestamp
    next_failure = pd.merge_asof(
        df[["timestamp"]].rename(columns={"timestamp": "ts"}),
        failure_times.to_frame(name="failure_ts"),
        left_on="ts", 
        right_on="failure_ts", 
        direction="forward"
    )
    df["next_failure_ts"] = next_failure["failure_ts"]
    df["failure_in_next_10min"] = (df["next_failure_ts"] - df["timestamp"]) <= future_window
    df["failure_in_next_10min"] = df["failure_in_next_10min"].fillna(False).astype(int)
else:
    df["failure_in_next_10min"] = 0
    df["next_failure_ts"] = pd.NaT

# drop helper
if "next_failure_ts" in df.columns:
    df = df.drop(columns=["next_failure_ts"])
    
print("Target distribution:", df["failure_in_next_10min"].value_counts(normalize=True))


# 3) Handle NaNs / imputing
print("\n--- 3. Imputation and Clean up ---")
feature_cols = [
    "temperature", "vibration", "speed", "energy",
    "temp_roll_mean_30s", "vib_roll_mean_30s",
    "temp_diff", "speed_pct_change",
    "tool_wear_ind", "tool_usage"
]

# Ensure features exist in df
available_features = [c for c in feature_cols if c in df.columns]
if len(available_features) < len(feature_cols):
    print(f"Warning: Missing features. Expected {feature_cols}, found {available_features}")

# Replace infs
df = df.replace([np.inf, -np.inf], np.nan)

# Impute
imp = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imp.fit_transform(df[available_features]), columns=available_features)
y = df["failure_in_next_10min"].values

# 4) Time-based train-test split (very important)
print("\n--- 4. Train/Test Split (Time-based) ---")
# time split like 80% train by time (no shuffle)
split_idx = int(len(df) * 0.8)

X_train = X_imp.iloc[:split_idx].values
y_train = y[:split_idx]
X_test  = X_imp.iloc[split_idx:].values
y_test  = y[split_idx:]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# 5) Scaling and pipeline + Train Models
print("\n--- 5. Model Training (Random Forest) ---")

# Using pipeline for scaling and model
# Added class_weight="balanced" to handle potential imbalance
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
])

rf_pipe.fit(X_train, y_train)

# 6) Evaluation
print("\n--- 6. Evaluation ---")
y_pred = rf_pipe.predict(X_test)
y_prob = rf_pipe.predict_proba(X_test)[:,1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

try:
    roc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:", roc)
except Exception as e:
    print("ROC AUC could not be calculated (likely only one class in test set):", e)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
# plt.show() # Commented out to avoid blocking execution if running non-interactively

# Feature Importance
print("\n--- Feature Importance ---")
importances = rf_pipe.named_steps["rf"].feature_importances_
indices = np.argsort(importances)[::-1]
for i in indices:
    print(f"{available_features[i]}: {importances[i]:.4f}")

# 7) Save Model
print("\n--- 7. Saving Model ---")
joblib.dump(rf_pipe, "model/final_model.pkl")
print("Model saved to model/final_model.pkl")

# --- Secondary Checks (Optional) ---

# ⭐ MODEL 2 — Anomaly Detection (Isolation Forest)
print("\n--- 8. Anomaly Detection (Isolation Forest) Check ---")
iso = IsolationForest(
    contamination=0.02,
    random_state=42
)
# Fit on full dataset or train set? Usually unsupervised can be full, but let's stick to X_imp
iso.fit(X_imp)
df["anomaly_score"] = iso.decision_function(X_imp)
df["anomaly_label"] = iso.predict(X_imp)
df["anomaly_label"] = df["anomaly_label"].apply(lambda x: 1 if x==-1 else 0)

# Compare anomalies vs actual future failures
print("Crosstab of Anomaly vs Next 10min Failure:")
print(pd.crosstab(df["failure_in_next_10min"], df["anomaly_label"]))

# ⭐ MODEL 3 — Tool Life Prediction (Simple Linear Regression)
if "tool_usage" in df.columns:
    print("\n--- 9. Tool Life Prediction (Linear Regression) ---")
    tool_df = df.groupby("tool_usage")["vibration"].mean().reset_index()
    # Simple Linear Regression
    from sklearn.linear_model import LinearRegression
    model_tool = LinearRegression()
    # Drop NaNs if any in grouped data
    tool_df = tool_df.dropna()
    
    if len(tool_df) > 1:
        model_tool.fit(tool_df[["tool_usage"]], tool_df["vibration"])
        
        future_usage = [[df["tool_usage"].max() + 100]]
        predicted_vibration = model_tool.predict(future_usage)
        print(f"Predicted vibration at usage {future_usage[0][0]}: {predicted_vibration[0]:.4f}")
    else:
        print("Not enough tool usage data for regression.")

print("\nProcessing Complete.")
