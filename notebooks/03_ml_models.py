# ================================================
# CNC Predictive Maintenance - ML Training
# ================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ================================================
# 1️⃣ LOAD DATASET
# ================================================

print("\n--- Loading Dataset ---")

df = pd.read_csv("data/cnc_features.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


# ================================================
# 2️⃣ CREATE FUTURE FAILURE TARGET
# ================================================

print("\n--- Creating Future Failure Target ---")

future_window = pd.Timedelta(minutes=10)

failure_times = df.loc[df["failure"] == 1, "timestamp"]

df["failure_in_next_10min"] = 0

if len(failure_times) > 0:

    next_failure = pd.merge_asof(
        df[["timestamp"]],
        failure_times.to_frame(name="failure_time"),
        left_on="timestamp",
        right_on="failure_time",
        direction="forward"
    )

    df["failure_in_next_10min"] = (
        (next_failure["failure_time"] - df["timestamp"]) <= future_window
    ).fillna(False).astype(int)

print("Target Distribution:")
print(df["failure_in_next_10min"].value_counts())


# ================================================
# 3️⃣ FEATURE SELECTION
# ================================================

feature_cols = [
    "temperature",
    "vibration",
    "speed",
    "energy",
    "temp_roll_mean",
    "vib_roll_mean",
    "temp_change",
    "vib_change",
    "machine_stress",
    "status_encoded"
]

X = df[feature_cols]
y = df["failure_in_next_10min"]

print("\nSelected Features:", feature_cols)


# ================================================
# 4️⃣ HANDLE MISSING VALUES
# ================================================

X = X.replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=feature_cols
)


# ================================================
# 5️⃣ TIME BASED TRAIN TEST SPLIT
# ================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ================================================
# 6️⃣ TRAIN MULTIPLE MODELS
# ================================================

print("\n--- Training Models ---")

models = {

    "Logistic Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

    "Random Forest":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42
            ))
        ]),

    "XGBoost":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                eval_metric="logloss",
                random_state=42
            ))
        ])
}

results = {}
best_model = None
best_score = 0

for name, pipe in models.items():

    print(f"\nTraining {name}")

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]

    roc = roc_auc_score(y_test, y_prob)

    results[name] = roc

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc)

    if roc > best_score:
        best_score = roc
        best_model = pipe


# ================================================
# 7️⃣ MODEL COMPARISON
# ================================================

print("\n--- Model Comparison ---")

for model, score in results.items():
    print(f"{model} → ROC-AUC: {score:.3f}")

print("\nBest Model Selected:", best_model)


# ================================================
# 8️⃣ CONFUSION MATRIX
# ================================================

print("\n--- Confusion Matrix ---")

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ================================================
# 9️⃣ ROC CURVE
# ================================================

print("\n--- ROC Curve ---")

y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="Model")
plt.plot([0,1],[0,1],'--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ================================================
# 🔟 FEATURE IMPORTANCE
# ================================================

print("\n--- Feature Importance ---")

model_obj = best_model.named_steps["model"]

if hasattr(model_obj, "feature_importances_"):

    importances = model_obj.feature_importances_

elif hasattr(model_obj, "coef_"):

    importances = np.abs(model_obj.coef_[0])

else:
    importances = None


if importances is not None:

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(importance_df)

    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Feature Importance")
    plt.show()


# ================================================
# 11️⃣ SAVE MODEL + FEATURE LIST
# ================================================

print("\n--- Saving Model ---")

model_package = {
    "model": best_model,
    "features": feature_cols
}

joblib.dump(model_package, "model/final_model.pkl")

print("Model saved successfully → model/final_model.pkl")


# ================================================
# 12️⃣ ANOMALY DETECTION (OPTIONAL)
# ================================================

print("\n--- Isolation Forest Anomaly Detection ---")

iso = IsolationForest(contamination=0.02, random_state=42)

iso.fit(X)

df["anomaly"] = iso.predict(X)

df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

print("\nAnomaly vs Failure Crosstab:")

print(pd.crosstab(df["failure_in_next_10min"], df["anomaly"]))


print("\n✔ ML Training Complete")