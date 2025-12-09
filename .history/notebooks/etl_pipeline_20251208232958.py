import pandas as pd
import numpy as np

# ------------------------------------
# LOAD RAW DATA
# ------------------------------------
df = pd.read_csv("data/cnc_data_raw.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)


# ------------------------------------
# CLEANING
# ------------------------------------

# 1. Remove missing values
df = df.dropna()

# 2. Clip unrealistic values
df["temperature"] = df["temperature"].clip(20, 90)
df["vibration"] = df["vibration"].clip(0.3, 7)
df["speed"] = df["speed"].clip(0, 6000)
df["energy"] = df["energy"].clip(0, 200)

# 3. Smooth extreme spikes
df["temperature_smooth"] = df["temperature"].rolling(window=3, min_periods=1).mean()
df["vibration_smooth"]  = df["vibration"].rolling(window=3, min_periods=1).mean()


# ------------------------------------
# FEATURE ENGINEERING
# ------------------------------------

# 1. Rolling averages (30-second window)
ROLLING_WINDOW = 3  # since interval = 10 sec → 3 rows = 30 sec

df["temp_roll_mean_30s"] = df["temperature_smooth"].rolling(ROLLING_WINDOW).mean()
df["vib_roll_mean_30s"]  = df["vibration_smooth"].rolling(ROLLING_WINDOW).mean()


# 2. Temperature difference
df["temp_diff"] = df["temperature_smooth"].diff().fillna(0)


# 3. Speed % change
df["speed_pct_change"] = df["speed"].pct_change().fillna(0) * 100


# 4. Tool wear indicator (normalized)
max_tool_usage = df["tool_usage"].max()
df["tool_wear_ind"] = df["tool_usage"] / max_tool_usage


# ------------------------------------
# FAILURE LABEL CREATION
# ------------------------------------
df["failure_label"] = np.where(
    (df["temperature_smooth"] > 80) | (df["vibration_smooth"] > 4),
    1,
    0
)


# ------------------------------------
# FINAL FEATURE STORE
# ------------------------------------

feature_cols = [
    "timestamp",
    "temperature_smooth",
    "vibration_smooth",
    "speed",
    "energy",
    "temp_roll_mean_30s",
    "vib_roll_mean_30s",
    "temp_diff",
    "speed_pct_change",
    "tool_wear_ind",
    "failure_label"
]

df_features = df[feature_cols]

df_features.rename(columns={
    "temperature_smooth": "temperature",
    "vibration_smooth": "vibration"
}, inplace=True)


# ------------------------------------
# SAVE DATASET
# ------------------------------------
df_features.to_csv("data/cnc_features.csv", index=False)

import sqlite3
conn = sqlite3.connect("data/cnc_features.sqlite")
df_features.to_sql("cnc_features", conn, if_exists="replace", index=False)
conn.close()

print("ETL + Feature Engineering Completed!")
print("Rows:", len(df_features))
print(df_features.head())
