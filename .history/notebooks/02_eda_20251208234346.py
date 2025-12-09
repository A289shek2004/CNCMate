# ⭐ STEP 1 — Import Libraries & Load Feature Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cnc_features.csv")
# Convert timestamp:

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ⭐ STEP 2 — Basic Descriptive Statistics
df.describe().T
# ⭐ STEP 3 — Time-Series Plots (Temperature & Vibration Trends)

plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["temperature"], label="Temperature (°C)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Trend Over Time")
plt.legend()
plt.show()

# Vibration Trend Plot
plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["vibration"], color="orange", label="Vibration (mm/s)")
plt.xlabel("Time")
plt.ylabel("Vibration Level")
plt.title("Vibration Trend Over Time")
plt.legend()
plt.show()

# ⭐ STEP 4 — Normal vs Failure Condition Comparison

normal = df[df["failure_label"] == 0]
failure = df[df["failure_label"] == 1]

# ✔ Temperature Distribution Comparison
plt.figure(figsize=(10,5))
sns.kdeplot(normal["temperature"], label="Normal", shade=True)
sns.kdeplot(failure["temperature"], label="Failure", shade=True, color="red")
plt.title("Temperature Distribution: Normal vs Failure")
plt.show()

# ✔ Vibration Distribution Comparison
plt.figure(figsize=(10,5))
sns.kdeplot(normal["vibration"], label="Normal", shade=True)
sns.kdeplot(failure["vibration"], label="Failure", shade=True, color="red")
plt.title("Vibration Distribution: Normal vs Failure")
plt.show()

# ⭐ STEP 5 — Correlation Heatmap

# Correlation heatmaps show which features influence failure.

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of CNC Features")
plt.show()

⭐ STEP 6 — Rolling Averages Visualization (Early Failure Patterns)
✔ Rolling Temperature vs Raw Temperature
plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["temperature"], alpha=0.4, label="Raw Temperature")
plt.plot(df["timestamp"], df["temp_roll_mean_30s"], label="30s Rolling Average", linewidth=2)
plt.title("Rolling Temperature Trend")
plt.legend()
plt.show()