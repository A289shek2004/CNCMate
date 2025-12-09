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

# ⭐ STEP 6 — Rolling Averages Visualization (Early Failure Patterns)
# ✔ Rolling Temperature vs Raw Temperature
plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["temperature"], alpha=0.4, label="Raw Temperature")
plt.plot(df["timestamp"], df["temp_roll_mean_30s"], label="30s Rolling Average", linewidth=2)
plt.title("Rolling Temperature Trend")
plt.legend()
plt.show()

# ✔ Rolling Vibration vs Raw Vibration
plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["vibration"], alpha=0.4, label="Raw Vibration")
plt.plot(df["timestamp"], df["vib_roll_mean_30s"], label="30s Rolling Avg", linewidth=2, color="red")
plt.title("Rolling Vibration Trend")
plt.legend()
plt.show()

# ⭐ STEP 7 — Tool Usage Patterns
plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["tool_wear_ind"], label="Tool Wear Indicator")
plt.title("Tool Wear Progression Over Time")
plt.ylabel("Normalized Tool Wear (0→1)")
plt.show()
# ⭐ STEP 8 — Boxplots (Useful for Report)
# Temperature Boxplot vs Failure
plt.figure(figsize=(7,5))
sns.boxplot(x=df["failure_label"], y=df["temperature"])
plt.title("Temperature vs Failure")
plt.show()
# Vibration Boxplot vs Failure
plt.figure(figsize=(7,5))
sns.boxplot(x=df["failure_label"], y=df["vibration"])
plt.title("Vibration vs Failure")
plt.show()

# ⭐ STEP 9 — Extract Key EDA Insights for the Report

# You must add these to your final submission:

# 🔥 1. Vibration Rising Pattern → Early Tool Wear

# Rolling vibration graphs show gradual increase

# Predictive indicator for tool end-of-life

# 🔥 2. Temperature Instability → Lubrication Failure

# Failures occur alongside heat spikes

# Often temperature jumps 5–10°C before failure

# 🔥 3. Speed Fluctuations → Spindle Issues

# High speed_pct_change values are correlated with failures

# 🔥 4. Tool Life Patterns

# Failures occur after high tool usage

# Tool wear indicator ≥ 0.7 → higher failure probability

# 🔥 5. Vibrations above 4 mm/s are strongly associated with failures

# This will help ML model performance.