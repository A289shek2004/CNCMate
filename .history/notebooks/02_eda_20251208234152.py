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

# Vibration Trend 
plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["vibration"], color="orange", label="Vibration (mm/s)")
plt.xlabel("Time")
plt.ylabel("Vibration Level")
plt.title("Vibration Trend Over Time")
plt.legend()
plt.show()