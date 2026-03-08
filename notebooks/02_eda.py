# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -----------------------------
# 2. LOAD DATASET
# -----------------------------

df = pd.read_csv("data/cnc_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

print("Dataset Shape:", df.shape)
print(df.head())

failure=df["failure"].value_counts()
print("\nFailure Distribution:", failure.to_dict())
# -----------------------------
# 3. DESCRIPTIVE STATISTICS
# -----------------------------

print("\nBasic Statistics:")
print(df.describe().T)

# -----------------------------
# 4. TEMPERATURE TREND
# -----------------------------

plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["temperature"], label="Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Trend Over Time")
plt.legend()
plt.show()

# -----------------------------
# 5. VIBRATION TREND
# -----------------------------

plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["vibration"], color="orange", label="Vibration")
plt.xlabel("Time")
plt.ylabel("Vibration")
plt.title("Vibration Trend Over Time")
plt.legend()
plt.show()

# -----------------------------
# 6. NORMAL VS FAILURE
# -----------------------------

normal = df[df["failure"] == 0]
failure = df[df["failure"] == 1]

plt.figure(figsize=(10,5))
sns.kdeplot(normal["temperature"], label="Normal", fill=True)
sns.kdeplot(failure["temperature"], label="Failure", fill=True, color="red")
plt.title("Temperature Distribution: Normal vs Failure")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
sns.kdeplot(normal["vibration"], label="Normal", fill=True)
sns.kdeplot(failure["vibration"], label="Failure", fill=True, color="red")
plt.title("Vibration Distribution: Normal vs Failure")
plt.legend()
plt.show()

# -----------------------------
# 7. CORRELATION HEATMAP
# -----------------------------

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------
# 8. ROLLING FEATURE ANALYSIS
# -----------------------------

plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["temperature"], alpha=0.4)
plt.plot(df["timestamp"], df["temp_roll_mean"], linewidth=2)
plt.title("Temperature vs Rolling Mean")
plt.show()

plt.figure(figsize=(14,5))
plt.plot(df["timestamp"], df["vibration"], alpha=0.4)
plt.plot(df["timestamp"], df["vib_roll_mean"], linewidth=2, color="red")
plt.title("Vibration vs Rolling Mean")
plt.show()

# -----------------------------
# 9. TOOL USAGE TREND
# -----------------------------

# plt.figure(figsize=(14,5))
# plt.plot(df["timestamp"], df["tool_usage"])
# plt.title("Tool Usage Over Time")
# plt.ylabel("Tool Usage")
# plt.show()

# -----------------------------
# 10. FAILURE BOXPLOTS
# -----------------------------

plt.figure(figsize=(7,5))
sns.boxplot(x=df["failure"], y=df["temperature"])
plt.title("Temperature vs Failure")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x=df["failure"], y=df["vibration"])
plt.title("Vibration vs Failure")
plt.show()