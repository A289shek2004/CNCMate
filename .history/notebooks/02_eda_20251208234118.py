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
