import pandas as pd
import numpy as np

RAW_DATA = "data/cnc_data_raw.csv"
OUTPUT_DATA = "data/cnc_features.csv"


def run_etl():

    print("🚀 Starting ETL Pipeline")

    # ---------------------------
    # 1. EXTRACT
    # ---------------------------

    try:
        df = pd.read_csv(RAW_DATA, parse_dates=["timestamp"])
        print(f"Loaded {len(df)} rows")

    except FileNotFoundError:
        print("Raw dataset not found")
        return None

    # ---------------------------
    # 2. CLEAN
    # ---------------------------

    df = df.sort_values("timestamp")

    # Handle missing values
    df = df.ffill().fillna(0)

    # ---------------------------
    # 3. FEATURE ENGINEERING
    # ---------------------------

    print("Creating rolling features")

    df["temp_roll_mean"] = df["temperature"].rolling(30, min_periods=1).mean()
    df["vib_roll_mean"] = df["vibration"].rolling(30, min_periods=1).mean()

    # Rate of change features
    df["temp_change"] = df["temperature"].diff().fillna(0)
    df["vib_change"] = df["vibration"].diff().fillna(0)

    # Machine stress index
    df["machine_stress"] = (
        0.4 * df["temperature"] +
        0.4 * df["vibration"] +
        0.2 * df["speed"]
    )

    # Status encoding
    df["status_encoded"] = df["status"].map({
        "ON": 2,
        "IDLE": 1,
        "OFF": 0
    })

    # ---------------------------
    # 4. FAILURE LABEL (SIMULATED)
    # ---------------------------

    df["failure"] = (
        (df["temperature"] > 70) |
        (df["vibration"] > 3) |
        (df["machine_stress"] > df["machine_stress"].quantile(0.95))).astype(int)

    # ---------------------------
    # 5. LOAD (SAVE DATASET)
    # ---------------------------

    df.to_csv(OUTPUT_DATA, index=False)

    print("ETL Completed")
    print(f"Saved ML dataset → {OUTPUT_DATA}")

    return df


if __name__ == "__main__":

    df = run_etl()

    if df is not None:
        print(df.head())