# src/etl_pipeline.py
import pandas as pd
import numpy as np

def run_etl(input_path="data/cnc_features.csv"):
    """
    Simulated ETL pipeline:
    1. Loads data
    2. Cleans columns / NaNs
    3. Merges tool_usage if present (conceptually)
    4. Computes derived features (rolling means)
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path, parse_dates=["timestamp"])
    except FileNotFoundError:
        print("Data file not found.")
        return None

    # Sort
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Simulated NaN handling
    if df.isnull().sum().sum() > 0:
        print("Imputing NaNs...")
        df = df.fillna(method="ffill").fillna(0)

    # Feature Engineering (if not already present)
    # This matches logic from 03_ML_Models.ipynb
    if "temp_roll_mean_30s" not in df.columns:
        print("Computing rolling features...")
        df["temp_roll_mean_30s"] = df["temperature"].rolling(window=30, min_periods=1).mean()
        df["vib_roll_mean_30s"] = df["vibration"].rolling(window=30, min_periods=1).mean()

    print("ETL pipeline completed successfully.")
    return df

if __name__ == "__main__":
    df = run_etl()
    if df is not None:
        print(f"Processed {len(df)} rows.")
