import streamlit as st
import pandas as pd
from datetime import timedelta

from components.charts import (
    temperature_chart,
    vibration_chart,
    speed_chart,
    energy_chart,
    machine_health_gauge
)

from components.api_client import predict_failure
from components.styles import load_css


st.set_page_config(layout="wide")

load_css()

st.title("🔴 Live Machine Monitoring")

# Load dataset
df = pd.read_csv("data/cnc_features.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")

latest = df.iloc[-1]

# --------------------------------------------------
# Machine Metrics
# --------------------------------------------------

st.subheader("Machine Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Temperature", f"{latest['temperature']:.2f} °C")
col2.metric("Vibration", f"{latest['vibration']:.2f} mm/s")
col3.metric("Speed", f"{int(latest['speed'])} RPM")
col4.metric("Energy", f"{latest['energy']:.2f} W")

# --------------------------------------------------
# AI Prediction
# --------------------------------------------------

st.subheader("AI Failure Prediction")

payload = {
    "temperature": float(latest["temperature"]),
    "vibration": float(latest["vibration"]),
    "speed": float(latest["speed"]),
    "energy": float(latest["energy"]),
    "temp_roll_mean": float(latest["temp_roll_mean"]),
    "vib_roll_mean": float(latest["vib_roll_mean"]),
    "temp_change": float(latest["temp_change"]),
    "vib_change": float(latest["vib_change"]),
    "machine_stress": float(latest["machine_stress"]),
    "status_encoded": int(latest["status_encoded"])
}

prediction = predict_failure(payload)

prob = prediction["failure_probability"]
health = (1 - prob) * 100

machine_health_gauge(health)

st.metric("Failure Probability", f"{prob:.2f}")
st.write("Status:", prediction["status"])
st.write("Recommended Action:", prediction["recommended_action"])

# --------------------------------------------------
# Recent Trends
# --------------------------------------------------

st.subheader("Recent Sensor Trends")

lookback_hours = 6

end_time = df["timestamp"].max()
start_time = end_time - timedelta(hours=lookback_hours)

df_view = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]

temperature_chart(df_view)
vibration_chart(df_view)
speed_chart(df_view)
energy_chart(df_view)