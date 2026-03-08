# =============================================
# CNCMate Industrial Monitoring Dashboard
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import os

st.set_page_config(layout="wide", page_title="CNCMate Industrial Dashboard")

# st.experimental_rerun()


st_autorefresh(interval=5000)
# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "data/cnc_features.csv"
ALERTS_CSV = "data/alerts_log.csv"

# ==========================================================
# DATA FUNCTIONS
# ==========================================================

@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    return df

def get_latest_row(df):
    return df.iloc[-1]

def call_predict_api(row):

    payload = {
    "temperature": float(row["temperature"]),
    "vibration": float(row["vibration"]),
    "speed": float(row["speed"]),
    "energy": float(row["energy"]),

    "temp_roll_mean": float(row.get("temp_roll_mean", row["temperature"])),
    "vib_roll_mean": float(row.get("vib_roll_mean", row["vibration"])),

    "temp_change": float(row.get("temp_change", 0)),
    "vib_change": float(row.get("vib_change", 0)),

    "machine_stress": float(row.get("machine_stress", 0)),
    "status_encoded": float(row.get("status_encoded", 0))
}

    try:

        r = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=3
        )

        result = r.json()

        # guarantee keys exist
        if "failure_probability" not in result:
            result["failure_probability"] = 0

        if "status" not in result:
            result["status"] = "UNKNOWN"

        return result

    except Exception:

        return {
            "failure_probability": 0,
            "status": "API_OFFLINE",
            "recommended_action": "Check API"
        }

# ==========================================================
# LOAD DATA
# ==========================================================

df = load_data()
latest = get_latest_row(df)

# ==========================================================
# DASHBOARD HEADER
# ==========================================================
pred = call_predict_api(latest)

st.write("API RESPONSE:", pred)
st.title("⚙ CNCMate Industrial Monitoring Dashboard")

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Dashboard Filters")

machine_id = st.sidebar.selectbox(
    "Machine",
    ["CNC_001"]
)

lookback_hours = st.sidebar.slider(
    "Time Window (hours)",
    min_value=1,
    max_value=72,
    value=12
)

# filter data

end_time = df["timestamp"].max()
start_time = end_time - pd.Timedelta(hours=lookback_hours)

df_view = df[
    (df["timestamp"]>=start_time) &
    (df["timestamp"]<=end_time)
]

# ==========================================================
# KPI CALCULATIONS
# ==========================================================

pred = call_predict_api(latest)

failure_rate = df_view["failure"].mean()*100

avg_temp = df_view["temperature"].mean()

avg_vib = df_view["vibration"].mean()

avg_energy = df_view["energy"].mean()

availability = 100 - failure_rate

# ==========================================================
# KPI PANEL
# ==========================================================
status = pred.get("status", "UNKNOWN")
st.write(status)
st.subheader("📊 Operational KPIs")


k1,k2,k3,k4,k5,k6 = st.columns(6)

failure_prob = pred.get("failure_probability", 0)

import plotly.graph_objects as go

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=failure_prob * 100,
    title={'text': "Failure Risk %"},
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 40], 'color': "green"},
            {'range': [40, 70], 'color': "orange"},
            {'range': [70, 100], 'color': "red"}
        ]
    }
))


st.plotly_chart(fig, use_container_width=True)  
health_score = (1 - failure_prob) * 100

st.metric(
    "Machine Health Score",
    f"{health_score:.1f}"
)


k1.metric(
    "Failure Risk",
    f"{failure_prob:.2f}"
)

k2.metric(
    "Failure Rate",
    f"{failure_rate:.2f}%"
)

k3.metric(
    "Availability",
    f"{availability:.2f}%"
)

k4.metric(
    "Avg Temperature",
    f"{avg_temp:.1f} °C"
)

k5.metric(
    "Avg Vibration",
    f"{avg_vib:.2f} mm/s"
)

k6.metric(
    "Avg Energy",
    f"{avg_energy:.1f} W"
)

# ==========================================================
# MACHINE HEALTH SNAPSHOT
# ==========================================================

st.subheader("🩺 Machine Health Snapshot")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Temperature",f"{latest['temperature']:.2f} °C")
c2.metric("Vibration",f"{latest['vibration']:.2f} mm/s")
c3.metric("Speed",f"{latest['speed']:.0f} RPM")
c4.metric("Tool Wear",f"{latest.get('tool_wear_ind',0):.2f}")

# ==========================================================
# SENSOR TRENDS
# ==========================================================

st.subheader("📈 Sensor Trends")

fig = px.line(
    df_view,
    x="timestamp",
    y=["temperature","vibration","energy"],
    template="plotly_dark"
)

st.plotly_chart(fig,use_container_width=True)

# ==========================================================
# ENERGY MONITORING
# ==========================================================

st.subheader("⚡ Energy Consumption")

fig = px.area(
    df_view,
    x="timestamp",
    y="energy",
    template="plotly_dark"
)

st.plotly_chart(fig,use_container_width=True)

# ==========================================================
# FAILURE EVENTS
# ==========================================================

st.subheader("🚨 Failure Events")

failures = df_view[df_view["failure"]==1]

st.write("Total Failures:",len(failures))

if not failures.empty:

    st.dataframe(
        failures[[
            "timestamp",
            "temperature",
            "vibration",
            "energy"
        ]].tail(20)
    )

# ==========================================================
# ALERT HISTORY
# ==========================================================

st.subheader("📋 Alert History")

if os.path.exists(ALERTS_CSV):

    alerts_df = pd.read_csv(ALERTS_CSV)

    alerts_df = alerts_df.sort_values(
        "timestamp",
        ascending=False
    )

    st.dataframe(alerts_df.head(20))

else:

    st.info("No alerts found")

# ==========================================================
# SHIFT ANALYTICS
# ==========================================================

st.subheader("👨‍🏭 Shift Analytics")

df["shift"] = df["timestamp"].dt.hour.apply(
    lambda h:
    "Morning" if 6<=h<14
    else "Evening" if 14<=h<22
    else "Night"
)

shift_stats = df.groupby("shift").agg(
    avg_temp=("temperature","mean"),
    avg_vib=("vibration","mean"),
    failures=("failure","sum")
).reset_index()

st.dataframe(shift_stats)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<center>
CNCMate Predictive Maintenance Platform
</center>
""",
unsafe_allow_html=True
)
