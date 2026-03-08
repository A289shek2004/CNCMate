import streamlit as st
import pandas as pd
import plotly.express as px

from components.styles import load_css

load_css()

st.title("📊 Machine Analytics")

# Load dataset
df = pd.read_csv("data/cnc_features.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")

# ----------------------------------------
# Filters
# ----------------------------------------

st.sidebar.header("Filters")

machine = st.sidebar.selectbox(
    "Machine ID",
    df["machine_id"].unique() if "machine_id" in df.columns else ["CNC_001"]
)

time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"]
)

if time_range == "Last 1 hour":
    df_view = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(hours=1)]

elif time_range == "Last 6 hours":
    df_view = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(hours=6)]

elif time_range == "Last 24 hours":
    df_view = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(hours=24)]

else:
    df_view = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=7)]


# ----------------------------------------
# Temperature Analysis
# ----------------------------------------

st.subheader("Temperature Analysis")

fig = px.line(
    df_view,
    x="timestamp",
    y="temperature",
    title="Temperature Trend",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# Vibration Analysis
# ----------------------------------------

st.subheader("Vibration Analysis")

fig = px.line(
    df_view,
    x="timestamp",
    y="vibration",
    title="Vibration Trend",
    template="plotly_dark",
    color_discrete_sequence=["orange"]
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# Machine Stress
# ----------------------------------------

st.subheader("Machine Stress")

fig = px.line(
    df_view,
    x="timestamp",
    y="machine_stress",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# Failure Events
# ----------------------------------------

st.subheader("Failure Events")

failures = df_view[df_view["failure"] == 1]

st.write("Total Failures:", len(failures))

if not failures.empty:

    st.dataframe(
        failures[[
            "timestamp",
            "temperature",
            "vibration",
            "machine_stress"
        ]]
    )