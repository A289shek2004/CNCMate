import streamlit as st
import pandas as pd
import os

st.title("🚨 Alert Center")

ALERTS_CSV = "data/alerts_log.csv"

# --------------------------------
# Load Alerts
# --------------------------------

if os.path.exists(ALERTS_CSV):

    alerts = pd.read_csv(ALERTS_CSV)

    # Fix column naming
    if "type" in alerts.columns:
        alerts = alerts.rename(columns={"type": "alert_type"})

    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"])

    # --------------------------------
    # Sidebar Filters
    # --------------------------------

    st.sidebar.header("Alert Filters")

    alert_type = st.sidebar.selectbox(
        "Alert Type",
        ["All"] + list(alerts["alert_type"].unique())
    )

    date_filter = st.sidebar.date_input(
        "Filter by Date",
        value=None
    )

    # --------------------------------
    # Apply Filters
    # --------------------------------

    df_view = alerts.copy()

    if alert_type != "All":
        df_view = df_view[df_view["alert_type"] == alert_type]

    if date_filter:
        df_view = df_view[
            df_view["timestamp"].dt.date == date_filter
        ]

    # --------------------------------
    # Recent Alerts
    # --------------------------------

    st.subheader("Recent Alerts")

    for _, row in df_view.tail(10).iterrows():

        if "FAILURE" in str(row["status"]):
            st.error(f"{row['timestamp']} | {row['status']} → {row['recommended_action']}")

        elif "Vibration" in str(row["status"]):
            st.warning(f"{row['timestamp']} | {row['status']} → {row['recommended_action']}")

        else:
            st.info(f"{row['timestamp']} | {row['status']} → {row['recommended_action']}")

    # --------------------------------
    # Alert Table
    # --------------------------------

    st.markdown("---")

    st.subheader("Alert History")

    st.dataframe(
        df_view.sort_values("timestamp", ascending=False)
    )

else:

    st.info("No alert log file found.")