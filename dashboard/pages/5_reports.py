import streamlit as st
import pandas as pd
from datetime import datetime
from components.api_client import generate_report
from components.styles import load_css

load_css()

st.title("📄 Machine Reports")

df = pd.read_csv("data/cnc_features.csv", parse_dates=["timestamp"])

# --------------------------------
# Daily Summary
# --------------------------------

st.subheader("Daily Summary")

today = df["timestamp"].max().date()

df_today = df[df["timestamp"].dt.date == today]

summary = {
    "Average Temperature": df_today["temperature"].mean(),
    "Max Temperature": df_today["temperature"].max(),
    "Average Vibration": df_today["vibration"].mean(),
    "Max Vibration": df_today["vibration"].max(),
    "Failures": int(df_today["failure"].sum())
}

st.json(summary)

# --------------------------------
# CSV Export
# --------------------------------

st.subheader("Export Dataset")

st.download_button(
    label="Download CSV",
    data=df_today.to_csv(index=False),
    file_name="cnc_daily_data.csv",
    mime="text/csv"
)

# --------------------------------
# AI PDF Report
# --------------------------------

st.subheader("AI Maintenance Report")

report_date = st.date_input("Select Report Date", datetime.now())

if st.button("Generate AI Report"):

    payload = {
        "machine_id": "CNC_001",
        "start": datetime.combine(report_date, datetime.min.time()).isoformat(),
        "end": datetime.combine(report_date, datetime.max.time()).isoformat()
    }

    with st.spinner("Generating AI report..."):

        pdf = generate_report(payload)

        if pdf:

            st.success("Report generated successfully!")

            st.download_button(
                label="Download PDF",
                data=pdf,
                file_name=f"CNCMate_Report_{report_date}.pdf",
                mime="application/pdf"
            )

        else:
            st.error("Report generation failed.")