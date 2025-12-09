# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import os
import csv
import time
import json
from streamlit.components.v1 import html

st.set_page_config(layout="wide", page_title="CNCMate Dashboard")

# ---------------------------
# Data & Config
# ---------------------------
ALERTS_CSV = "data/alerts_log.csv"
os.makedirs("data", exist_ok=True)
# Init alerts log if needed
if not os.path.exists(ALERTS_CSV):
    with open(ALERTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","type","value","status","recommended_action","note"])

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data(ttl=300)
def load_data(path="data/cnc_features.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    return df

def get_latest_row(df):
    return df.iloc[-1]

def call_predict_api(row, api_url="http://127.0.0.1:8000/predict", timeout=4):
    """
    Call FastAPI /predict endpoint.
    Falls back to simple rule if API fails.
    """
    payload = {
        "temperature": float(row["temperature"]),
        "vibration": float(row["vibration"]),
        "speed": float(row["speed"]),
        "energy": float(row["energy"]),
        "temp_roll_mean_30s": float(row.get("temp_roll_mean_30s", row["temperature"])),
        "vib_roll_mean_30s": float(row.get("vib_roll_mean_30s", row["vibration"])),
        "temp_diff": float(row.get("temp_diff", 0.0)),
        "speed_pct_change": float(row.get("speed_pct_change", 0.0)),
        "tool_wear_ind": float(row.get("tool_wear_ind", 0.0))
    }
    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # fallback simple rule if API not reachable
        prob = 1.0 if (row["temperature"] > 80 or row["vibration"] > 4) else 0.0
        status = "FAILURE_SOON" if prob == 1.0 else "NORMAL"
        return {"failure_probability": prob, "status": status, "recommended_action": "API unreachable; use local rule."}

def log_alert(alert_type, value, status, recommended_action, note=""):
    row = [datetime.now().isoformat(), alert_type, value, status, recommended_action, note]
    with open(ALERTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def browser_notify(title, message):
    # small JS to request permission and send a notification
    js = f"""
    <script>
    (function() {{
      function notifyMe() {{
        if (!("Notification" in window)) {{
          return;
        }}
        if (Notification.permission === "granted") {{
          new Notification("{title}", {{ body: "{message}" }});
        }} else if (Notification.permission !== "denied") {{
          Notification.requestPermission().then(function (permission) {{
            if (permission === "granted") {{
              new Notification("{title}", {{ body: "{message}" }});
            }}
          }});
        }}
      }}
      notifyMe();
    }})();
    </script>
    """
    # embed small invisible html to run the script
    html(js)

def send_telegram(message, bot_token, chat_id):
    if not bot_token or not chat_id:
        return False, "Missing token/chat"
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        r = requests.post(url, data=data, timeout=5)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

def plot_time_series(df, col, title=None, rolling=None, height=250):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(df["timestamp"], df[col], linewidth=0.6)
    if rolling:
        ax.plot(df["timestamp"], df[col].rolling(rolling).mean(), linewidth=1.2)
    ax.set_title(title or col)
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

def daily_summary(df):
    last_day = df["timestamp"].max().normalize()
    mask = (df["timestamp"] >= last_day) & (df["timestamp"] < last_day + pd.Timedelta(days=1))
    day_df = df.loc[mask]
    if day_df.empty:
        return {}
    summary = {
        "avg_temp": day_df["temperature"].mean(),
        "max_temp": day_df["temperature"].max(),
        "avg_vib": day_df["vibration"].mean(),
        "max_vib": day_df["vibration"].max(),
        "failures": int(day_df["failure_label"].sum())
    }
    return summary

def export_pdf_summary(summary: dict, short_text="CNCMate Daily Summary"):
    """
    Create a one-page PDF using FPDF. Returns bytes.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, short_text, ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    for k, v in summary.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    # return binary stream
    return pdf.output(dest="S").encode("latin-1")

# ---------------------------
# Main UI
# ---------------------------
st.title("CNCMate — Machine Analytics Dashboard")

# Load data
df = load_data()
latest = get_latest_row(df)

# Sidebar controls
st.sidebar.header("Controls")
lookback_hours = st.sidebar.slider("Lookback (hours) for trends", min_value=1, max_value=72, value=12)
update_button = st.sidebar.button("Refresh Data")
api_url = st.sidebar.text_input("Prediction API URL", value="http://127.0.0.1:8000/predict")

st.sidebar.markdown("---")
st.sidebar.header("Alert Settings")
vibration_threshold = st.sidebar.number_input("Vibration threshold (mm/s)", min_value=0.1, max_value=20.0, value=4.0, step=0.1)
prob_threshold = st.sidebar.slider("Failure probability threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
enable_browser_notifs = st.sidebar.checkbox("Enable browser pop-up notifications", value=False)
enable_telegram = st.sidebar.checkbox("Enable Telegram alerts", value=False)
if enable_telegram:
    telegram_bot_token = st.sidebar.text_input("Telegram bot token", "")
    telegram_chat_id = st.sidebar.text_input("Telegram chat id", "")
else:
    telegram_bot_token = ""
    telegram_chat_id = ""

if update_button:
    load_data.clear()

# Filter data for lookback
end_time = df["timestamp"].max()
start_time = end_time - pd.Timedelta(hours=lookback_hours)
df_view = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)].copy()

# ---------------------------
# Alerts Logic
# ---------------------------
# Call prediction API first so we can check alerts
pred = call_predict_api(latest, api_url=api_url)

alerts = []
# 1) Vibration threshold alert
if float(latest["vibration"]) >= vibration_threshold:
    msg = f"High Vibration: {latest['vibration']:.2f} mm/s (threshold {vibration_threshold})"
    alerts.append(("vibration", latest["vibration"], "High Vibration", msg))

# 2) Failure probability alert
if float(pred.get("failure_probability", 0)) >= prob_threshold:
    msg = f"High failure probability: {pred['failure_probability']:.2f}"
    alerts.append(("failure_probability", pred["failure_probability"], pred["status"], pred["recommended_action"]))

# Display alerts on UI
if alerts:
    st.subheader("⚠️ Active Alerts")
    for a_type, value, status, rec in alerts:
        # show prominent banner for critical (prob >= prob_threshold)
        if a_type == "failure_probability":
            st.error(f"🔴 {status} — {rec} (prob={value:.2f})")
        else:
            st.warning(f"🟠 {status}: {rec}")

        # log alert
        log_alert(a_type, value, status, rec, note="generated in dashboard")

        # browser notification
        if enable_browser_notifs:
            browser_notify(f"CNCMate Alert — {status}", rec)

        # telegram
        if enable_telegram and telegram_bot_token and telegram_chat_id:
            ok, res = send_telegram(f"CNCMate Alert — {status}: {rec}", telegram_bot_token, telegram_chat_id)
            if not ok:
                st.info(f"Telegram failed: {res}")


# ---------------------------
# Page Layout
# ---------------------------

# Top row: Machine Overview
st.header("Machine Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Temperature (°C)", f"{latest['temperature']:.2f}")
    st.metric("Temp Rolling (30s)", f"{latest.get('temp_roll_mean_30s',0):.2f}")

with col2:
    st.metric("Vibration (mm/s)", f"{latest['vibration']:.2f}")
    st.metric("Vib Rolling (30s)", f"{latest.get('vib_roll_mean_30s',0):.2f}")

with col3:
    st.metric("Speed (RPM)", f"{int(latest['speed'])}")
    st.metric("Energy (W)", f"{latest['energy']:.1f}")

with col4:
    st.metric("Tool Wear", f"{latest.get('tool_wear_ind',0):.2f}")
    st.metric("Failure Prob", f"{pred['failure_probability']:.2f}")
    st.write(pred["status"])

st.markdown("---")

# Page 2: Trends & Analytics
st.header("Trends & Analytics")
st.subheader("Temperature")
plot_time_series(df_view, "temperature", title="Temperature Trend", rolling=3)
st.subheader("Vibration")
plot_time_series(df_view, "vibration", title="Vibration Trend", rolling=3)

st.subheader("Tool Wear (Normalized)")
fig, ax = plt.subplots(figsize=(10,2.8))
ax.plot(df_view["timestamp"], df_view["tool_wear_ind"], linewidth=0.8)
ax.set_title("Tool Wear Indicator")
ax.set_xlabel("")
st.pyplot(fig)

st.subheader("Failure Events Over Time")
failures = df_view[df_view["failure_label"]==1]
st.write(f"Failures in view: {len(failures)}")
if not failures.empty:
    st.table(failures[["timestamp","temperature","vibration","tool_wear_ind"]].tail(10))

st.markdown("---")

# Page 3: Alert History (NEW)
st.header("Recent Alerts Log")
if os.path.exists(ALERTS_CSV):
    alerts_df = pd.read_csv(ALERTS_CSV)
    if not alerts_df.empty:
        # sort by timestamp desc
        alerts_df = alerts_df.sort_values("timestamp", ascending=False).head(10)
        st.dataframe(alerts_df)
    else:
        st.info("No alerts logged yet.")
else:
    st.info("Alert log file not found.")

st.markdown("---")

# Page 4: Operator / Shift Analytics
st.header("Operator / Shift Analytics (Optional)")
df["shift"] = df["timestamp"].dt.hour.apply(lambda h: "Morning" if 6<=h<14 else ("Evening" if 14<=h<22 else "Night"))
shift_summary = df.groupby("shift").agg(
    avg_temp=("temperature","mean"),
    avg_vib=("vibration","mean"),
    failures=("failure_label","sum")
).reset_index()
st.dataframe(shift_summary)

st.markdown("---")

# Page 5: Reports
st.header("Reports")
summary = daily_summary(df)

col1_rep, col2_rep = st.columns([1, 1])

with col1_rep:
    st.subheader("Daily Summary JSON")
    st.json(summary)
    st.download_button("Export CSV (last 24h)", data=df_view.to_csv(index=False).encode('utf-8'),
                       file_name="cnc_last24h.csv", mime="text/csv")

with col2_rep:
    st.subheader("AI PDF Report Generator")
    report_date = st.date_input("Select Report Date", datetime.now())
    
    if st.button("Generate AI Report"):
        with st.spinner("Analyzing data and generating narrative..."):
            # Define range for that day
            start_rpt = datetime.combine(report_date, datetime.min.time())
            end_rpt = datetime.combine(report_date, datetime.max.time())
            
            payload = {
                "machine_id": "CNC_001",
                "start": start_rpt.isoformat(),
                "end": end_rpt.isoformat()
            }
            
            try:
                # Need absolute full URL if running in docker or different ports
                # For local defaults:
                api_gen_url = api_url.replace("/predict", "/generate_report")
                
                resp = requests.post(api_gen_url, json=payload, timeout=20)
                if resp.status_code == 200:
                    st.success("Report Generated Successfully!")
                    st.download_button(
                        label="Download PDF Report",
                        data=resp.content,
                        file_name=f"CNCMate_Report_{report_date}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error(f"Failed: {resp.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

st.markdown("### Tips for Blackbook Screenshots")
st.write("- Use full-screen browser and take screenshots of each section (Overview, Trends, Failures, Reports).")
st.write("- Use `streamlit run dashboard/app.py` then press F11 (Windows) for full-screen.")
