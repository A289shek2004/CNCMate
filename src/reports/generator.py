# src/reports/generator.py
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import numpy as np

def aggregate_report_data(df, start, end, machine_id="CNC_001"):
    """
    Aggregates data for the given time range and machine.
    """
    # Ensure timestamp is datetime
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    sub = df.loc[mask].copy()
    
    if sub.empty:
        return None

    avg_temp = sub["temperature"].mean()
    max_temp = sub["temperature"].max()
    avg_vib = sub["vibration"].mean()
    max_vib = sub["vibration"].max()
    failures = int(sub["failure_label"].sum())
    
    # Slopes (simple linear regression)
    try:
        tnum = (sub["timestamp"] - sub["timestamp"].min()).dt.total_seconds().values.reshape(-1,1)
        # Check if we have enough points
        if len(tnum) > 1:
            slope_temp = np.polyfit(tnum.flatten(), sub["temperature"].values, 1)[0] * 3600  # deg per hour
            slope_vib = np.polyfit(tnum.flatten(), sub["vibration"].values, 1)[0] * 3600 # mm/s per hour
        else:
            slope_temp = 0.0
            slope_vib = 0.0
    except Exception as e:
        print(f"Error calculating slopes: {e}")
        slope_temp = 0.0
        slope_vib = 0.0

    # Make plots
    plots_dir = "reports/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Unique filename based on range
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_plot = os.path.join(plots_dir, f"temp_{timestamp_str}.png")
    vib_plot = os.path.join(plots_dir, f"vib_{timestamp_str}.png")

    # Temp Plot
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(sub["timestamp"], sub["temperature"], linewidth=0.8, color='tab:red', label='Temp')
    ax.set_title("Temperature Trend")
    ax.legend()
    fig.tight_layout()
    fig.savefig(temp_plot, dpi=100)
    plt.close(fig)

    # Vib Plot
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(sub["timestamp"], sub["vibration"], linewidth=0.8, color='tab:blue', label='Vibration')
    ax.set_title("Vibration Trend")
    ax.legend()
    fig.tight_layout()
    fig.savefig(vib_plot, dpi=100)
    plt.close(fig)

    return {
        "machine_id": machine_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "avg_temp": round(avg_temp, 2),
        "max_temp": round(max_temp, 2),
        "avg_vib": round(avg_vib, 3),
        "max_vib": round(max_vib, 3),
        "failures": failures,
        "slope_temp": round(slope_temp, 4),
        "slope_vib": round(slope_vib, 4),
        "temp_plot": temp_plot,
        "vib_plot": vib_plot
    }

def render_pdf_from_text(data, narrative, outpath="reports/daily_report.pdf"):
    """
    Creates a PDF report from the data and narrative text.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"CNCMate Report - {data['machine_id']}", ln=True, align='C')
    pdf.ln(5)
    
    # Metadata
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, f"Period: {data['start']} to {data['end']}", ln=True, align='C')
    pdf.ln(5)
    
    # Narrative (Executive Summary & content)
    pdf.set_font("Arial", size=11)
    # Multi_cell handles text wrapping
    pdf.multi_cell(0, 6, narrative)
    pdf.ln(5)
    
    # Plots
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Visual Trends", ln=True)
    
    if os.path.exists(data["temp_plot"]):
        pdf.image(data["temp_plot"], w=170)
        pdf.ln(2)
        
    if os.path.exists(data["vib_plot"]):
        pdf.image(data["vib_plot"], w=170)
        pdf.ln(2)
        
    # Output
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    pdf.output(outpath)
    return outpath
