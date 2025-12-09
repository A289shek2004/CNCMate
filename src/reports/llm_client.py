# src/reports/llm_client.py

def generate_narrative(data):
    """
    Generates a narrative report based on the aggregated data.
    Uses a rule-based template fallback (deterministic), but structure allows for LLM.
    """
    if not data:
        return "No data available for the selected period."

    parts = []
    
    # Executive Summary structure
    parts.append("EXECUTIVE SUMMARY")
    parts.append(f"Report for {data['machine_id']} covering the period from {data['start']} to {data['end']}.")
    
    summary_line = "Operations were generally stable."
    if data['failures'] > 0:
        summary_line = f"CRITICAL: {data['failures']} failure event(s) occurred during this period."
    elif data['max_temp'] > 80:
        summary_line = "WARNING: High temperature peaks detected."
    elif data['max_vib'] > 5:
        summary_line = "WARNING: High vibration levels observed."
        
    parts.append(summary_line)
    parts.append("") # newline

    # Key Metrics
    parts.append("KEY METRICS")
    parts.append(f"- Avg Temperature: {data['avg_temp']} C (Max: {data['max_temp']} C)")
    parts.append(f"- Avg Vibration: {data['avg_vib']} mm/s (Max: {data['max_vib']} mm/s)")
    parts.append(f"- Detected Failures: {data['failures']}")
    parts.append("")

    # Observations (Rule Based)
    parts.append("OBSERVATIONS & ROOT CAUSE HINTS")
    
    # Temp Rules
    if data['slope_temp'] > 0.5:
        parts.append(f"- Temperature is rising rapidly ({data['slope_temp']} deg/hr). Check cooling system.")
    elif data['max_temp'] > 85:
        parts.append("- Extreme temperature spikes > 85C indicate potential lubrication failure or overload.")
    else:
        parts.append("- Temperature levels appear within normal range.")

    # Vib Rules
    if data['slope_vib'] > 0.1:
        parts.append(f"- Vibration increasing ({data['slope_vib']} mm/s/hr). Possible tool wear progression.")
    elif data['max_vib'] > 5.0:
        parts.append("- Severe vibration peaks > 5.0 mm/s. Immediate spindle inspection recommended.")
    else:
        parts.append("- Vibration signature is stable.")

    parts.append("")

    # Recommended Actions
    parts.append("RECOMMENDED ACTIONS")
    if data['failures'] > 0 or data['max_vib'] > 5:
        parts.append("1. IMMEDIATE: Stop machine and inspect spindle/tooling.")
        parts.append("2. Check for loose components or bearing damage.")
    elif data['max_temp'] > 80:
        parts.append("1. Check coolant levels and flow rate.")
        parts.append("2. Inspect lubrication system for blockages.")
    else:
        parts.append("1. Continue standard monitoring.")
        parts.append("2. Schedule routine maintenance as per plan.")

    # Combine
    return "\n".join(parts)
