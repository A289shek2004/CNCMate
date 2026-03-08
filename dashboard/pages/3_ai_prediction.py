import streamlit as st

from components.api_client import predict_failure
from components.styles import load_css

load_css()

st.title("🤖 AI Failure Prediction")

st.write("Enter machine sensor values to predict failure risk.")

# -----------------------------------
# Input Form
# -----------------------------------

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input("Temperature (°C)", 0.0, 200.0, 50.0)
        vibration = st.number_input("Vibration (mm/s)", 0.0, 20.0, 1.0)
        speed = st.number_input("Speed (RPM)", 0.0, 10000.0, 2000.0)
        energy = st.number_input("Energy (W)", 0.0, 500.0, 50.0)

    with col2:

        temp_roll_mean = st.number_input("Temperature Rolling Mean", 0.0, 200.0, 50.0)
        vib_roll_mean = st.number_input("Vibration Rolling Mean", 0.0, 20.0, 1.0)
        temp_change = st.number_input("Temperature Change", -50.0, 50.0, 0.0)
        vib_change = st.number_input("Vibration Change", -10.0, 10.0, 0.0)

    machine_stress = st.number_input("Machine Stress", 0.0, 1000.0, 500.0)
    status_encoded = st.selectbox("Machine Status", [0, 1, 2])

    submit = st.form_submit_button("Predict Failure")

# -----------------------------------
# Prediction
# -----------------------------------

if submit:

    payload = {
        "temperature": temperature,
        "vibration": vibration,
        "speed": speed,
        "energy": energy,
        "temp_roll_mean": temp_roll_mean,
        "vib_roll_mean": vib_roll_mean,
        "temp_change": temp_change,
        "vib_change": vib_change,
        "machine_stress": machine_stress,
        "status_encoded": status_encoded
    }

    result = predict_failure(payload)

    prob = result["failure_probability"]

    st.subheader("Prediction Result")

    st.metric("Failure Probability", f"{prob:.2f}")

    if prob > 0.7:
        st.error("⚠ HIGH FAILURE RISK")

    elif prob > 0.4:
        st.warning("⚠ MACHINE AT RISK")

    else:
        st.success("Machine Operating Normally")

    st.write("Recommended Action:")
    st.info(result["recommended_action"])