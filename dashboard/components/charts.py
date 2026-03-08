import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def temperature_chart(df):
    fig = px.line(
        df,
        x="timestamp",
        y="temperature",
        title="Temperature Trend",
        template="plotly_dark"
    )

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def vibration_chart(df):
    fig = px.line(
        df,
        x="timestamp",
        y="vibration",
        title="Vibration Trend",
        template="plotly_dark",
        color_discrete_sequence=["orange"]
    )

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def speed_chart(df):
    fig = px.line(
        df,
        x="timestamp",
        y="speed",
        title="Spindle Speed",
        template="plotly_dark",
        color_discrete_sequence=["green"]
    )

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def energy_chart(df):
    fig = px.line(
        df,
        x="timestamp",
        y="energy",
        title="Energy Consumption",
        template="plotly_dark",
        color_discrete_sequence=["red"]
    )

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def machine_health_gauge(health_score):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "Machine Health"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"},
            ]
        }
    ))

    fig.update_layout(height=300)

    st.plotly_chart(fig, use_container_width=True)