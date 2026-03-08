import streamlit as st


def load_css():

    with open("dashboard/assets/custom.css") as f:
        css = f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )