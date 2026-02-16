import streamlit as st
import pandas as pd

def display_table(
    df: pd.DataFrame,
    title: str = None,
    show_index: bool = False,
    container=None
):
    """
    Reusable table display component.

    container:
        - None → main page
        - st.sidebar → sidebar
        - any Streamlit container
    """

    if container is None:
        container = st

    container.markdown("""
    <style>
        table {
            width: 100% !important;
            text-align: center;
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
        }
        th, td {
            text-align: center !important;
            padding: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    if title:
        container.markdown(f"##### {title}")

    container.write(
        df.to_html(escape=False, index=show_index),
        unsafe_allow_html=True
    )
