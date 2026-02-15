import streamlit as st
import pandas as pd

def display_table(detailed_df: pd.DataFrame):
    # Add CSS to center content and expand table
    st.markdown("""
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
            padding: 16px;
        }
       
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## Detailed Metrics Table")
    st.write(detailed_df.to_html(escape=False, index=False), unsafe_allow_html=True)
