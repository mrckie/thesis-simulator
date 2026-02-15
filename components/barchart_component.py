import streamlit as st
import pandas as pd
import plotly.express as px

def display_metrics_chart(detailed_df: pd.DataFrame):
    # Convert string values to numeric
    metrics_plot_df = detailed_df.copy()
    metrics_plot_df['Baseline'] = [
        float(x.strip('%')) if '%' in x else float(x.split()[0]) for x in metrics_plot_df['Baseline Model']
    ]
    metrics_plot_df['Modified'] = [
        float(x.strip('%')) if '%' in x else float(x.split()[0]) for x in metrics_plot_df['Modified Model']
    ]

    plot_df = metrics_plot_df.melt(
        id_vars="Metric",
        value_vars=["Baseline", "Modified"],
        var_name="Model",
        value_name="Value"
    )

    fig = px.bar(
        plot_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        text="Value",
        title="Baseline vs Modified Metrics"
    )
    st.plotly_chart(fig, use_container_width=True)
