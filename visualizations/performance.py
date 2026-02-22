import streamlit as st
import plotly.express as px

def plot_performance(df, metric_choice):

    fig = px.bar(
        df,
        x="model_name",
        y=metric_choice,
        color="model_name",
        text=metric_choice,
        title=f"{metric_choice} Comparison",
    )

    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=metric_choice,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)