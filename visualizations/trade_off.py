import matplotlib.pyplot as plt
import streamlit as st

def plot_tradeoff(df, metric_choice):

    fig, ax = plt.subplots()

    ax.plot(
        df["parameter_count"],
        df[metric_choice],
        marker="o"
    )

    ax.set_xlabel("Parameter Count")
    ax.set_ylabel(metric_choice)
    ax.set_title(f"{metric_choice} vs Model Size")

    st.pyplot(fig)