import matplotlib.pyplot as plt
import streamlit as st

def plot_efficiency(df):

    metrics = [
        "train_time",
        "peak_gpu_usage",
        "parameter_count"
    ]

    for metric in metrics:
        fig, ax = plt.subplots()
        ax.bar(df["model_name"], df[metric])
        ax.set_title(f"{metric} Comparison")
        ax.set_xticks(range(len(df["model_name"])))
        ax.set_xticklabels(df["model_name"], rotation=45)
        st.pyplot(fig)