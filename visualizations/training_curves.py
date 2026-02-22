import matplotlib.pyplot as plt
import streamlit as st

def plot_training_curves(curve_df, selected_models):

    for model in selected_models:
        model_df = curve_df[curve_df["model_name"] == model]

        fig, ax = plt.subplots()
        ax.plot(model_df["epoch"], model_df["training_loss"], label="Training Loss")
        ax.plot(model_df["epoch"], model_df["validation_loss"], label="Validation Loss")

        ax.set_title(f"{model} Training Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        st.pyplot(fig)