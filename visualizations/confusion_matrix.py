import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_confusion_matrix(cm_df, model_name):

    df = cm_df[cm_df["model_name"] == model_name]

    matrix = df.pivot(
        index="true_label",
        columns="predicted_label",
        values="count"
    )

    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)