import streamlit as st
import plotly.express as px


def render_training_curves(filtered_summary, filtered_curves):
    st.subheader("Epochs Configuration")

    epochs_info = (
        filtered_summary[["model_name", "epochs", "trained_epochs"]]
        .rename(
            columns={
                "model_name": "Model Name",
                "epochs": "Hyperparameter Epochs",
                "trained_epochs": "Actual Epochs (with Early Stopping)",
            }
        )
        .drop_duplicates()
    )

    col1, col2, col3 = st.columns(3)
    for idx, row in epochs_info.iterrows():
        delta_val = int(row["Actual Epochs (with Early Stopping)"]) - int(
            row["Hyperparameter Epochs"]
        )
        metric_str = f"{int(row['Actual Epochs (with Early Stopping)'])}/{int(row['Hyperparameter Epochs'])}"

        if idx % 3 == 0:
            with col1:
                st.metric(row["Model Name"], metric_str, delta=f"{delta_val} epochs")
        elif idx % 3 == 1:
            with col2:
                st.metric(row["Model Name"], metric_str, delta=f"{delta_val} epochs")
        else:
            with col3:
                st.metric(row["Model Name"], metric_str, delta=f"{delta_val} epochs")

    st.info("ⓘ Early stopping is configured with patience of 2 epochs.")

    curves_df_display = filtered_curves.rename(
        columns={
            "model_name": "Model",
            "epoch": "Epoch",
            "training_loss": "Training Loss",
            "validation_loss": "Validation Loss",
        }
    )

    st.subheader("Training Loss")
    fig = px.line(
        curves_df_display, x="Epoch", y="Training Loss", color="Model", markers=True
    )
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Training Loss")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Validation Loss")
    fig2 = px.line(
        curves_df_display, x="Epoch", y="Validation Loss", color="Model", markers=True
    )
    fig2.update_layout(xaxis_title="Epoch", yaxis_title="Validation Loss")
    st.plotly_chart(fig2, use_container_width=True)
