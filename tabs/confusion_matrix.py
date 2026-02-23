import streamlit as st
import plotly.express as px


def render_confusion_matrix(confusion_df, selected_models):
    selected_cm_model = st.selectbox("Select Model", selected_models)

    cm_df_model = confusion_df[confusion_df["model_name"] == selected_cm_model]

    cm_display = cm_df_model.rename(
        columns={"true_label": "True Label", "predicted_label": "Predicted Label"}
    )

    matrix = cm_display.pivot(
        index="True Label", columns="Predicted Label", values="count"
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"{selected_cm_model} Confusion Matrix",
    )

    fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")

    st.plotly_chart(fig, use_container_width=True)