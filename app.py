import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import (
    load_summary,
    load_curves,
    load_confusion
)

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Model Compression Evaluation Dashboard",
    layout="wide"
)

st.title("Model Compression Evaluation Dashboard")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (GoEmotions Only)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the GoEmotions dataset to proceed.")
    st.stop()

if "goemotions" not in uploaded_file.name.lower():
    st.error("Invalid dataset. Please upload the GoEmotions dataset used during training.")
    st.stop()

st.sidebar.success("GoEmotions dataset uploaded successfully.")

# -------------------------------------------------
# Load Experiment Data
# -------------------------------------------------
summary_df = load_summary()
curves_df = load_curves()
confusion_df = load_confusion()

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
st.sidebar.header("Model Selection")

selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    options=summary_df["model_name"].unique(),
    default=summary_df["model_name"].unique()[:2]
)

if len(selected_models) < 2:
    st.warning("Select at least two models.")
    st.stop()

metric_choice = st.sidebar.selectbox(
    "Primary Performance Metric",
    ["accuracy", "f1_weighted_avg", "f1_macro_avg"]
)

sort_by = st.sidebar.selectbox(
    "Sort Models By",
    ["reduction_percent", "accuracy", "parameter_count"]
)

# -------------------------------------------------
# Filter Data
# -------------------------------------------------
filtered_summary = summary_df[
    summary_df["model_name"].isin(selected_models)
].sort_values(sort_by)

filtered_curves = curves_df[
    curves_df["model_name"].isin(selected_models)
]

# -------------------------------------------------
# TABS LAYOUT
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Performance",
    "Efficiency",
    "Training Curves",
    "Confusion Matrix"
])

# =================================================
# TAB 1 — OVERVIEW
# =================================================
with tab1:

    st.subheader("Architecture Overview")

    st.dataframe(
        filtered_summary[
            [
                "model_name",
                "reduction_percent",
                "attention_heads",
                "hidden_dim",
                "ffn",
                "parameter_count",
                "trained_epochs"
            ]
        ],
        use_container_width=True
    )

    st.subheader("Performance vs Model Size")

    fig = px.line(
        filtered_summary,
        x="parameter_count",
        y=metric_choice,
        text="model_name",
        markers=True,
        title=f"{metric_choice} vs Parameter Count"
    )

    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# =================================================
# TAB 2 — PERFORMANCE
# =================================================
with tab2:

    st.subheader("Performance Comparison")

    fig = px.bar(
        filtered_summary,
        x="model_name",
        y=metric_choice,
        color="model_name",
        title=f"{metric_choice} Comparison",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Macro vs Weighted F1")

    melted = filtered_summary.melt(
        id_vars="model_name",
        value_vars=["f1_macro_avg", "f1_weighted_avg"],
        var_name="Metric",
        value_name="Score"
    )

    fig2 = px.bar(
        melted,
        x="model_name",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=True
    )

    st.plotly_chart(fig2, use_container_width=True)

# =================================================
# TAB 3 — EFFICIENCY
# =================================================
with tab3:

    st.subheader("Training Time Comparison")

    fig = px.bar(
        filtered_summary,
        x="model_name",
        y="train_time",
        color="model_name",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("GPU Memory Usage")

    fig2 = px.bar(
        filtered_summary,
        x="model_name",
        y="peak_gpu_usage",
        color="model_name",
        text_auto=True
    )

    st.plotly_chart(fig2, use_container_width=True)

# =================================================
# TAB 4 — TRAINING CURVES
# =================================================
with tab4:

    st.subheader("Training & Validation Loss")

    fig = px.line(
        filtered_curves,
        x="epoch",
        y="training_loss",
        color="model_name",
        markers=True,
        title="Training Loss"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        filtered_curves,
        x="epoch",
        y="validation_loss",
        color="model_name",
        markers=True,
        title="Validation Loss"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =================================================
# TAB 5 — CONFUSION MATRIX
# =================================================
with tab5:

    selected_cm_model = st.selectbox(
        "Select Model",
        selected_models
    )

    cm_df_model = confusion_df[
        confusion_df["model_name"] == selected_cm_model
    ]

    matrix = cm_df_model.pivot(
        index="true_label",
        columns="predicted_label",
        values="count"
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"{selected_cm_model} Confusion Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)