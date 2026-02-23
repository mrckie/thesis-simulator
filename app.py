import streamlit as st
import pandas as pd
import plotly.express as px
import time

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
# SIDEBAR — DATASET UPLOAD
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
else:
    if "uploaded" not in st.session_state:
        success_placeholder = st.sidebar.empty()
        success_placeholder.success("GoEmotions dataset uploaded successfully.")
        time.sleep(1)
        success_placeholder.empty()
        st.session_state.uploaded = True

# -------------------------------------------------
# Load Experiment Data
# -------------------------------------------------
summary_df = load_summary()
curves_df = load_curves()
confusion_df = load_confusion()

# -------------------------------------------------
# Model Selection (global)
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

# Filtered Data
filtered_summary = summary_df[
    summary_df["model_name"].isin(selected_models)
]

filtered_curves = curves_df[
    curves_df["model_name"].isin(selected_models)
]

# -------------------------------------------------
# TABS
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

    col1, col2 = st.columns(2)

    with col1:
        metric_choice = st.selectbox(
            "Select Primary Performance Metric",
            ["Accuracy", "F1 Score (Weighted Average)", "F1 Score (Macro Average)"]
        )

    with col2:
        sort_by = st.selectbox(
            "Sort Models By",
            ["Reduction (%)", "Accuracy", "Parameter Count"]
        )

    metric_mapping = {
        "Accuracy": "accuracy",
        "F1 Score (Weighted Average)": "f1_weighted_avg",
        "F1 Score (Macro Average)": "f1_macro_avg"
    }

    sort_mapping = {
        "Reduction (%)": "reduction_percent",
        "Accuracy": "accuracy",
        "Parameter Count": "parameter_count"
    }

    sorted_df = filtered_summary.sort_values(sort_mapping[sort_by])

    st.subheader("Architecture Configuration")

    architecture_table = sorted_df[
        [
            "model_name",
            "reduction_percent",
            "attention_heads",
            "hidden_dim",
            "ffn",
            "parameter_count",
            "trained_epochs"
        ]
    ].rename(columns={
        "model_name": "Model Name",
        "reduction_percent": "Reduction (%)",
        "attention_heads": "Attention Heads",
        "hidden_dim": "Hidden Dimension",
        "ffn": "FFN Size",
        "parameter_count": "Parameter Count",
        "trained_epochs": "Trained Epochs"
    })

    st.dataframe(architecture_table, use_container_width=True)

    st.subheader("Performance vs Model Size")

    overview_df = sorted_df.rename(columns={
        "accuracy": "Accuracy",
        "parameter_count": "Parameter Count",
        "model_name": "Model"
    })

    fig = px.line(
        overview_df,
        x="Parameter Count",
        y="Accuracy",
        text="Model",
        markers=True,
        title=f"{metric_choice} vs Parameter Count"
    )

    fig.update_traces(
        textposition="top center",
        texttemplate='%{y:.2%}'  # Display as percentage
    )
    fig.update_layout(
        xaxis_title="Parameter Count",
        yaxis_title=metric_choice,
        yaxis_tickformat=".0%"  # Percentage format
    )

    st.plotly_chart(fig, use_container_width=True)

# =================================================
# TAB 2 — PERFORMANCE
# =================================================
with tab2:

    metric_options = {
        "Accuracy": "accuracy",
        "Precision (Weighted Average)": "precision_weighted_avg",
        "Precision (Macro Average)": "precision_macro_avg",
        "Recall (Weighted Average)": "recall_weighted_avg",
        "Recall (Macro Average)": "recall_macro_avg",
        "F1 Score (Weighted Average)": "f1_weighted_avg",
        "F1 Score (Macro Average)": "f1_macro_avg"
    }

    selected_metric_label = st.selectbox(
        "Select Performance Metric",
        list(metric_options.keys())
    )

    metric_choice = metric_options[selected_metric_label]

    st.subheader(f"{selected_metric_label} Comparison Across Models")

    perf_df = filtered_summary.rename(columns={"model_name": "Model"})

    fig = px.bar(
        perf_df,
        x="Model",
        y=metric_choice,
        color="Model",
        text_auto=True
    )

    fig.update_traces(texttemplate='%{y:.2%}')
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=selected_metric_label,
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Classification Metrics")

    classification_table = filtered_summary[
        [
            "model_name",
            "accuracy",
            "precision_weighted_avg",
            "precision_macro_avg",
            "recall_weighted_avg",
            "recall_macro_avg",
            "f1_weighted_avg",
            "f1_macro_avg"
        ]
    ].rename(columns={
        "model_name": "Model Name",
        "accuracy": "Accuracy",
        "precision_weighted_avg": "Precision (Weighted)",
        "precision_macro_avg": "Precision (Macro)",
        "recall_weighted_avg": "Recall (Weighted)",
        "recall_macro_avg": "Recall (Macro)",
        "f1_weighted_avg": "F1 Score (Weighted)",
        "f1_macro_avg": "F1 Score (Macro)"
    })

    # Convert decimal to percentage for display
    classification_table_pct = classification_table.copy()
    for col in classification_table.columns[1:]:
        classification_table_pct[col] = (classification_table_pct[col] * 100).round(2).astype(str) + "%"

    st.dataframe(classification_table_pct, use_container_width=True)

# =================================================
# TAB 3 — EFFICIENCY
# =================================================
with tab3:

    col1, col2 = st.columns(2)

    with col1:
        time_unit = st.selectbox(
            "Training Time Unit",
            ["Seconds", "Minutes"]
        )

    with col2:
        memory_unit = st.selectbox(
            "Memory Unit",
            ["MB", "GB"]
        )

    # Prepare data for training time
    efficiency_df = filtered_summary.copy()
    
    if time_unit == "Minutes":
        efficiency_df["train_time_display"] = efficiency_df["train_time"] / 60
        time_label = "Training Time (Minutes)"
    else:
        efficiency_df["train_time_display"] = efficiency_df["train_time"]
        time_label = "Training Time (Seconds)"

    if memory_unit == "MB":
        efficiency_df["gpu_memory_display"] = efficiency_df["peak_gpu_usage"] * 1000
        efficiency_df["cpu_memory_display"] = efficiency_df["peak_cpu_usage"] * 1000
        gpu_memory_label = "Peak GPU Memory Usage (MB)"
        cpu_memory_label = "Peak CPU Memory Usage (MB)"
    else:
        efficiency_df["gpu_memory_display"] = efficiency_df["peak_gpu_usage"]
        efficiency_df["cpu_memory_display"] = efficiency_df["peak_cpu_usage"]
        gpu_memory_label = "Peak GPU Memory Usage (GB)"
        cpu_memory_label = "Peak CPU Memory Usage (GB)"

    # Rename columns for display
    efficiency_display = efficiency_df.rename(columns={
        "model_name": "Model",
        "train_time_display": "Training Time",
        "gpu_memory_display": gpu_memory_label,
        "cpu_memory_display": cpu_memory_label
    })

    st.subheader("Training Time Comparison")

    fig = px.bar(
        efficiency_display,
        x="Model",
        y="Training Time",
        color="Model",
        text_auto=True,
        title=time_label
    )

    fig.update_traces(texttemplate='%{y:.2f}')
    fig.update_layout(xaxis_title="Model", yaxis_title="Training Time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Peak GPU Memory Usage")

    fig2 = px.bar(
        efficiency_display,
        x="Model",
        y=gpu_memory_label,
        color="Model",
        text_auto=True,
        title=gpu_memory_label
    )

    fig2.update_traces(texttemplate='%{y:.2f}')
    fig2.update_layout(xaxis_title="Model", yaxis_title=gpu_memory_label)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Peak CPU Memory Usage")

    fig3 = px.bar(
        efficiency_display,
        x="Model",
        y=cpu_memory_label,
        color="Model",
        text_auto=True,
        title=cpu_memory_label
    )

    fig3.update_traces(texttemplate='%{y:.2f}')
    fig3.update_layout(xaxis_title="Model", yaxis_title=cpu_memory_label)
    st.plotly_chart(fig3, use_container_width=True)

# =================================================
# TAB 4 — TRAINING CURVES
# =================================================
with tab4:

    st.subheader("Epochs Configuration")
    
    # Create a table showing hyperparameter epochs vs actual epochs with early stopping
    epochs_info = filtered_summary[
        ["model_name", "epochs", "trained_epochs"]
    ].rename(columns={
        "model_name": "Model Name",
        "epochs": "Hyperparameter Epochs",
        "trained_epochs": "Actual Epochs (with Early Stopping)"
    }).drop_duplicates()
    
    col1, col2, col3 = st.columns(3)
    for idx, row in epochs_info.iterrows():
        if idx % 3 == 0:
            with col1:
                st.metric(row["Model Name"], f"{int(row['Actual Epochs (with Early Stopping)'])}/{int(row['Hyperparameter Epochs'])}", 
                         delta=f"{int(row['Actual Epochs (with Early Stopping)']) - int(row['Hyperparameter Epochs'])} epochs")
        elif idx % 3 == 1:
            with col2:
                st.metric(row["Model Name"], f"{int(row['Actual Epochs (with Early Stopping)'])}/{int(row['Hyperparameter Epochs'])}", 
                         delta=f"{int(row['Actual Epochs (with Early Stopping)']) - int(row['Hyperparameter Epochs'])} epochs")
        else:
            with col3:
                st.metric(row["Model Name"], f"{int(row['Actual Epochs (with Early Stopping)'])}/{int(row['Hyperparameter Epochs'])}", 
                         delta=f"{int(row['Actual Epochs (with Early Stopping)']) - int(row['Hyperparameter Epochs'])} epochs")
    
    st.info("ⓘ Early stopping is configured with patience of 2 epochs.")

    curves_df_display = filtered_curves.rename(columns={
        "model_name": "Model",
        "epoch": "Epoch",
        "training_loss": "Training Loss",
        "validation_loss": "Validation Loss"
    })

    st.subheader("Training Loss")

    fig = px.line(
        curves_df_display,
        x="Epoch",
        y="Training Loss",
        color="Model",
        markers=True
    )

    fig.update_layout(xaxis_title="Epoch", yaxis_title="Training Loss")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Validation Loss")

    fig2 = px.line(
        curves_df_display,
        x="Epoch",
        y="Validation Loss",
        color="Model",
        markers=True
    )

    fig2.update_layout(xaxis_title="Epoch", yaxis_title="Validation Loss")
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

    cm_display = cm_df_model.rename(columns={
        "true_label": "True Label",
        "predicted_label": "Predicted Label"
    })

    matrix = cm_display.pivot(
        index="True Label",
        columns="Predicted Label",
        values="count"
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"{selected_cm_model} Confusion Matrix"
    )

    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )

    st.plotly_chart(fig, use_container_width=True)