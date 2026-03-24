import streamlit as st
import pandas as pd
import plotly.express as px
import time
from transformers.pipelines import pipeline

from utils.data_loader import (
    load_summary,
    load_curves,
    load_confusion,
    load_dataset_info,
    load_class_imbalance
)

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="DistilBERT Architectural Optimization Dashboard",
    layout="wide"
)

# --- CSS FOR BIG UPLOAD BUTTON & ALIGNMENT ---
st.markdown("""
<style>
    /* 1. Target the file uploader section to make it massive */
    div[data-testid="stFileUploader"] section {
        min-height: 300px !important; 
        padding: 50px !important; 
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important; 
        border-width: 2px !important; 
        width: 100% !important; /* Stretch to fill container */
    }
    
    /* Force inner divs of the dropzone to center */
    div[data-testid="stFileUploader"] section > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* 2. Make the 'Browse files' button bigger and centered */
    div[data-testid="stFileUploader"] button {
        height: 60px !important;
        font-size: 20px !important;
        font-weight: bold !important;
        padding: 0 30px !important;
        margin: 0 auto !important;
    }
    
    /* 3. Center subheaders */
    h3, .stSubheader {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Page Title - Centered via HTML
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>DistilBERT Architectural Optimization for Sentiment Analysis</h1>",
    unsafe_allow_html=True
)


# =================================================
# MAIN PANEL LAYOUT (8% Left | 84% Center | 8% Right)
# =================================================
col_left, col_main, col_right = st.columns([8, 84, 8])

with col_main:
    # We wrap the uploader and info box in a single container
    main_container = st.container()
    
    with main_container:
        
        # Hidden label so the text inside the dropzone stays perfectly centered
        uploaded_file = st.file_uploader(
            "Upload Dataset (GoEmotions Only) to proceed",
            type=["csv"],
            label_visibility="hidden" 
        )
        
        # -------------------------------------------------
        # NOTIFICATIONS 
        # -------------------------------------------------
        if uploaded_file is None:
            st.info("Please upload the GoEmotions dataset used during experiment.")
            st.stop()

        if "goemotions" not in uploaded_file.name.lower():
            st.error("Invalid dataset. Please upload the GoEmotions dataset used during experiment.")
            st.stop()
        else:
            if "uploaded" not in st.session_state:
                success_placeholder = st.empty()
                success_placeholder.success("GoEmotions dataset uploaded successfully. Loading dashboard...")
                time.sleep(1)
                success_placeholder.empty()
                st.session_state.uploaded = True

        st.markdown("<br>", unsafe_allow_html=True)
        

st.divider()

# -------------------------------------------------
# Load Experiment Data
# -------------------------------------------------
summary_df = load_summary()
curves_df = load_curves()
confusion_df = load_confusion()
dataset_info_df = load_dataset_info()
class_imbalance_df = load_class_imbalance()

# Loading the model from hugging face
@st.cache_resource(show_spinner="Loading models into memory (this takes a moment)...")
def load_sentiment_models():
    # Just the username/repo-name! No https:// or /tree/main
    baseline_path = "marckieee/baseline" 
    compressed_path = "marckieee/41.67_percent_reduction" 
    
    baseline_pipe = pipeline("text-classification", model=baseline_path, tokenizer=baseline_path)
    compressed_pipe = pipeline("text-classification", model=compressed_path, tokenizer=compressed_path)
    
    return baseline_pipe, compressed_pipe

# FIX: Replace all Pandas NaN values with the string "N/A"
summary_df = summary_df.fillna("N/A")
curves_df = curves_df.fillna("N/A")
confusion_df = confusion_df.fillna("N/A")
dataset_info_df = dataset_info_df.fillna("N/A")


# =================================================
# EXPERIMENT PIPELINE
# =================================================
st.markdown("<h1 style='text-align: center;'>DistilBERT Optimization Experiment Pipeline</h1>", unsafe_allow_html=True)

st.info(
"""
The experiment follows multiple phases based on the objectives:

Phase 1 — Match Baseline Training Setup  
Train the original DistilBERT using the 58k dataset to observe its training behavior.

Phase 2 — Reduced Dataset Investigation  
Reduce the dataset to 12.5k with the same architecture and hyperparameters to simulate a low-resource condition.

Phase 3 — Strong Baseline Construction  
Build a controlled baseline after identifying issues during analysis to ensure stable and fair comparison.

Phase 4 — Architectural Reduction  
Reduce attention heads and FFN size, then compare all models using the same dataset, preprocessing, and splits.
"""
)

st.divider()

# -------------------------------------------------------------------------
# PART 1: BASELINE ANALYSIS
# -------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Baseline Analysis</h1>", unsafe_allow_html=True)

baseline_names = ["baseline_58k_analysis", "baseline_12.5k_analysis"]
baseline_summary = summary_df[summary_df["model_name"].isin(baseline_names)].copy()

# Define Tabs specific to Part 1 (No Efficiency tab)
base_tab1, base_tab2, base_tab3, base_tab4 = st.tabs([
    "Dataset Configuration",
    "Classification Performance",
    "Training Curves",
    "Confusion Matrix"
])

# --- TAB 1: DATASET CONFIGURATION & IMBALANCE ---
with base_tab1:
    st.markdown("### Dataset Configuration")

    dataset_df = dataset_info_df.copy()
    dataset_df = dataset_df.rename(columns={
        "dataset": "Dataset",
        "total": "Total Samples",
        "train": "Train",
        "val": "Validation",
        "test": "Test"
    })
    dataset_df["Dataset"] = dataset_df["Dataset"].map({
        "baseline_58k_analysis": "58k Baseline Analysis",
        "baseline_12.5k_analysis": "12.5k Baseline Analysis",
    })

    plot_df = dataset_df.melt(
        id_vars="Dataset",
        value_vars=["Train", "Validation", "Test"], 
        var_name="Split",
        value_name="Samples"
    )
    plot_df["Samples"] = pd.to_numeric(plot_df["Samples"], errors='coerce')
    plot_df = plot_df.dropna(subset=["Samples"])

    fig_dataset = px.bar(
        plot_df, x="Dataset", y="Samples", color="Split", barmode="group", text="Samples"
    )
    fig_dataset.update_traces(textposition='outside')
    fig_dataset.update_layout(xaxis_title="Dataset", yaxis_title="Number of Samples", yaxis_range=[0, plot_df["Samples"].max() * 1.2])
    st.plotly_chart(fig_dataset, use_container_width=True)

    st.markdown("### Class Imbalance")

    imbalance_df = class_imbalance_df.copy()
    imbalance_df['Dataset'] = imbalance_df['dataset'].map({
        "baseline_58k_analysis": "58k Baseline Analysis",
        "baseline_12.5k_analysis": "12.5k Baseline Analysis",
    })
    imbalance_df = imbalance_df.dropna(subset=['Dataset'])
    imbalance_df['Class'] = imbalance_df['target'].map({0: "Negative", 1: "Positive"})
    imbalance_df['Proportion (%)'] = (imbalance_df['proportion'] * 100).round(1)

    fig_imbalance = px.bar(
        imbalance_df, x="Dataset", y="Proportion (%)", color="Class", barmode="group", text="Proportion (%)"
    )
    fig_imbalance.update_traces(textposition='outside')
    fig_imbalance.update_layout(xaxis_title="Dataset", yaxis_title="Proportion (%)", yaxis_range=[0, 115])
    st.plotly_chart(fig_imbalance, use_container_width=True)

# --- TAB 2: CLASSIFICATION REPORTS ---
with base_tab2:
    st.markdown("### Classification Performance")

    class_report_table = baseline_summary[
        ["model_name", "accuracy", "precision_macro_avg", "recall_macro_avg", "f1_macro_avg", 
         "precision_weighted_avg", "recall_weighted_avg", "f1_weighted_avg"]
    ].rename(columns={
        "model_name": "Model",
        "accuracy": "Accuracy",
        "precision_macro_avg": "Precision (Macro)",
        "recall_macro_avg": "Recall (Macro)",
        "f1_macro_avg": "F1 (Macro)",
        "precision_weighted_avg": "Precision (Weighted)",
        "recall_weighted_avg": "Recall (Weighted)",
        "f1_weighted_avg": "F1 (Weighted)"
    })

    # Format to percentages
    for col in class_report_table.columns[1:]:
        class_report_table[col] = (pd.to_numeric(class_report_table[col], errors='coerce') * 100).round(2).astype(str) + "%"
        
    st.dataframe(class_report_table, use_container_width=True)


# --- TAB 3: TRAINING CURVES ---
with base_tab3:
    st.markdown("### Training Curves")

    # Isolate baseline curves
    base_curves = curves_df[curves_df["model_name"].isin(baseline_names)].copy()
    base_curves["training_loss"] = pd.to_numeric(base_curves["training_loss"], errors='coerce')
    base_curves["validation_loss"] = pd.to_numeric(base_curves["validation_loss"], errors='coerce')

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.write("**58k Baseline**")
        curve_58k = base_curves[base_curves["model_name"] == "baseline_58k_analysis"]
        if not curve_58k.empty:
            # 58k only has training loss, val is N/A
            fig_c1 = px.line(curve_58k, x="epoch", y=["training_loss"], markers=True)
            fig_c1.update_layout(xaxis_title="Epoch", yaxis_title="Loss", showlegend=True)
            st.plotly_chart(fig_c1, use_container_width=True)
            
    with col_c2:
        st.write("**12.5k Baseline**")
        curve_12k = base_curves[base_curves["model_name"] == "baseline_12.5k_analysis"]
        if not curve_12k.empty:
            fig_c2 = px.line(curve_12k, x="epoch", y=["training_loss", "validation_loss"], markers=True)
            fig_c2.update_layout(xaxis_title="Epoch", yaxis_title="Loss", showlegend=True)
            st.plotly_chart(fig_c2, use_container_width=True)

# --- TAB 4: CONFUSION MATRICES ---
with base_tab4:
    st.markdown("### Confusion Matrices")

    cm_base = confusion_df[confusion_df["model_name"].isin(baseline_names)].copy()

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.write("**58k Baseline CM**")
        cm_58k = cm_base[cm_base["model_name"] == "baseline_58k_analysis"]
        if not cm_58k.empty:
            matrix_58k = cm_58k.pivot(index="true_label", columns="predicted_label", values="count")
            fig_m1 = px.imshow(matrix_58k, text_auto=True, color_continuous_scale="Blues")
            fig_m1.update_layout(xaxis_title="Predicted", yaxis_title="True")
            st.plotly_chart(fig_m1, use_container_width=True)

    with col_m2:
        st.write("**12.5k Baseline CM**")
        cm_12k = cm_base[cm_base["model_name"] == "baseline_12.5k_analysis"]
        if not cm_12k.empty:
            matrix_12k = cm_12k.pivot(index="true_label", columns="predicted_label", values="count")
            fig_m2 = px.imshow(matrix_12k, text_auto=True, color_continuous_scale="Blues")
            fig_m2.update_layout(xaxis_title="Predicted", yaxis_title="True")
            st.plotly_chart(fig_m2, use_container_width=True)

st.divider()

# -------------------------------------------------------------------------
# PART 2: BASELINE VS MODIFIED COMPARISON
# -------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Baseline vs Modified Architectures</h1>", unsafe_allow_html=True)


# --- GLOBAL ENVIRONMENT & MODEL SELECTION ---
st.markdown("### Environment & Model Selection")

env_col, mod_col = st.columns(2)

with env_col:
    selected_environment = st.selectbox(
        "Select Environment",
        options=["gpu_t4", "cpu_default"],
        index=0
    )

# Filter Data by Environment immediately
env_summary = summary_df[summary_df["environment"].isin([selected_environment, "N/A"])]

# 1. Get all unique models for the environment
all_env_models = env_summary["model_name"].unique()

# 2. Filter out the analysis baselines so they don't appear in the dropdown
models_to_hide = ["baseline_58k_analysis", "baseline_12.5k_analysis"]
available_models = [m for m in all_env_models if m not in models_to_hide]

# 3. Define your desired defaults
desired_defaults = ["baseline_12.5k_comparison", "33.33% reduction"]

# Safety check: Only apply defaults if they actually exist in the available options
valid_defaults = [m for m in desired_defaults if m in available_models]

with mod_col:
    selected_models = st.multiselect(
        "Select Models to Compare / Show",
        options=available_models,
        default=valid_defaults
    )

# -------------------------------------------------------------------------
# GLOBAL DATA FILTERING
# -------------------------------------------------------------------------

# Final Filtered Data based on selected models
filtered_summary = env_summary[env_summary["model_name"].isin(selected_models)].copy()

# Safely clean numeric columns globally here so ALL tabs work flawlessly
filtered_summary["parameter_count"] = filtered_summary["parameter_count"].astype(str).str.replace(',', '', regex=False)
filtered_summary["parameter_count"] = pd.to_numeric(filtered_summary["parameter_count"], errors='coerce').fillna(0)
filtered_summary["accuracy"] = pd.to_numeric(filtered_summary["accuracy"], errors='coerce').fillna(0)
filtered_summary["reduction_percent"] = pd.to_numeric(filtered_summary["reduction_percent"], errors='coerce').fillna(0)

env_curves = curves_df[curves_df["environment"].isin([selected_environment, "N/A"])]
filtered_curves = env_curves[env_curves["model_name"].isin(selected_models)]

env_confusion = confusion_df[confusion_df["environment"].isin([selected_environment, "N/A"])]
filtered_confusion = env_confusion[env_confusion["model_name"].isin(selected_models)]


# Define Tabs specific to Part 2 (We now have 6 tabs!)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Architecture & Size",     
    "Performance Comparison",   
    "Computational Efficiency",                
    "Training Curves",          
    "Confusion Matrix",         
    "Sentiment Analysis (Live)" 
])


# =================================================
# TAB 1 — ARCHITECTURE & PARAMETER COUNT
# =================================================
with tab1:
    st.markdown("### Architecture Configuration")

    # Simple sort just for architecture
    sort_by_arch = st.selectbox("Sort Table By", ["Parameter Count", "Reduction (%)"], key="arch_sort")
    sort_col = "parameter_count" if sort_by_arch == "Parameter Count" else "reduction_percent"
    
    # Sort so the largest model (baseline) sits naturally at the top
    arch_df = filtered_summary.sort_values(sort_col, ascending=False)

    architecture_table = arch_df[
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
        "model_name": "Model",
        "reduction_percent": "Reduction (%)",
        "attention_heads": "Heads",
        "hidden_dim": "Hidden",
        "ffn": "FFN",
        "parameter_count": "Parameters",
        "trained_epochs": "Epochs"
    })

    st.dataframe(architecture_table.fillna("N/A"), use_container_width=True)

    st.markdown("### Model Parameter Count")
    
    # Replaced the Line Chart with a clean Bar Chart showing only parameters
    fig_params = px.bar(
        architecture_table,
        x="Model",
        y="Parameters",
        text="Parameters",
        color="Model",
    )
    # Format the text on the bars with commas (e.g., 66,955,010)
    fig_params.update_traces(texttemplate='%{text:,}', textposition='outside') 
    fig_params.update_layout(
        xaxis_title="Model", 
        yaxis_title="Total Parameters",
        yaxis_range=[0, architecture_table["Parameters"].max() * 1.2] # Give headroom for the text above the bars
    )
    st.plotly_chart(fig_params, use_container_width=True)

# =================================================
# TAB 2 — PREDICTIVE PERFORMANCE COMPARISON 
# =================================================
with tab2:
    st.markdown("### Predictive Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        metric_options = {
            "Accuracy": "accuracy",
            "Precision (Weighted Average)": "precision_weighted_avg",
            "Precision (Macro Average)": "precision_macro_avg",
            "Recall (Weighted Average)": "recall_weighted_avg",
            "Recall (Macro Average)": "recall_macro_avg",
            "F1 Score (Weighted Average)": "f1_weighted_avg",
            "F1 Score (Macro Average)": "f1_macro_avg"
        }
        selected_metric_label = st.selectbox("Select Metric to Visualize", list(metric_options.keys()))
        detailed_metric_choice = metric_options[selected_metric_label]

    with col2:
        sort_by_perf = st.selectbox("Sort Models By", ["Highest Score", "Model Name"], key="perf_sort")

    perf_df = filtered_summary.rename(columns={"model_name": "Model"}).copy()
    
    # Ensure the detailed metric is numeric before charting
    perf_df[detailed_metric_choice] = pd.to_numeric(perf_df[detailed_metric_choice], errors='coerce').fillna(0)

    # Apply Sorting logic
    if sort_by_perf == "Highest Score":
        perf_df = perf_df.sort_values(detailed_metric_choice, ascending=False)
    else:
        perf_df = perf_df.sort_values("Model")

    fig_detailed = px.bar(
        perf_df,
        x="Model",
        y=detailed_metric_choice,
        color="Model",
        text_auto=True
    )

    fig_detailed.update_traces(texttemplate='%{y:.2%}', textposition='outside')
    fig_detailed.update_layout(
        xaxis_title="Model",
        yaxis_title=selected_metric_label,
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.1] # Gives headroom so the percentage text doesn't get cut off
    )

    st.plotly_chart(fig_detailed, use_container_width=True)

    st.markdown("#### Detailed Classification Metrics Table")

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
        classification_table_pct[col] = (pd.to_numeric(classification_table_pct[col], errors='coerce') * 100).round(2).astype(str) + "%"

    st.dataframe(classification_table_pct, use_container_width=True)


# =================================================
# TAB 3 — COMPUTATIONAL EFFICIENCY
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

    efficiency_df = filtered_summary.copy()
    
    # Safely convert to numeric before doing math
    efficiency_df["train_time"] = pd.to_numeric(efficiency_df["train_time"], errors='coerce').fillna(0)
    efficiency_df["peak_gpu_usage"] = pd.to_numeric(efficiency_df["peak_gpu_usage"], errors='coerce').fillna(0)
    efficiency_df["peak_ram_usage"] = pd.to_numeric(efficiency_df["peak_ram_usage"], errors='coerce').fillna(0)
    
    if time_unit == "Minutes":
        efficiency_df["train_time_display"] = efficiency_df["train_time"] / 60
        time_label = "Training Time (Minutes)"
    else:
        efficiency_df["train_time_display"] = efficiency_df["train_time"]
        time_label = "Training Time (Seconds)"

    if memory_unit == "MB":
        efficiency_df["gpu_memory_display"] = efficiency_df["peak_gpu_usage"] * 1000
        efficiency_df["cpu_memory_display"] = efficiency_df["peak_ram_usage"] * 1000
        gpu_memory_label = "Peak GPU Memory Usage (MB)"
        cpu_memory_label = "Peak CPU Memory Usage (MB)"
    else:
        efficiency_df["gpu_memory_display"] = efficiency_df["peak_gpu_usage"]
        efficiency_df["cpu_memory_display"] = efficiency_df["peak_ram_usage"]
        gpu_memory_label = "Peak GPU Memory Usage (GB)"
        cpu_memory_label = "Peak CPU Memory Usage (GB)"

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
# =================================================
# TAB 4 — TRAINING CURVES
# =================================================
with tab4:

    st.subheader("Epochs Configuration")
    
    epochs_info = filtered_summary[
        ["model_name", "epochs", "trained_epochs"]
    ].rename(columns={
        "model_name": "Model Name",
        "epochs": "Hyperparameter Epochs",
        "trained_epochs": "Actual Epochs (with Early Stopping)"
    }).drop_duplicates()
    
    # Safely handle numeric conversion
    for col in ["Actual Epochs (with Early Stopping)", "Hyperparameter Epochs"]:
        epochs_info[col] = pd.to_numeric(epochs_info[col], errors='coerce').fillna(0)
    
    # FIX: Dynamically create exactly enough columns for the number of models selected
    num_models = len(epochs_info)
    cols = st.columns(num_models)
    
    # Loop through the columns and place one metric in each
    for idx, row in epochs_info.reset_index(drop=True).iterrows():
        actual = int(row['Actual Epochs (with Early Stopping)'])
        hyper = int(row['Hyperparameter Epochs'])
        delta_val = actual - hyper
        
        with cols[idx]:
            st.metric(row["Model Name"], f"{actual}/{hyper}", delta=f"{delta_val} epochs")
    
    st.info("ⓘ Early stopping is configured with patience of 3 epochs.")

    curves_df_display = filtered_curves.copy()
    curves_df_display["training_loss"] = pd.to_numeric(curves_df_display["training_loss"], errors='coerce').fillna(0)
    curves_df_display["validation_loss"] = pd.to_numeric(curves_df_display["validation_loss"], errors='coerce').fillna(0)
    
    curves_df_display = curves_df_display.rename(columns={
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

    cm_df_model = filtered_confusion[
        filtered_confusion["model_name"] == selected_cm_model
    ].copy()

    # FIX 3: Ensure count is strictly numeric before Plotly generates the heatmap
    cm_df_model["count"] = pd.to_numeric(cm_df_model["count"], errors='coerce').fillna(0)

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

# =================================================
# TAB 6 — SENTIMENT ANALYSIS (LIVE APPLICATION)
# =================================================
with tab6:
    st.markdown("### Live Sentiment Analysis Application")
    st.info("Test the baseline and the optimized architecture in a real sentiment analysis inference task.")

    # Get user input
    user_input = st.text_area(
        "Enter a review or sentence:", 
        placeholder="e.g., I feel happy today.",
        height=100
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            try:
                # Load models
                baseline_model, compressed_model = load_sentiment_models()
                
                # Run inference
                base_result = baseline_model(user_input)[0]
                comp_result = compressed_model(user_input)[0]
                
                # Map 0 and 1 to your actual labels (Change these if your model uses different mappings)
                label_map = {"LABEL_0": "Negative 🔴", "LABEL_1": "Positive 🟢"}
                
                base_label = label_map.get(base_result['label'], base_result['label'])
                base_score = base_result['score']
                
                comp_label = label_map.get(comp_result['label'], comp_result['label'])
                comp_score = comp_result['score']
                
                # Display Results Side-by-Side
                st.markdown("#### Inference Results")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("**Baseline Model**")
                    st.metric(label="Prediction", value=base_label)
                    st.progress(base_score, text=f"Confidence: {base_score:.2%}")
                    
                with res_col2:
                    st.markdown("**Optimized Model (41.67% Architecture Reduction)**")
                    st.metric(label="Prediction", value=comp_label)
                    st.progress(comp_score, text=f"Confidence: {comp_score:.2%}")
                    
                if base_label != comp_label:
                    st.warning("The models disagree on this sentiment!")
                else:
                    st.success("Both models agree on the sentiment!")
                    
            except Exception as e:
                st.error(f"Error loading or running models: {e}")
                st.markdown("*(Did you remember to replace the 'your-username/...' paths in the code with your actual Hugging Face model IDs?)*")