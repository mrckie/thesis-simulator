import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Uncomment these if you want prediction mode
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="DistilBERT Small Dataset Simulator",
    layout="wide"
)

# ---------------------------
# TITLE & SUBTITLE
# ---------------------------
st.title("📊 DistilBERT Small Dataset Simulator")
st.subheader("Analyze model performance and prediction on small datasets")

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("Simulator Controls")

# Upload CSV
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with text data for prediction",
    type=["csv"]
)

# Dataset size slider
dataset_sizes = [50, 100, 500, 1000]  # Modify based on your experiments
selected_size = st.sidebar.selectbox("Select training dataset size", dataset_sizes)

# Optional: text input
user_param = st.sidebar.text_input("Enter label/parameter (optional)")

# Run button
run_sim = st.sidebar.button("Run Simulation")

# ---------------------------
# LOAD METRICS CSV
# ---------------------------
# Example results CSV structure:
# size,accuracy,f1,train_loss,val_loss
# 50,0.62,0.60,0.48,0.55
metrics_file = Path("results.csv")
if metrics_file.exists():
    metrics_df = pd.read_csv(metrics_file)
else:
    st.error("results.csv not found! Place your Colab metrics CSV here.")
    st.stop()

# ---------------------------
# FILTER METRICS FOR SELECTED SIZE
# ---------------------------
size_metrics = metrics_df[metrics_df["size"] == selected_size]

# ---------------------------
# SIMULATION LOGIC
# ---------------------------
if run_sim:
    st.success(f"Simulation for dataset size {selected_size} samples")

    # ---------------------------
    # DISPLAY METRICS
    # ---------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{size_metrics['accuracy'].values[0]:.2f}")
    col2.metric("F1 Score", f"{size_metrics['f1'].values[0]:.2f}")
    col3.metric("Training Loss", f"{size_metrics['train_loss'].values[0]:.2f}")
    col4.metric("Validation Loss", f"{size_metrics['val_loss'].values[0]:.2f}")

    # ---------------------------
    # METRICS CHARTS
    # ---------------------------
    st.write("### 📈 Metrics Across Dataset Sizes")
    metrics_chart = metrics_df.melt(
        id_vars="size",
        value_vars=["accuracy", "f1", "train_loss", "val_loss"],
        var_name="Metric",
        value_name="Value"
    )
    fig = px.line(metrics_chart, x="size", y="Value", color="Metric", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # DATA TABLE & BAR CHART
    # ---------------------------
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

        st.write("### 📋 Uploaded Data Preview")
        st.dataframe(user_df)

        # Example bar chart using first numeric column if exists
        numeric_cols = user_df.select_dtypes(include="number").columns
        if len(numeric_cols) >= 1:
            st.write("### 📊 Bar Chart of Uploaded Data")
            st.bar_chart(user_df[numeric_cols[0]])
        else:
            st.warning("No numeric columns found for bar chart.")

        # ---------------------------
        # OPTIONAL: LIVE PREDICTION (Uncomment if model is available)
        # ---------------------------
        # st.write("### 🤖 Predictions")
        # model_path = "model"  # folder from Colab
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #
        # texts = user_df['text'].tolist()
        # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # outputs = model(**inputs)
        # preds = torch.argmax(outputs.logits, dim=1)
        # user_df['predicted'] = preds.numpy()
        # st.dataframe(user_df)

else:
    st.info("Upload a CSV and select dataset size, then click 'Run Simulation'")
