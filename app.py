import streamlit as st
import pandas as pd
from pathlib import Path

from components.table_component import display_table
from components.barchart_component import display_metrics_chart

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="DistilBERT Optimization Dashboard", layout="wide")
st.title("DistilBERT Optimization Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Simulation Controls")

uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])

dataset_sizes = [50, 100, 500, 1000]
selected_size = st.sidebar.selectbox(
    "Select Dataset Size",
    ["Select Size"] + dataset_sizes
)

run_sim = st.sidebar.button("Run Simulation")

# ---------------- RUN SIMULATION ----------------
if run_sim:

    # Validation
    if uploaded_file is None:
        st.error("⚠ Please upload a dataset CSV file before running simulation.")
        st.stop()

    if selected_size == "Select Size":
        st.error("⚠ Please select a dataset size.")
        st.stop()

    metrics_file = Path("data/results.csv")  # You can change path

    if not metrics_file.exists():
        st.error("⚠ results.csv not found in data folder.")
        st.stop()

    metrics_df = pd.read_csv(metrics_file)

    required_columns = [
        "model", "size", "accuracy", "precision", "recall", "f1",
        "train_loss", "val_loss", "train_time", "memory_usage"
    ]

    if not all(col in metrics_df.columns for col in required_columns):
        st.error("Incorrect results.csv structure.")
        st.stop()

    size_data = metrics_df[metrics_df["size"] == selected_size]

    baseline = size_data[size_data["model"] == "baseline"].iloc[0]
    modified = size_data[size_data["model"] == "modified"].iloc[0]

    # ------------- VARIANCE FUNCTION -------------
    def compute_variance(base, mod, higher_is_better=True):
        gap = abs(mod - base)
        improved = mod > base if higher_is_better else mod < base
        color = "green" if improved else "red"
        return gap, improved, color

    rows = []
    metrics_info = [
        ("Accuracy", "accuracy", True, "%"),
        ("Precision", "precision", True, "%"),
        ("Recall", "recall", True, "%"),
        ("F1 Score", "f1", True, "%"),
        ("Training Loss", "train_loss", False, "%"),
        ("Validation Loss", "val_loss", False, "%"),
        ("Training Time", "train_time", False, "min"),
        ("Memory Usage", "memory_usage", False, "GB"),
    ]

    for name, key, higher_is_better, unit in metrics_info:
        base = baseline[key]
        mod = modified[key]
        gap, improved, color = compute_variance(base, mod, higher_is_better)

        if unit == "%":
            base_fmt = f"{base:.1f}%"
            mod_fmt = f"{mod:.1f}%"
            gap_fmt = f"{gap:.2f}%"
        elif unit == "min":
            base_fmt = f"{base:.1f} min"
            mod_fmt = f"{mod:.1f} min"
            gap_fmt = f"{gap:.1f} min"
        else:
            base_fmt = f"{base:.1f} GB"
            mod_fmt = f"{mod:.1f} GB"
            gap_fmt = f"{gap:.1f} GB"

        variance_html = f"<span style='color:{color}; font-weight:bold;'>{gap_fmt}</span>"
        rows.append([name, base_fmt, mod_fmt, variance_html])

    detailed_df = pd.DataFrame(
        rows,
        columns=["Metric", "Baseline Model", "Modified Model", "Variance"]
    )

    # ---------------- DISPLAY ----------------
    display_table(detailed_df)
    display_metrics_chart(detailed_df)

else:
    st.info("Upload dataset and select size before running simulation.")
