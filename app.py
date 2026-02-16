import streamlit as st
import pandas as pd
from pathlib import Path

from components.table_component import display_table
from components.barchart_component import display_metrics_chart

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="DistilBERT Optimization Dashboard", layout="wide")
st.title("DistilBERT Optimization Dashboard", text_alignment='center')
st.subheader("Comparative Analysis: Baseline vs Modified Models", text_alignment='center')

# ---------------- LOAD RESULTS ----------------
metrics_file = Path("data/results.csv")

if not metrics_file.exists():
    st.write("")
    st.error("⚠ results.csv not found in data folder.")
    st.stop()

metrics_df = pd.read_csv(metrics_file)

required_columns = [
    "model",
    "learning_rate",
    "batch_size",
    "epochs",
    "weight_decay",
    "dropout",
    "warmup_steps",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "train_loss",
    "val_loss",
    "train_time",
    "memory_usage"
]

if not all(col in metrics_df.columns for col in required_columns):
    st.write("")
    st.error("Incorrect results.csv structure.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("<h1 style='font-size: 28px; text-align: center;'>Simulation Controls</h1>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("", type=["csv"])

st.sidebar.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

# ---------------- ARCHITECTURE SECTION ----------------
arch_data = pd.DataFrame({
    "Component": ["Attention Heads", "Hidden Size", "FFN Hidden Size"],
    "Baseline": [12, 768, 3072],
    "Modified": [10, 760, 3070]
})

display_table(
    arch_data,
    title="Architectural Comparison",
    show_index=False
)

# ---------------- BASELINE INFO ----------------
st.sidebar.markdown("<h2 style='font-size: 18px; text-align: center;'>Baseline Hyperparameters</h2>", unsafe_allow_html=True)

baseline_row = metrics_df[metrics_df["model"] == "baseline"].iloc[0]

baseline_hyperpara = pd.DataFrame({
    "Hyperparameter": [
        "Learning Rate",
        "Batch Size",
        "Epochs",
        "Weight Decay",
        "Dropout",
        "Warmup Steps"
    ],
    "Value": [
        baseline_row["learning_rate"],
        baseline_row["batch_size"],
        baseline_row["epochs"],
        baseline_row["weight_decay"],
        baseline_row["dropout"],
        baseline_row["warmup_steps"]
    ]
})

display_table(
    baseline_hyperpara,
    show_index=False,
    container=st.sidebar
)

st.sidebar.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

# ---------------- MODIFIED MODEL (DYNAMIC OPTIONS) ----------------
st.sidebar.markdown("<h2 style='font-size: 18px; text-align: center; margin-bottom: 7px;'>Modified Hyperparameters</h2>", unsafe_allow_html=True)

modified_df = metrics_df[metrics_df["model"] == "modified"]

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    sorted(modified_df["learning_rate"].unique())
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    sorted(modified_df["batch_size"].unique())
)

epochs = st.sidebar.selectbox(
    "Number of Epochs",
    sorted(modified_df["epochs"].unique())
)

weight_decay = st.sidebar.selectbox(
    "Weight Decay",
    sorted(modified_df["weight_decay"].unique())
)

dropout = st.sidebar.selectbox(
    "Dropout Rate",
    sorted(modified_df["dropout"].unique())
)

warmup_steps = st.sidebar.selectbox(
    "Warmup Steps",
    sorted(modified_df["warmup_steps"].unique())
)

st.sidebar.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
run_sim = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

# ---------------- RUN SIMULATION ----------------
if run_sim:

    if uploaded_file is None:
        st.write("")
        st.error("⚠ Please upload a dataset CSV file before running simulation.")
        st.stop()

    modified_filtered = modified_df[
        (modified_df["learning_rate"] == learning_rate) &
        (modified_df["batch_size"] == batch_size) &
        (modified_df["epochs"] == epochs) &
        (modified_df["weight_decay"] == weight_decay) &
        (modified_df["dropout"] == dropout) &
        (modified_df["warmup_steps"] == warmup_steps)
    ]

    if modified_filtered.empty:
        st.write("")
        st.error("⚠ No results found for selected hyperparameter combination.")
        st.stop()

    baseline = baseline_row
    modified = modified_filtered.iloc[0]

    def compute_variance(base, mod, higher_is_better=True):
        gap = abs(mod - base)
        improved = mod > base if higher_is_better else mod < base
        color = "green" if improved else "red"
        return gap, color

    rows = []

    metrics_info = [
        ("Accuracy", "accuracy", True, "%"),
        ("Precision", "precision", True, "%"),
        ("Recall", "recall", True, "%"),
        ("F1 Score", "f1", True, "%"),
        ("Training Loss", "train_loss", False, ""),
        ("Validation Loss", "val_loss", False, ""),
        ("Training Time", "train_time", False, "min"),
        ("Memory Usage", "memory_usage", False, "GB"),
    ]

    for name, key, higher_is_better, unit in metrics_info:
        base = baseline[key]
        mod = modified[key]
        gap, color = compute_variance(base, mod, higher_is_better)

        if unit == "%":
            base_fmt = f"{base:.1f}%"
            mod_fmt = f"{mod:.1f}%"
            gap_fmt = f"{gap:.2f}%"
        elif unit == "min":
            base_fmt = f"{base:.1f} min"
            mod_fmt = f"{mod:.1f} min"
            gap_fmt = f"{gap:.1f} min"
        elif unit == "GB":
            base_fmt = f"{base:.1f} GB"
            mod_fmt = f"{mod:.1f} GB"
            gap_fmt = f"{gap:.1f} GB"
        else:
            base_fmt = f"{base:.3f}"
            mod_fmt = f"{mod:.3f}"
            gap_fmt = f"{gap:.3f}"

        variance_html = f"<span style='color:{color}; font-weight:bold;'>{gap_fmt}</span>"
        rows.append([name, base_fmt, mod_fmt, variance_html])

    detailed_df = pd.DataFrame(
        rows,
        columns=["Metric", "Baseline Model", "Modified Model", "Variance"]
    )

    display_table(
    detailed_df,
    title="Detailed Metrics Table",
    show_index=False
    )

    display_metrics_chart(detailed_df)

else:
    st.write("")
    st.info("Upload dataset and click Run Simulation.")
