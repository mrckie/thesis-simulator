import streamlit as st
import time
from pathlib import Path
from transformers.pipelines import pipeline

# Import modules
from components.styles import apply_custom_css
from components.baseline_tab import render_baseline_section
from components.comparison_tab import render_comparison_section

from utils.data_loader import (
    load_summary,
    load_curves,
    load_confusion,
    load_dataset_info,
    load_class_imbalance
)

# -------------------------------------------------
# Page Config & Styles
# -------------------------------------------------
st.set_page_config(
    page_title="DistilBERT Architectural Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply our CSS
apply_custom_css()

# -------------------------------------------------
# Page Title
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>DistilBERT Architectural Optimization for Sentiment Analysis</h1>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# File Uploader & Welcome Guide
# -------------------------------------------------
col_left, col_main, col_right = st.columns([8, 84, 8])

with col_main:
    main_container = st.container()
    with main_container:
        uploaded_file = st.file_uploader(
            "Upload Dataset (GoEmotions Only) to proceed",
            type=["csv"],
            label_visibility="hidden" 
        )
        
        if uploaded_file is None:
            # The new Welcome Guide replaces the simple warning
            st.info("""
            **Welcome!** Let's have a quick overview! the web application allows you to interactively explore the results of the experiment, where the researchers mathematically reduced the DistilBERT model to optimize it under small dataset conditions while keeping any performance trade-offs relative to the baseline.

            **How to navigate this tool:**
            1. **Upload the Dataset:** To begin, please upload the specific `GoEmotions` CSV dataset used during the experiment in the dropzone above.
            2. **Explore Section 1 (Baseline Analysis):** The section proves the experimental control. It shows how the original, untouched DistilBERT behaves, establishing a reliable benchmark to compare against.
            3. **Explore Section 2 (Baseline vs. Modified):** The is the core of the research. Here, you can directly compare the optimized, reduced architectures against the heavy baseline. You can view parameter counts, training times, accuracy metrics, and even test the best model in live!
            """)
            st.stop()

        uploaded_name = Path(uploaded_file.name).name.lower()
        if uploaded_name != "goemotions.csv":
            st.error("Invalid dataset. Please upload the GoEmotions file used during experiment.  `goemotions.csv`")
            st.stop()
        else:
            if "uploaded" not in st.session_state:
                success_placeholder = st.empty()
                success_placeholder.success("GoEmotions dataset authenticated! Loading dashboard...")
                time.sleep(1.5)
                success_placeholder.empty()
                st.session_state.uploaded = True

st.divider()

# -------------------------------------------------
# Load Experiment Data & Models
# -------------------------------------------------
summary_df = load_summary().fillna("N/A")
curves_df = load_curves().fillna("N/A")
confusion_df = load_confusion().fillna("N/A")
dataset_info_df = load_dataset_info().fillna("N/A")
class_imbalance_df = load_class_imbalance()

@st.cache_resource(show_spinner="Loading models into memory (this takes a moment)...")
def load_sentiment_models():
    baseline_path = "marckieee/baseline" 
    compressed_path = "marckieee/41.67_percent_reduction" 
    
    baseline_pipe = pipeline("text-classification", model=baseline_path, tokenizer=baseline_path)
    compressed_pipe = pipeline("text-classification", model=compressed_path, tokenizer=compressed_path)
    
    return baseline_pipe, compressed_pipe

# -------------------------------------------------
# Dashboard Navigation (Centered Tabs)
# -------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Select a Section to Explore</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
    .stTabs [data-baseweb="tablist"] {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Create the Tabs
tab_baseline, tab_modified = st.tabs(["Section 1: Baseline Analysis", "Section 2: Baseline vs. Modified"])

# -------------------------------------------------
# Render Sections inside the Tabs
# -------------------------------------------------

with tab_baseline:
    # --- PART 1 EXPLANATION ---
    st.markdown("### Phase 1: Establishing the Ground Truth")
    st.info(
        "**Objective:** Before architectural optimization can begin, a strong and reliable baseline must be established. "
        "This section presents the performance of the original, full-sized DistilBERT architecture. "
        "The researchers analyzed the model using both a massive 58k dataset and a simulated low-resource 12.5k dataset. "
        "The primary goal of this baseline analysis is to rigorously investigate the behaviors and challenges that arise "
        "when the standard baseline model is trained on a constrained, small-scale dataset. Specifically, this phase aims "
        "to identify potential issues such as overfitting, class imbalance sensitivity, and generalization limitations "
        "prior to implementing any architectural reductions."
    )

    render_baseline_section(summary_df, curves_df, confusion_df, dataset_info_df, class_imbalance_df)

with tab_modified:
    # --- PART 2 EXPLANATION ---
    st.markdown("### Phase 2: Architectural Reduction & Comparison")
    st.info(
            "**Objective:** This phase represents the core of the research. The researchers systematically reduced DistilBERT's Attention Heads, "
            "Hidden Dimensions, and Feed-Forward Networks. The tabs below facilitate the evaluation of the trade-offs between "
            "**Computational Efficiency** (i.e., speed, memory) and **Predictive Performance** (i.e., accuracy, f1 score). "
            "Additionally, the best optimized model can be tested interactively against the baseline in the **Sentiment Analysis (Live)** tab."
        )

    # Pass the model loader function so Tab 6 can use it
    render_comparison_section(summary_df, curves_df, confusion_df, load_sentiment_models)