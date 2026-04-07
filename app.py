import streamlit as st
import time
from transformers.pipelines import pipeline

from components.styles import apply_custom_css
from components.baseline_tab import render_baseline_section
from components.comparison_tab import render_comparison_section

# Keep your existing data loader
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
    layout="wide"
)

# Apply CSS
apply_custom_css()

# -------------------------------------------------
# Page Title
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>DistilBERT Architectural Optimization Dashboard</h1>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# File Uploader
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
# Render Sections from Components
# -------------------------------------------------

tab1, tab2 = st.tabs(["Baseline Analysis", "Model Comparison"])

with tab1:
    render_baseline_section(summary_df, curves_df, confusion_df, dataset_info_df, class_imbalance_df)

with tab2:
    render_comparison_section(summary_df, curves_df, confusion_df, load_sentiment_models)