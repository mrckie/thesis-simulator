import streamlit as st
import pandas as pd
import time
import re
from pathlib import Path
from transformers.pipelines import pipeline
from spellchecker import SpellChecker
import streamlit.components.v1 as components

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

# Apply custom CSS
apply_custom_css()

# Initialize State Manager for UI Routing
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"

# -------------------------------------------------
# Global Page Title
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>DistilBERT Architectural Optimization</h1>",
    unsafe_allow_html=True
)
st.divider()

# ADD THIS LINE RIGHT HERE:
top_anchor = st.empty()

# =====================================================================
# VIEW 1: HOME PAGE (Sentiment App + Dataset Gateway)
# =====================================================================
if st.session_state.current_view == "home":
    
    @st.cache_resource(show_spinner="Loading models into memory (this takes a moment)...")
    def load_sentiment_models():
        baseline_path = "marckieee/baseline" 
        compressed_path = "marckieee/41.67_percent_reduction" 
        
        baseline_pipe = pipeline("text-classification", model=baseline_path, tokenizer=baseline_path)
        compressed_pipe = pipeline("text-classification", model=compressed_path, tokenizer=compressed_path)
        
        return baseline_pipe, compressed_pipe

    # Warm up models immediately
    baseline_model, compressed_model = load_sentiment_models()

    st.markdown("### Live Sentiment Analysis")
    st.info("Test the optimized DistilBERT model against the baseline in real-time. Enter a sentence below to see how both architectures predict your emotion.")

    user_input = st.text_area("Enter a short sentence expressing emotion or sentiment:", placeholder="e.g., I feel incredibly happy today.", height=100)

    if st.button("Analyze Sentiment", type="primary"):
        spell = SpellChecker()
        is_valid = True
        
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
            is_valid = False
        elif len(user_input.strip()) < 3:
            st.warning("Input is too short. Please enter a meaningful sentence expressing an emotion.")
            is_valid = False
        elif len(user_input) > 1000:
            st.warning("Input is too long! Please limit your text to a short paragraph.")
            is_valid = False
        elif not re.search('[a-zA-Z]', user_input):
            st.error("Invalid Input: The inputted data consists only of symbols or numbers.")
            is_valid = False
                
        if is_valid:
            words = re.findall(r'\b[A-Za-z]+\b', user_input)
            lowercase_words = [w for w in words if not w[0].isupper()]
            misspelled = spell.unknown(lowercase_words)
            
            if misspelled:
                bad_words = ", ".join(misspelled)
                st.warning(f"Spelling Check: Please correct the following unrecognized words before analyzing: **{bad_words}**")
                is_valid = False

        if is_valid:
            try:
                base_result = baseline_model(user_input)[0]
                comp_result = compressed_model(user_input)[0]
                
                label_map = {"LABEL_0": "Negative 🔴", "LABEL_1": "Positive 🟢"}
                
                base_label = label_map.get(base_result['label'], base_result['label'])
                base_score = base_result['score']
                comp_label = label_map.get(comp_result['label'], comp_result['label'])
                comp_score = comp_result['score']
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("**Baseline Model**")
                    st.metric(label="Prediction", value=base_label)
                    st.progress(base_score, text=f"Confidence: {base_score:.2%}")
                    
                with res_col2:
                    st.markdown("**Optimized Model (41.67% Reduction)**")
                    st.metric(label="Prediction", value=comp_label)
                    st.progress(comp_score, text=f"Confidence: {comp_score:.2%}")
                    
            except Exception as e:
                st.error(f"Error running inference: {e}")

    st.divider()

    st.markdown("### Explore the Experimental Metrics")
    st.info("To view the experimental metrics, training logs, and baseline comparisons, please upload the dataset used during the experiment.")

    col_left, col_main, col_right = st.columns([8, 84, 8])

    with col_main:
        main_container = st.container()
        with main_container:
            uploaded_file = st.file_uploader(
                "Upload Dataset (GoEmotions Only) to proceed",
                type=["csv", "zip"],
                label_visibility="hidden" 
            )
            
            if uploaded_file is not None:
                uploaded_name = Path(uploaded_file.name).name.lower()
                if uploaded_name not in ["goemotions.csv", "goemotions.zip"]:
                    st.error("Invalid dataset. Please upload the GoEmotions file used during the experiment (`goemotions.csv` or `goemotions.zip`).")
                else:
                    st.success("Dataset authenticated! Here is a quick preview of the contents:")
                    
                    try:
                        preview_df = pd.read_csv(uploaded_file, nrows=15)
                        st.dataframe(preview_df, use_container_width=True)
                    except Exception:
                        pass 
                    
                    # Routing Button: Changes the view state and refreshes the page
                    if st.button("Proceed to Dashboard", type="primary"):
                        st.session_state.current_view = "metrics"
                        st.rerun()


# =====================================================================
# VIEW 2: METRICS DASHBOARD (Only shows when current_view == "metrics")
# =====================================================================
elif st.session_state.current_view == "metrics":
    
    # NEW SCROLL METHOD: Tell Streamlit to scroll up to the anchor
    # We put it inside a small time delay so it fires AFTER the charts load
    time.sleep(0.1) 
    st.components.v1.html(
        """
        <script>
            window.parent.scrollTo({ top: 0, behavior: 'smooth' });
        </script>
        """, 
        height=0
    )

    # Back Navigation Button
    if st.button("← Back to Sentiment App & Uploader"):
        st.session_state.current_view = "home"
        st.rerun()

    st.divider()

    # Load Data
    summary_df = load_summary().fillna("N/A")
    curves_df = load_curves().fillna("N/A")
    confusion_df = load_confusion().fillna("N/A")
    dataset_info_df = load_dataset_info().fillna("N/A")
    class_imbalance_df = load_class_imbalance()

    st.markdown("<h3 style='text-align: center;'>Select a Section to Explore</h3>", unsafe_allow_html=True)

    # CSS trick to force Streamlit Tabs to center themselves
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tablist"] {
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)

    tab_baseline, tab_modified = st.tabs(["Section 1: Baseline Analysis", "Section 2: Baseline vs. Modified"])

    with tab_baseline:
        st.markdown("### Phase 1: Establishing the Ground Truth")
        render_baseline_section(summary_df, curves_df, confusion_df, dataset_info_df, class_imbalance_df)

    with tab_modified:
        st.markdown("### Phase 2: Architectural Reduction & Comparison")
        render_comparison_section(summary_df, curves_df, confusion_df)