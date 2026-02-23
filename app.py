import streamlit as st
from tabs.confusion_matrix import render_confusion_matrix
from tabs.efficiency import render_efficiency
from tabs.overview import render_overview
from tabs.performance import render_performance
from tabs.training_curves import render_training_curves
from utils.Sidebar import ShowSidebar
from utils.data_loader import load_summary, load_curves, load_confusion
from utils.model_testing import render_model_testing

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Model Compression Evaluation Dashboard ")

# Sidebar


# Load Data
summary_df = load_summary()
curves_df = load_curves()
confusion_df = load_confusion()

selected_models = ShowSidebar(summary_df)

render_model_testing(selected_models)



# Filtered Data
filtered_summary = summary_df[summary_df["model_name"].isin(selected_models)]
filtered_curves = curves_df[curves_df["model_name"].isin(selected_models)]

# Render Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Performance", "Efficiency", "Training Curves", "Confusion Matrix"]
)

# Overview
with tab1:
    render_overview(filtered_summary)
# Performance
with tab2:
    render_performance(filtered_summary)
# Efficiency
with tab3:
    render_efficiency(filtered_summary)
# Training Curves
with tab4:
    render_training_curves(filtered_summary, filtered_curves)
# Confusion Matrix
with tab5:
    render_confusion_matrix(confusion_df, selected_models)
