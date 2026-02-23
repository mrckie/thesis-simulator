import time
import streamlit as st

def ShowSidebar(summary_df):
    with st.sidebar:
        # --- Dataset Upload ---
        st.header("Dataset Upload")
        uploaded_file = st.file_uploader(
            "Upload Dataset (GoEmotions Only)",
            type=["csv"]
        )
        
        if uploaded_file is None:
            st.warning("Please upload the GoEmotions dataset to proceed.")
            st.stop()
        if "goemotions" not in uploaded_file.name.lower():
            st.error("Invalid dataset. Please upload the GoEmotions dataset")
            st.stop()
        else:
            if "uploaded" not in st.session_state:
                success_placeholder = st.empty()
                success_placeholder.success("GoEmotions dataset uploaded successfully.")
                time.sleep(1)
                success_placeholder.empty()
                st.session_state.uploaded = True
        
        # --- Model Selection ---
        st.header("Model Selection")
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=summary_df["model_name"].unique(),
            default=summary_df["model_name"].unique()[:2]
        )

        if len(selected_models) < 2:
            st.warning("Select at least two models.")
            st.stop()

        return selected_models