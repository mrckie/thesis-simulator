import streamlit as st
import time

def render_model_testing(selected_models):
    st.header("Model Testing")
    
    # Use the selected_models from the sidebar
    selected_model = st.selectbox(
        "Select Model to Test", 
        options=selected_models
    )
    
    input_text = st.text_area(
        "Comments", 
        placeholder="Type a sentence here..."
    )
    
    if st.button("Test"):
        if not input_text.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner(f"Running inference with {selected_model}..."):
                # Mock inference delay
                time.sleep(1) 
                
                st.success("Complete!")
                st.write(f"**Input:** {input_text}")
                st.info(f"**{selected_model} Prediction:** Joy //Sample Rani Dawg")