import streamlit as st

def apply_custom_css():
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
            width: 100% !important; 
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

        /* 4. Hide the Streamlit top menu to permanently lock the theme */
        [data-testid="stHeader"] {
            display: none !important;
        }

    </style>
    """, unsafe_allow_html=True)
    