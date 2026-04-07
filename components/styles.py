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

        /* 5. Select Section Title */
        .select-section-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin: 0.75rem 0 1rem;
        }
        
        /* 6. Tab styling - center and make bold */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            gap: 2rem;
        }
        
        .stTabs [role="tab"] {
            font-size: 1.3rem;
            font-weight: 700;
            padding: 0.75rem 1.5rem;
        }
        
        .stTabs [role="tab"][aria-selected="true"] {
            font-weight: 800;
        }

    </style>
    """, unsafe_allow_html=True)
    