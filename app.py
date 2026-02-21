import streamlit as st
import os

# 1. PAGE CONFIGURATION (Must be the first command)
st.set_page_config(
    page_title="AI Nexus: Enterprise Pipeline",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. GLOBAL SESSION STATE INITIALIZATION
# This is critical. Without this, Streamlit will delete the user's dataset 
# every time they click a button or switch a page.
def initialize_system_memory():
    # Data states
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'data_type' not in st.session_state:
        st.session_state.data_type = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Model states
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
        
    # UI States
    if 'system_logs' not in st.session_state:
        st.session_state.system_logs = ["System initialized. Memory allocated."]

initialize_system_memory()

# 3. SIDEBAR BRANDING & LOGS
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.success("Core Engine: Online")
    if st.session_state.raw_data is not None:
        st.info(f"Memory Loaded: {st.session_state.data_type}")
    else:
        st.warning("Memory Empty: Awaiting Ingestion")
    
    st.markdown("---")
    st.markdown("### 📜 System Logs")
    for log in st.session_state.system_logs[-5:]: # Show last 5 logs
        st.caption(f"> {log}")

# 4. ROUTING / NAVIGATION ENGINE
# This uses Streamlit's native routing to link to the separate large files.
pages = {
    "Data Engineering Phase": [
        st.Page("pages/1_ingestion.py", title="1. Multi-Modal Ingestion", icon="📥"),
        st.Page("pages/2_preprocessing.py", title="2. Universal Preprocessing", icon="🛠️"),
    ],
    "Intelligence Phase": [
        st.Page("pages/3_ml_studio.py", title="3. ML Foundry", icon="🧠"),
        st.Page("pages/4_dl_cv_nlp.py", title="4. DL & Vision", icon="🔥"),
    ],
    "Deployment Phase": [
        st.Page("pages/5_evaluation.py", title="5. Evaluation & Export", icon="📦"),
    ]
}

pg = st.navigation(pages)
pg.run()
