import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from PIL import Image

# 🚀 NEW: Import the Google Drive Engine
try:
    from utils.drive_handler import download_csv_from_drive, extract_file_id
    DRIVE_ENABLED = True
except ImportError:
    DRIVE_ENABLED = False

st.title("📥 Multi-Modal Data Ingestion")
st.markdown("Securely upload, stream, or connect to your data sources. Supported formats: Tabular, Image Arrays, Audio/Video Streams.")

# --- HELPER FUNCTION: LOGGER ---
def add_log(message):
    if 'system_logs' not in st.session_state:
        st.session_state.system_logs = []
    st.session_state.system_logs.append(message)

# --- UI: DATA SOURCE SELECTION ---
ingest_method = st.radio("Select Ingestion Protocol", 
                        ["Local Storage Upload", "Google Drive / Cloud Bucket", "Raw Dictionary / Text Input"],
                        horizontal=True)

st.markdown("---")

# ==========================================
# PROTOCOL 1: LOCAL UPLOAD (Multi-format)
# ==========================================
if ingest_method == "Local Storage Upload":
    # (Keep your existing Protocol 1 code here - omitted for brevity to focus on Drive)
    st.info("Local upload module active. Switch to Google Drive to test the cloud connection.")

# ==========================================
# PROTOCOL 2: CLOUD CONNECTION (Live Drive API)
# ==========================================
elif ingest_method == "Google Drive / Cloud Bucket":
    st.info("☁️ Google Workspace Native Connector")
    
    if not DRIVE_ENABLED:
        st.error("🚨 `utils/drive_handler.py` not found or missing dependencies (google-api-python-client).")
        st.stop()
        
    drive_url = st.text_input("Paste Google Drive Shareable Link (CSV File)")
    
    if st.button("Establish Connection & Download", type="primary"):
        if not drive_url:
            st.warning("Please provide a valid Drive URL.")
        else:
            file_id = extract_file_id(drive_url)
            
            if not file_id:
                st.error("Could not extract a valid File ID from that link.")
            else:
                with st.spinner("Authenticating via OAuth2 and pulling data stream..."):
                    df = download_csv_from_drive(file_id)
                    
                    if df is not None:
                        # Save to pipeline memory
                        st.session_state.raw_data = df
                        st.session_state.data_type = "tabular"
                        add_log(f"Drive API: Loaded tabular file ID {file_id[:8]}... with shape {df.shape}")
                        
                        st.success(f"Data successfully ingested from Cloud! Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
                        st.subheader("Data Inspector")
                        st.dataframe(df.head(10), use_container_width=True)
                    else:
                        st.error("Failed to retrieve file. Ensure the file is a CSV and you have access permissions.")

# ==========================================
# PROTOCOL 3: MANUAL DICTIONARY / TEXT
# ==========================================
elif ingest_method == "Raw Dictionary / Text Input":
    # (Keep your existing Protocol 3 code here)
    st.info("Manual entry module active.")
