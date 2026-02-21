import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from PIL import Image

# Import the Google Drive Engine
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
    data_format = st.selectbox("Specify Data Modality", 
                              ["Tabular (CSV, Excel, Parquet)", "JSON (Nested/Flat)", "Computer Vision (Images)", "Audio (WAV, MP3)"])
    
    if data_format == "Tabular (CSV, Excel, Parquet)":
        uploaded_file = st.file_uploader("Drop Tabular File Here", type=["csv", "xlsx", "parquet"])
        
        if uploaded_file:
            try:
                with st.spinner("Parsing tabular data..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(uploaded_file)
                
                st.session_state.raw_data = df
                st.session_state.data_type = "tabular"
                add_log(f"Loaded tabular file: {uploaded_file.name} with shape {df.shape}")
                
                st.success(f"Data successfully ingested! Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
                st.subheader("Data Inspector")
                st.dataframe(df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    elif data_format == "JSON (Nested/Flat)":
        uploaded_file = st.file_uploader("Drop JSON File Here", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                df = pd.json_normalize(data)
                st.session_state.raw_data = df
                st.session_state.data_type = "tabular"
                add_log("Loaded and flattened JSON data.")
                
                st.success("JSON Flattened successfully.")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"JSON Parsing Error: {e}")

    elif data_format == "Computer Vision (Images)":
        uploaded_files = st.file_uploader("Drop Image Batch Here", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            images = []
            for file in uploaded_files:
                img = Image.open(file)
                images.append({"name": file.name, "image": img, "mode": img.mode, "size": img.size})
            
            st.session_state.raw_data = images
            st.session_state.data_type = "vision"
            add_log(f"Loaded {len(images)} images into memory.")
            
            st.success(f"Ingested {len(images)} images.")
            cols = st.columns(min(len(images), 4))
            for idx, col in enumerate(cols):
                col.image(images[idx]['image'], caption=images[idx]['name'], use_container_width=True)

    elif data_format == "Audio (WAV, MP3)":
        uploaded_audio = st.file_uploader("Drop Audio File", type=["wav", "mp3"])
        if uploaded_audio:
            st.session_state.raw_data = uploaded_audio.getvalue()
            st.session_state.data_type = "audio"
            add_log(f"Loaded Audio file: {uploaded_audio.name}")
            st.audio(uploaded_audio)
            st.success("Audio buffered into memory.")

# ==========================================
# PROTOCOL 2: CLOUD CONNECTION (Live Drive API)
# ==========================================
elif ingest_method == "Google Drive / Cloud Bucket":
    st.info("☁️ Google Workspace Native Connector")
    
    if not DRIVE_ENABLED:
        st.error("🚨 `utils/drive_handler.py` not found. Please ensure the `utils` folder exists with an `__init__.py` file inside it.")
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
    st.subheader("Manual Data Injection")
    raw_text = st.text_area("Paste Python Dictionary or JSON string here", height=200)
    if st.button("Inject into Pipeline"):
        try:
            import ast
            try:
                parsed_data = json.loads(raw_text)
            except:
                parsed_data = ast.literal_eval(raw_text)
            
            df = pd.DataFrame(parsed_data)
            st.session_state.raw_data = df
            st.session_state.data_type = "tabular"
            add_log("Manual dictionary injected and converted to tabular.")
            
            st.success("Data injected successfully.")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Syntax Error: Could not parse input. {e}")
