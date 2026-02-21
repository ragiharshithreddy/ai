import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from PIL import Image

st.title("📥 Multi-Modal Data Ingestion")
st.markdown("Securely upload, stream, or connect to your data sources. Supported formats: Tabular, Image Arrays, Audio/Video Streams.")

# --- HELPER FUNCTION: LOGGER ---
def add_log(message):
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
                    # Dynamic Parsing based on extension
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(uploaded_file)
                
                # Save to memory
                st.session_state.raw_data = df
                st.session_state.data_type = "tabular"
                add_log(f"Loaded tabular file: {uploaded_file.name} with shape {df.shape}")
                
                st.success(f"Data successfully ingested! Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
                
                # Data Preview Panel
                st.subheader("Data Inspector")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Quick automated stats
                with st.expander("Show Automated Profiling"):
                    st.write("Column Types:", df.dtypes.astype(str).to_dict())
                    st.write("Missing Values:", df.isna().sum().to_dict())
                    
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    elif data_format == "JSON (Nested/Flat)":
        uploaded_file = st.file_uploader("Drop JSON File Here", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                # Attempt to flatten nested JSON automatically into a DataFrame
                df = pd.json_normalize(data)
                st.session_state.raw_data = df
                st.session_state.data_type = "tabular" # Treat as tabular moving forward
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
            st.write("Preview:")
            cols = st.columns(min(len(images), 4))
            for idx, col in enumerate(cols):
                col.image(images[idx]['image'], caption=images[idx]['name'], use_container_width=True)

    elif data_format == "Audio (WAV, MP3)":
        uploaded_audio = st.file_uploader("Drop Audio File", type=["wav", "mp3"])
        if uploaded_audio:
            st.session_state.raw_data = uploaded_audio.getvalue() # Store raw bytes
            st.session_state.data_type = "audio"
            add_log(f"Loaded Audio file: {uploaded_audio.name}")
            st.audio(uploaded_audio)
            st.success("Audio buffered into memory.")

# ==========================================
# PROTOCOL 2: CLOUD CONNECTION
# ==========================================
elif ingest_method == "Google Drive / Cloud Bucket":
    st.info("Cloud Connector Architecture")
    col1, col2 = st.columns(2)
    with col1:
        bucket_url = st.text_input("G-Drive Folder Link or S3 Bucket URL")
    with col2:
        api_key = st.text_input("Authentication Token / API Key", type="password")
        
    if st.button("Establish Connection"):
        if not bucket_url or not api_key:
            st.error("Authentication rejected. Please provide valid credentials.")
        else:
            with st.spinner("Authenticating via OAuth2..."):
                # MOCK LOGIC for Cloud Connection
                import time; time.sleep(2)
                st.success("Secure connection established to Cloud Directory.")
                add_log("Cloud connection authorized.")
                st.warning("Note: In a live environment, this requires configuring Google Cloud Console OAuth redirect URIs to `localhost:8501`.")

# ==========================================
# PROTOCOL 3: MANUAL DICTIONARY / TEXT
# ==========================================
elif ingest_method == "Raw Dictionary / Text Input":
    st.subheader("Manual Data Injection")
    raw_text = st.text_area("Paste Python Dictionary or JSON string here", height=200)
    if st.button("Inject into Pipeline"):
        try:
            # Safely evaluate dictionary or json
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
