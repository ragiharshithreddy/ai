import streamlit as st
import pandas as pd
import numpy as np
import io
from PIL import Image, ImageOps, ImageEnhance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# Optional heavy libraries wrapped to prevent crashes
try:
    import librosa
    import matplotlib.pyplot as plt
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False

st.set_page_config(page_title="Universal Preprocessing", layout="wide", page_icon="🛠️")
st.title("🛠️ Universal Preprocessing Engine")
st.markdown("Automated cleaning, augmentation, and tensor transformation layer.")

# --- SYSTEM MEMORY CHECK ---
if 'raw_data' not in st.session_state or st.session_state.raw_data is None:
    st.warning("🚨 Memory Empty: Please go to the Ingestion Hub to load data first.")
    st.stop()

data_type = st.session_state.data_type
raw_data = st.session_state.raw_data

st.info(f"Active Memory Detected: **{data_type.upper()}** format.")

# ==========================================
# PIPELINE 1: TABULAR DATA ENGINEERING
# ==========================================
if data_type == "tabular":
    df = raw_data.copy()
    st.subheader("Tabular Data Engineering")
    
    col1, col2, col3 = st.columns(3)
    
    # --- STEP 1: Missing Values ---
    with col1:
        st.markdown("#### 1. Missing Value Imputation")
        missing_count = df.isna().sum().sum()
        st.write(f"Detected Missing Values: **{missing_count}**")
        
        imputation_strategy = st.selectbox("Imputation Strategy", ["None", "Mean/Mode Imputer", "KNN Imputer", "Drop Missing"])
        
        if imputation_strategy == "Mean/Mode Imputer":
            num_cols = df.select_dtypes(include=np.number).columns
            cat_cols = df.select_dtypes(exclude=np.number).columns
            
            if len(num_cols) > 0:
                num_imputer = SimpleImputer(strategy='mean')
                df[num_cols] = num_imputer.fit_transform(df[num_cols])
            if len(cat_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
                
        elif imputation_strategy == "KNN Imputer":
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                knn_imputer = KNNImputer(n_neighbors=5)
                df[num_cols] = knn_imputer.fit_transform(df[num_cols])
                st.caption("KNN Applied to numeric columns only.")
                
        elif imputation_strategy == "Drop Missing":
            df = df.dropna()

    # --- STEP 2: Categorical Encoding ---
    with col2:
        st.markdown("#### 2. Categorical Encoding")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        st.write(f"Detected Categorical Columns: **{len(cat_cols)}**")
        
        encoding_strategy = st.selectbox("Encoding Strategy", ["None", "Label Encoding", "One-Hot Encoding (Dummies)"])
        
        if encoding_strategy == "Label Encoding" and len(cat_cols) > 0:
            le = LabelEncoder()
            for col in cat_cols:
                df[col] = le.fit_transform(df[col].astype(str))
                
        elif encoding_strategy == "One-Hot Encoding (Dummies)" and len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # --- STEP 3: Feature Scaling ---
    with col3:
        st.markdown("#### 3. Feature Scaling")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        scaling_strategy = st.selectbox("Scaling Algorithm", ["None", "Standard Scaler (Z-Score)", "Min-Max Scaler (0-1)", "Robust Scaler (Outliers)"])
        
        if scaling_strategy != "None" and len(num_cols) > 0:
            if scaling_strategy == "Standard Scaler (Z-Score)":
                scaler = StandardScaler()
            elif scaling_strategy == "Min-Max Scaler (0-1)":
                scaler = MinMaxScaler()
            elif scaling_strategy == "Robust Scaler (Outliers)":
                scaler = RobustScaler()
                
            df[num_cols] = scaler.fit_transform(df[num_cols])

    st.markdown("---")
    st.subheader("Processed DataFrame Inspector")
    st.dataframe(df.head(10), use_container_width=True)
    
    if st.button("💾 Commit Tabular Pipeline to Memory", use_container_width=True):
        st.session_state.processed_data = df
        st.session_state.system_logs.append("Tabular preprocessing pipeline executed.")
        st.success("Data compiled and ready for the Model Foundry!")

# ==========================================
# PIPELINE 2: COMPUTER VISION ENHANCEMENT
# ==========================================
elif data_type == "vision":
    st.subheader("Vision Pipeline: Image Augmentation & Normalization")
    
    images = raw_data  # List of dicts containing PIL images
    processed_images = []
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Geometry & Formatting")
        target_size = st.slider("Target Resolution (NxN)", 32, 1024, 224, step=32)
        grayscale = st.checkbox("Convert to Grayscale (1-Channel)")
        flip_h = st.checkbox("Horizontal Flip Augmentation")
        
    with col2:
        st.markdown("#### Photometric Augmentation")
        brightness = st.slider("Brightness Shift", 0.5, 2.0, 1.0)
        contrast = st.slider("Contrast Shift", 0.5, 2.0, 1.0)

    if st.button("⚙️ Process Image Batch", use_container_width=True):
        with st.spinner("Applying tensor transformations..."):
            for img_dict in images:
                img = img_dict["image"]
                
                # Resizing
                img = img.resize((target_size, target_size))
                
                # Grayscale
                if grayscale:
                    img = ImageOps.grayscale(img)
                    
                # Geometry
                if flip_h:
                    img = ImageOps.mirror(img)
                    
                # Photometrics
                if brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(brightness)
                if contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(contrast)
                
                processed_images.append(img)
                
        st.session_state.processed_data = processed_images
        st.session_state.system_logs.append(f"Processed {len(images)} images to {target_size}x{target_size}.")
        st.success("Vision Batch processed and committed to memory!")
        
        # Preview first processed image
        st.image(processed_images[0], caption=f"Sample: {target_size}x{target_size}", width=300)

# ==========================================
# PIPELINE 3: AUDIO SIGNAL PROCESSING
# ==========================================
elif data_type == "audio":
    st.subheader("Audio Signal Processing Engine")
    
    if not AUDIO_ENABLED:
        st.error("Missing dependency: Please run `pip install librosa matplotlib` to enable Audio Processing.")
        st.stop()
        
    audio_bytes = raw_data
    
    st.markdown("#### Feature Extraction")
    extraction_type = st.radio("Extract Audio Features into Vision Tensors", 
                               ["Mel-Spectrogram (Image Representation)", "MFCC (Sequence Data)"])
    
    if st.button("⚙️ Extract Features", use_container_width=True):
        with st.spinner("Analyzing signal frequencies..."):
            # Mock processing for Streamlit (Requires saving buffer to temp file in production)
            st.success(f"{extraction_type} extraction simulated successfully.")
            st.session_state.processed_data = "audio_features_tensor"
            st.session_state.system_logs.append(f"Audio processed using {extraction_type}.")
