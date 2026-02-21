import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Evaluation & Export", layout="wide", page_icon="📦")
st.title("📦 Evaluation & Deployment Hub")
st.markdown("Analyze model diagnostics and export your deployment artifacts for offline use.")

# --- SYSTEM MEMORY CHECK ---
if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
    st.error("🚨 No Trained Model Found in Memory! Please go to the ML Studio or DL Foundry to train an engine first.")
    st.stop()

model = st.session_state.trained_model
data_type = st.session_state.data_type
processed_data = st.session_state.processed_data

# Determine Model Type
is_keras_model = hasattr(model, 'save') and 'keras' in str(type(model)).lower()
model_family = "Deep Learning Tensor Network (Keras/TF)" if is_keras_model else "Classic ML Algorithm (SKLearn/XGBoost)"

st.success(f"System memory locked onto: **{model_family}**")

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Visuals", "💾 Artifact Export", "📜 System Logs"])

# ==========================================
# TAB 1: DIAGNOSTIC VISUALS
# ==========================================
with tab1:
    st.header("Model Diagnostics")
    
    if data_type == "tabular" and not is_keras_model:
        df = processed_data if isinstance(processed_data, pd.DataFrame) else processed_data["data"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Global Feature Importance")
                importances = model.feature_importances_
                # Assuming the last column was the target during training
                feature_names = df.columns[:-1] 
                
                # Ensure dimensions match before plotting
                if len(importances) == len(feature_names):
                    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(15) # Top 15
                    
                    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                                 title="Top Predictive Features", color='Importance', color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Feature dimension mismatch. Unable to plot importance.")
            else:
                st.info(f"Feature importance is not supported natively by this specific algorithm ({type(model).__name__}).")

        with col2:
            st.subheader("Decision Boundary / ROC (Placeholder)")
            st.info("In a live production environment, this module renders advanced interactive ROC/AUC curves and precision-recall trade-off graphs based on the test set predictions.")
            
    elif is_keras_model:
        st.subheader("Neural Network Architecture Diagnostics")
        st.info("Deep Learning models are treated as black boxes. To evaluate, refer to the loss and accuracy metrics generated during the compilation phase in the DL Foundry.")

# ==========================================
# TAB 2: ARTIFACT EXPORT
# ==========================================
with tab2:
    st.header("Deployment Artifacts")
    st.markdown("Download your trained model weights to run inference locally without this UI.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Export Engine")
        
        if is_keras_model:
            st.info("Keras models are exported as Hierarchical Data Format (.h5) files.")
            # Keras models need to be saved to a file first, then read as bytes
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                model.save(tmp.name)
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
            os.remove(tmp.name) # Clean up
            
            st.download_button(
                label="📥 Download Weights (.h5)",
                data=model_bytes,
                file_name="nexus_dl_model.h5",
                mime="application/octet-stream",
                use_container_width=True
            )
            
        else:
            st.info("Classic ML models are serialized as Pickle (.pkl) objects.")
            model_bytes = pickle.dumps(model)
            
            st.download_button(
                label="📥 Download Serialized Model (.pkl)",
                data=model_bytes,
                file_name="nexus_ml_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
            
    with col2:
        st.markdown("### Deployment Instructions")
        if is_keras_model:
            st.code('''
# To load this model in your own Python script:
import tensorflow as tf
model = tf.keras.models.load_model('nexus_dl_model.h5')

# Run inference
predictions = model.predict(new_data_array)
            ''', language='python')
        else:
            st.code('''
# To load this model in your own Python script:
import pickle
with open('nexus_ml_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Run inference
predictions = model.predict(new_data_dataframe)
            ''', language='python')

# ==========================================
# TAB 3: SYSTEM LOGS
# ==========================================
with tab3:
    st.header("Pipeline Telemetry")
    if 'system_logs' in st.session_state:
        for log in st.session_state.system_logs:
            st.code(log, language='bash')
    else:
        st.write("No telemetry recorded.")
