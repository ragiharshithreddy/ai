import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# --- HEAVY LIBRARY MANAGERS ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
    from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
    TF_ENABLED = True
except ImportError:
    TF_ENABLED = False

try:
    from transformers import pipeline
    HF_ENABLED = True
except ImportError:
    HF_ENABLED = False

st.set_page_config(page_title="DL & Transformers", layout="wide", page_icon="🔥")
st.title("🔥 Deep Learning & Transformer Foundry")
st.markdown("Design custom neural architectures, leverage transfer learning, or deploy Hugging Face Transformers.")

# --- SYSTEM MEMORY CHECK ---
if 'processed_data' not in st.session_state or 'data_type' not in st.session_state:
    st.warning("🚨 Memory Empty: Please complete Data Ingestion first.")
    st.stop()

data_type = st.session_state.data_type
raw_data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.raw_data

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🧠 Tabular Deep Learning", "👁️ Computer Vision", "📝 NLP & Transformers"])

# ==========================================
# TAB 1: TABULAR DEEP LEARNING (Dense NNs)
# ==========================================
with tab1:
    st.header("Custom Neural Network Architect")
    if data_type != "tabular":
        st.info("Tabular data required for this module. Please ingest a CSV/Excel file.")
    elif not TF_ENABLED:
        st.error("TensorFlow not installed. Run `pip install tensorflow`.")
    else:
        df = raw_data
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Global Architecture")
            target_col = st.selectbox("Target Variable (Y)", df.columns.tolist(), key="dl_target")
            num_layers = st.slider("Number of Hidden Layers", 1, 10, 3)
            
            st.markdown("#### Optimization Phase")
            optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "nadam"])
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            epochs = st.slider("Training Epochs", 1, 500, 50)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128, 256], index=2)

        with col2:
            st.markdown("#### Layer Configuration")
            layers_config = []
            for i in range(num_layers):
                c1, c2, c3 = st.columns(3)
                neurons = c1.number_input(f"L{i+1} Neurons", 4, 1024, 64, key=f"n_{i}")
                activation = c2.selectbox(f"L{i+1} Activation", ["relu", "selu", "tanh", "sigmoid"], key=f"a_{i}")
                dropout = c3.slider(f"L{i+1} Dropout Rate", 0.0, 0.8, 0.2, key=f"d_{i}")
                layers_config.append((neurons, activation, dropout))
                
            if st.button("🔨 Compile Neural Network", use_container_width=True):
                # Calculate input shape based on features
                input_dim = len(df.columns) - 1
                
                # Build the Sequential Model
                model = Sequential()
                model.add(Input(shape=(input_dim,)))
                
                for units, act, drop in layers_config:
                    model.add(Dense(units, activation=act))
                    if drop > 0.0:
                        model.add(Dropout(drop))
                        
                # Output Layer Setup (Inferring Binary vs Multi-class vs Regression)
                unique_classes = df[target_col].nunique()
                if unique_classes == 2:
                    model.add(Dense(1, activation='sigmoid'))
                    loss_fn = 'binary_crossentropy'
                elif unique_classes < 20: # Arbitrary threshold for classification
                    model.add(Dense(unique_classes, activation='softmax'))
                    loss_fn = 'sparse_categorical_crossentropy'
                else: # Regression
                    model.add(Dense(1, activation='linear'))
                    loss_fn = 'mse'
                
                # Optimizer configuration
                if optimizer == "adam": opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                elif optimizer == "sgd": opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                elif optimizer == "rmsprop": opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                else: opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

                model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
                st.success(f"Network Compiled Successfully! Loss Function: {loss_fn}")
                
                # Save string representation of summary
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                st.code("\n".join(stringlist))
                st.session_state.trained_model = model

# ==========================================
# TAB 2: COMPUTER VISION (Transfer Learning)
# ==========================================
with tab2:
    st.header("Vision Hub: Transfer Learning")
    if data_type != "vision":
        st.info("Vision data required for this module. Please ingest image files.")
    elif not TF_ENABLED:
        st.error("TensorFlow not installed. Run `pip install tensorflow`.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            backbone_choice = st.selectbox("Select Pre-Trained Backbone", ["MobileNetV2 (Fast)", "ResNet50 (Balanced)", "EfficientNetB0 (Accurate)"])
            freeze_weights = st.checkbox("Freeze Base Weights (Recommended)", value=True)
            num_classes = st.number_input("Number of Output Classes", 2, 1000, 2)
            
        with col2:
            st.markdown("#### Custom Head Configuration")
            dense_neurons = st.slider("Top Layer Neurons", 64, 2048, 512, step=64)
            final_activation = st.selectbox("Output Activation", ["sigmoid (Binary)", "softmax (Multi-class)"])
            
            if st.button("👁️ Build Vision Model", use_container_width=True):
                with st.spinner("Downloading/Loading ImageNet Weights..."):
                    input_tensor = Input(shape=(224, 224, 3)) # Standardizing input for preview
                    
                    if backbone_choice == "MobileNetV2 (Fast)":
                        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
                    elif backbone_choice == "ResNet50 (Balanced)":
                        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
                    else:
                        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_tensor)
                        
                    base_model.trainable = not freeze_weights
                    
                    # Add Custom Head
                    x = base_model.output
                    x = GlobalAveragePooling2D()(x)
                    x = Dense(dense_neurons, activation='relu')(x)
                    x = Dropout(0.5)(x)
                    predictions = Dense(num_classes, activation=final_activation.split()[0])(x)
                    
                    vision_model = Model(inputs=base_model.input, outputs=predictions)
                    vision_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    st.success(f"{backbone_choice} backbone loaded and custom head attached!")
                    st.session_state.trained_model = vision_model

# ==========================================
# TAB 3: NLP & TRANSFORMERS (HuggingFace)
# ==========================================
with tab3:
    st.header("NLP Hub: HuggingFace Pipelines")
    if not HF_ENABLED:
        st.error("Transformers not installed. Run `pip install transformers torch`.")
    else:
        task = st.selectbox("Select NLP Task", ["Text Classification (Sentiment)", "Text Summarization", "Text Generation"])
        
        if task == "Text Classification (Sentiment)":
            model_id = st.selectbox("Model", ["distilbert-base-uncased-finetuned-sst-2-english", "roberta-base"])
            user_text = st.text_area("Test the Model: Enter text here")
            
            if st.button("Run Inference"):
                with st.spinner("Loading Transformer pipeline..."):
                    nlp_pipe = pipeline("sentiment-analysis", model=model_id)
                    result = nlp_pipe(user_text)
                    st.write("Result:", result)
                    
        elif task == "Text Summarization":
            model_id = st.selectbox("Model", ["sshleifer/distilbart-cnn-12-6", "t5-base"])
            user_text = st.text_area("Paste large text to summarize", height=200)
            
            if st.button("Generate Summary"):
                with st.spinner("Abstracting text..."):
                    summarizer = pipeline("summarization", model=model_id)
                    result = summarizer(user_text, max_length=130, min_length=30, do_sample=False)
                    st.success(result[0]['summary_text'])
