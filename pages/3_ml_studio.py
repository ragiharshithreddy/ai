import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px

# --- ALGORITHM LIBRARIES ---
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Optional heavy boosters
try:
    from xgboost import XGBClassifier
    import lightgbm as lgb
    BOOSTERS_ENABLED = True
except ImportError:
    BOOSTERS_ENABLED = False

st.set_page_config(page_title="ML Studio", layout="wide", page_icon="🧠")
st.title("🧠 The Machine Learning Foundry")
st.markdown("Select your engine, tune the hyperparameters, and compile the model.")

# --- SYSTEM MEMORY CHECK ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.error("🚨 Memory Empty: Please complete Data Ingestion and Preprocessing first.")
    st.stop()

if st.session_state.data_type != "tabular":
    st.warning("🚨 The Classic ML Studio requires Tabular data. For Images/Audio, proceed to the DL & Vision module.")
    st.stop()

df = st.session_state.processed_data
st.success("Tabular memory loaded and ready for training.")

# --- UI ARCHITECTURE ---
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: PIPELINE CONFIG ---
with col1:
    st.header("1. Pipeline Configuration")
    target_col = st.selectbox("Select Target Variable (Y)", df.columns.tolist(), index=len(df.columns)-1)
    
    st.markdown("#### Train/Test Split")
    test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
    random_state = st.number_input("Random Seed (for reproducibility)", value=42)

    st.header("2. Engine Selection")
    model_category = st.selectbox("Algorithm Category", [
        "Tree-Based Ensembles", 
        "Gradient Boosters", 
        "Linear Models", 
        "Support Vector Machines", 
        "Distance & Bayesian"
    ])

# --- COLUMN 2: DYNAMIC HYPERPARAMETERS ---
params = {}
model = None
algo_name = ""

with col2:
    st.header("3. Hyperparameter Engineering")
    st.markdown("Fine-tune the mathematical behavior of the selected engine.")
    
    # --- CATEGORY: TREE ENSEMBLES ---
    if model_category == "Tree-Based Ensembles":
        algo_name = st.selectbox("Specific Engine", ["Random Forest", "Extra Trees", "Decision Tree"])
        
        params['criterion'] = st.selectbox("Split Criterion", ["gini", "entropy"])
        params['max_depth'] = st.slider("Maximum Depth", 1, 100, 10)
        params['min_samples_split'] = st.slider("Min Samples to Split Node", 2, 20, 2)
        
        if algo_name in ["Random Forest", "Extra Trees"]:
            params['n_estimators'] = st.slider("Number of Trees", 10, 1000, 100, step=10)
            
            if algo_name == "Random Forest":
                model = RandomForestClassifier(random_state=random_state, **params)
            else:
                model = ExtraTreesClassifier(random_state=random_state, **params)
        else:
            model = DecisionTreeClassifier(random_state=random_state, **params)

    # --- CATEGORY: GRADIENT BOOSTERS ---
    elif model_category == "Gradient Boosters":
        if not BOOSTERS_ENABLED:
            st.error("XGBoost/LightGBM not installed. Run `pip install xgboost lightgbm`")
            st.stop()
            
        algo_name = st.selectbox("Specific Engine", ["XGBoost", "LightGBM", "AdaBoost", "SKLearn GBM"])
        
        params['learning_rate'] = st.number_input("Learning Rate (eta)", 0.001, 1.0, 0.1, format="%.3f")
        params['n_estimators'] = st.slider("Number of Boosting Stages", 50, 1000, 100, step=50)
        
        if algo_name == "XGBoost":
            params['max_depth'] = st.slider("Max Depth", 3, 20, 6)
            params['subsample'] = st.slider("Subsample Ratio", 0.5, 1.0, 0.8)
            model = XGBClassifier(random_state=random_state, **params)
            
        elif algo_name == "LightGBM":
            params['num_leaves'] = st.slider("Number of Leaves", 10, 200, 31)
            model = lgb.LGBMClassifier(random_state=random_state, **params)
            
        elif algo_name == "SKLearn GBM":
            params['max_depth'] = st.slider("Max Depth", 1, 15, 3)
            model = GradientBoostingClassifier(random_state=random_state, **params)
            
        elif algo_name == "AdaBoost":
            model = AdaBoostClassifier(random_state=random_state, learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])

    # --- CATEGORY: LINEAR MODELS ---
    elif model_category == "Linear Models":
        algo_name = st.selectbox("Specific Engine", ["Logistic Regression", "Ridge Classifier", "Stochastic Gradient Descent (SGD)"])
        
        if algo_name == "Logistic Regression":
            params['C'] = st.number_input("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
            params['max_iter'] = st.slider("Max Iterations", 100, 2000, 500)
            model = LogisticRegression(random_state=random_state, **params)
            
        elif algo_name == "Ridge Classifier":
            params['alpha'] = st.number_input("Regularization Strength (Alpha)", 0.1, 10.0, 1.0)
            model = RidgeClassifier(random_state=random_state, **params)
            
        elif algo_name == "Stochastic Gradient Descent (SGD)":
            params['loss'] = st.selectbox("Loss Function", ["hinge", "log_loss", "modified_huber"])
            params['penalty'] = st.selectbox("Penalty", ["l2", "l1", "elasticnet"])
            model = SGDClassifier(random_state=random_state, **params)

    # --- CATEGORY: SVM ---
    elif model_category == "Support Vector Machines":
        st.warning("⚠️ SVMs scale poorly to datasets with >100,000 rows.")
        algo_name = "Support Vector Classifier (SVC)"
        params['C'] = st.number_input("Regularization Parameter (C)", 0.1, 100.0, 1.0)
        params['kernel'] = st.selectbox("Kernel Type", ["rbf", "linear", "poly", "sigmoid"])
        params['probability'] = True # Required for ROC curves later
        
        if params['kernel'] == 'poly':
            params['degree'] = st.slider("Polynomial Degree", 2, 10, 3)
            
        model = SVC(random_state=random_state, **params)

    # --- CATEGORY: DISTANCE & BAYESIAN ---
    elif model_category == "Distance & Bayesian":
        algo_name = st.selectbox("Specific Engine", ["K-Nearest Neighbors", "Gaussian Naive Bayes"])
        
        if algo_name == "K-Nearest Neighbors":
            params['n_neighbors'] = st.slider("Number of Neighbors (K)", 1, 50, 5)
            params['weights'] = st.selectbox("Weight Function", ["uniform", "distance"])
            model = KNeighborsClassifier(**params)
            
        elif algo_name == "Gaussian Naive Bayes":
            st.info("Gaussian NB has no significant hyperparameters to tune.")
            model = GaussianNB()

# --- EXECUTION & EVALUATION BLOCK ---
st.markdown("---")
if st.button(f"🚀 Compile & Train {algo_name}", use_container_width=True):
    
    # 1. Feature/Target Isolation
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Ensure numerical inputs for ML models
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        st.error("No numeric features found. Did you forget to encode categorical variables in the Preprocessing step?")
        st.stop()
        
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    with st.spinner(f"Initializing matrix calculations for {algo_name}..."):
        try:
            # 3. Model Training
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            st.success("Training Sequence Completed!")
            st.session_state.system_logs.append(f"Trained {algo_name} successfully.")
            
            # 4. Metrics Dashboard
            st.subheader("Performance Metrics")
            
            # Determine averaging method for multi-class vs binary
            avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_test, preds):.4f}")
            m2.metric("Precision", f"{precision_score(y_test, preds, average=avg_method, zero_division=0):.4f}")
            m3.metric("Recall", f"{recall_score(y_test, preds, average=avg_method, zero_division=0):.4f}")
            m4.metric("F1-Score", f"{f1_score(y_test, preds, average=avg_method, zero_division=0):.4f}")
            
            # 5. Visualizations
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                            labels=dict(x="Predicted Label", y="True Label", color="Count"))
            st.plotly_chart(fig, use_container_width=True)
            
            # 6. Save to Global Memory for Export
            st.session_state.trained_model = model
            st.session_state.model_metrics = {
                "name": algo_name,
                "accuracy": accuracy_score(y_test, preds),
                "params": params
            }
            st.info("Model weights and state saved to memory. Proceed to the Export tab.")
            
        except Exception as e:
            st.error(f"Engine Failure: {e}")
            st.info("Check if your target variable is categorical. Regressors are not yet implemented in this view.")
