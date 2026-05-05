# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-friendly styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f0f7f0 0%, #e5f0e5 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e5a1e 0%, #2d7a2d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(-20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #c8e0c8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1e5a1e;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e5a1e 0%, #2d7a2d 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Message styling */
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        animation: fadeIn 0.5s;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        animation: fadeIn 0.5s;
    }
    
    .info-box {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e5a1e 0%, #2d7a2d 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Header
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>Advanced AI-Powered Early Detection & Risk Assessment Platform</p>
    <p style="font-size: 0.9rem; margin-top: 10px;">🏥 Clinical Decision Support System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📂 Data Management")
    
    # Data import section
    uploaded_file = st.file_uploader(
        "Upload CSV Dataset", 
        type=['csv'], 
        help="Upload your heart disease dataset in CSV format"
    )
    
    use_sample = st.checkbox("Use Sample Dataset", value=True)
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.success(f"✅ {len(data)} records loaded successfully!")
            use_sample = False
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            data = None
            st.session_state.data_loaded = False
    elif use_sample:
        # Create comprehensive sample data
        sample_data = """Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
40,M,ATA,140,289,0,Normal,172,N,0,Up,0
49,F,NAP,160,180,0,Normal,156,N,1,Flat,1
37,M,ATA,130,283,0,ST,98,N,0,Up,0
48,F,ASY,138,214,0,Normal,108,Y,1.5,Flat,1
54,M,NAP,150,195,0,Normal,122,N,0,Up,0
39,M,NAP,120,339,0,Normal,170,N,0,Up,0
45,F,ATA,130,237,0,Normal,170,N,0,Up,0
54,M,ATA,110,208,0,Normal,142,N,0,Up,0
37,M,ASY,140,207,0,Normal,130,Y,1.5,Flat,1
48,F,ATA,120,284,0,Normal,120,N,0,Up,0
37,F,NAP,130,211,0,Normal,142,N,0,Up,0
58,M,ATA,136,164,0,ST,99,Y,2,Flat,1
39,M,ATA,120,204,0,Normal,145,N,0,Up,0
49,M,ASY,140,234,0,Normal,140,Y,1,Flat,1
42,F,NAP,115,211,0,ST,137,N,0,Up,0"""
        
        from io import StringIO
        data = pd.read_csv(StringIO(sample_data))
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.info(f"📊 Using sample dataset with {len(data)} records")
    else:
        st.warning("⚠️ Please upload data or use sample dataset")
        st.stop()
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("### 🤖 Model Configuration")
        
        # Model selection
        models_to_train = st.multiselect(
            "Select ML Models for Training",
            options=['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 
                     'Random Forest', 'Gradient Boosting', 'SVM'],
            default=['Logistic Regression', 'Random Forest'],
            help="Select one or more models to train and compare"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Training Parameters")
        
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05, help="Proportion of data for testing")
        random_state = st.number_input("Random Seed", value=42, step=1, help="For reproducible results")
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, help="Number of CV folds")
        
        st.markdown("---")
        st.markdown("### 📈 Dataset Statistics")
        
        if 'data' in st.session_state:
            df = st.session_state.data
            st.metric("Total Records", len(df))
            st.metric("Features", len(df.columns) - 1)
            if 'HeartDisease' in df.columns:
                disease_pct = (df['HeartDisease'].sum() / len(df)) * 100
                st.metric("Disease Prevalence", f"{disease_pct:.1f}%")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Clinical Features:**
        - Age & Gender
        - Chest Pain Type
        - Resting BP & Cholesterol
        - Fasting Blood Sugar
        - Resting ECG
        - Max Heart Rate
        - Exercise Angina
        - ST Depression
        - ST Slope
        
        **Target:** Heart Disease (0=No, 1=Yes)
        """)

# Main content area
if st.session_state.data_loaded:
    data = st.session_state.data
    
    # Data Overview Section
    st.markdown("## 📊 Clinical Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(data)}</div>
            <div class="metric-label">Total Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(data.columns) - 1}</div>
            <div class="metric-label">Clinical Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'HeartDisease' in data.columns:
            disease_count = data['HeartDisease'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{disease_count}</div>
                <div class="metric-label">Disease Cases</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'HeartDisease' in data.columns:
            healthy_count = len(data) - data['HeartDisease'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{healthy_count}</div>
                <div class="metric-label">Healthy Cases</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Tabs for different sections
    tab_overview, tab_eda, tab_training, tab_prediction = st.tabs([
        "📋 Dataset Preview", "📈 Exploratory Analysis", "🤖 Model Training", "🔮 Predictions"
    ])
    
    # Tab 1: Dataset Preview
    with tab_overview:
        st.markdown("### Patient Records Preview")
        st.dataframe(data.head(20), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Numerical Features Statistics")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            st.dataframe(data[numeric_cols].describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### 🏷️ Categorical Features Distribution")
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                st.write(f"**{col}:**")
                st.write(data[col].value_counts())
                st.write("---")
    
    # Tab 2: Exploratory Data Analysis
    with tab_eda:
        st.markdown("### Exploratory Data Analysis")
        
        # Visualization type selector
        viz_type = st.radio(
            "Select Visualization Type",
            ["Distribution Analysis", "Correlation Analysis", "Comparative Analysis"],
            horizontal=True
        )
        
        if viz_type == "Distribution Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    feature = st.selectbox("Select Feature", numeric_cols)
                    fig = px.histogram(
                        data, x=feature, 
                        color='HeartDisease' if 'HeartDisease' in data.columns else None,
                        title=f"Distribution of {feature}",
                        color_discrete_sequence=['#2d7a2d', '#ff6b6b'],
                        marginal='box'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'HeartDisease' in data.columns and numeric_cols:
                    fig = px.box(
                        data, x='HeartDisease', y=feature,
                        title=f"{feature} by Heart Disease Status",
                        color='HeartDisease',
                        color_discrete_sequence=['#2d7a2d', '#ff6b6b']
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Analysis":
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                corr_matrix = numeric_data.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title="Feature Correlation Matrix",
                    width=800,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance based on correlation with target
                if 'HeartDisease' in corr_matrix.columns:
                    st.markdown("#### Top Correlations with Heart Disease")
                    correlations = corr_matrix['HeartDisease'].drop('HeartDisease').sort_values(ascending=False)
                    fig = px.bar(
                        x=correlations.values, 
                        y=correlations.index,
                        orientation='h',
                        title="Correlation with Heart Disease",
                        color=correlations.values,
                        color_continuous_scale='RdYlGn',
                        labels={'x': 'Correlation', 'y': 'Feature'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:  # Comparative Analysis
            if 'HeartDisease' in data.columns:
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    cat_col = st.selectbox("Select Categorical Feature", categorical_cols)
                    cross_tab = pd.crosstab(data[cat_col], data['HeartDisease'], normalize='index') * 100
                    
                    fig = px.bar(
                        cross_tab,
                        title=f"Heart Disease Rate by {cat_col}",
                        labels={'value': 'Percentage (%)', 'variable': 'Heart Disease'},
                        color_discrete_sequence=['#2d7a2d', '#ff6b6b'],
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Model Training
    with tab_training:
        st.markdown("### Machine Learning Model Training")
        
        if len(models_to_train) == 0:
            st.warning("Please select at least one model from the sidebar to train.")
        else:
            if st.button("🚀 Start Training", use_container_width=True):
                # Preprocessing function
                def preprocess_data(df):
                    df_processed = df.copy()
                    
                    # Encode categorical variables
                    label_encoders = {}
                    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
                    
                    for col in categorical_cols:
                        if col in df_processed.columns:
                            le = LabelEncoder()
                            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                            label_encoders[col] = le
                    
                    # Handle missing values
                    df_processed = df_processed.fillna(df_processed.median())
                    
                    # Separate features and target
                    X = df_processed.drop('HeartDisease', axis=1)
                    y = df_processed['HeartDisease']
                    
                    return X, y, label_encoders
                
                with st.spinner("Processing data and training models..."):
                    # Preprocess
                    X, y, encoders = preprocess_data(data)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Define models
                    models = {
                        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
                        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                        'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=5),
                        'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100, max_depth=10),
                        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state, n_estimators=100),
                        'SVM': SVC(random_state=random_state, probability=True, kernel='rbf')
                    }
                    
                    # Train selected models
                    results = {}
                    trained_models = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, model_name in enumerate(models_to_train):
                        if model_name in models:
                            status_text.text(f"Training {model_name}...")
                            model = models[model_name]
                            
                            try:
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                                
                                # Calculate metrics
                                accuracy = accuracy_score(y_test, y_pred)
                                precision = precision_score(y_test, y_pred)
                                recall = recall_score(y_test, y_pred)
                                f1 = f1_score(y_test, y_pred)
                                
                                # Cross-validation
                                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                                
                                # ROC-AUC
                                if hasattr(model, 'predict_proba'):
                                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                                else:
                                    roc_auc = None
                                
                                results[model_name] = {
                                    'accuracy': accuracy,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1_score': f1,
                                    'cv_mean': cv_scores.mean(),
                                    'cv_std': cv_scores.std(),
                                    'roc_auc': roc_auc,
                                    'model': model,
                                    'predictions': y_pred
                                }
                                trained_models[model_name] = model
                                
                            except Exception as e:
                                st.error(f"Error training {model_name}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(models_to_train))
                    
                    status_text.text("Training complete!")
                    progress_bar.empty()
                    
                    if results:
                        # Store in session state
                        st.session_state.results = results
                        st.session_state.trained_models = trained_models
                        st.session_state.scaler = scaler
                        st.session_state.encoders = encoders
                        st.session_state.X_columns = X.columns.tolist()
                        st.session_state.models_trained = True
                        
                        # Display results
                        st.markdown("### 📊 Model Performance Comparison")
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'Model': list(results.keys()),
                            'Accuracy': [results[m]['accuracy'] for m in results],
                            'Precision': [results[m]['precision'] for m in results],
                            'Recall': [results[m]['recall'] for m in results],
                            'F1-Score': [results[m]['f1_score'] for m in results],
                            'CV Score': [results[m]['cv_mean'] for m in results],
                            'ROC-AUC': [results[m]['roc_auc'] if results[m]['roc_auc'] else 0 for m in results]
                        })
                        
                        # Sort by accuracy
                        results_df = results_df.sort_values('Accuracy', ascending=False)
                        
                        # Display metrics table with styling
                        st.dataframe(
                            results_df.style.format({
                                'Accuracy': '{:.3f}',
                                'Precision': '{:.3f}',
                                'Recall': '{:.3f}',
                                'F1-Score': '{:.3f}',
                                'CV Score': '{:.3f}',
                                'ROC-AUC': '{:.3f}'
                            }).background_gradient(subset=['Accuracy', 'F1-Score'], cmap='RdYlGn'),
                            use_container_width=True
                        )
                        
                        # Performance visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.bar(
                                results_df, x='Model', y='Accuracy',
                                title='Model Accuracy Comparison',
                                color='Accuracy',
                                color_continuous_scale='RdYlGn',
                                text='Accuracy'
                            )
                            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                            fig.update_layout(xaxis_tickangle=-45, height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.bar(
                                results_df, x='Model', y='F1-Score',
                                title='Model F1-Score Comparison',
                                color='F1-Score',
                                color_continuous_scale='RdYlGn',
                                text='F1-Score'
                            )
                            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                            fig.update_layout(xaxis_tickangle=-45, height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Best model highlight
                        best_model = results_df.iloc[0]['Model']
                        best_accuracy = results_df.iloc[0]['Accuracy']
                        
                        st.markdown(f"""
                        <div class="success-message">
                            <strong>🏆 Best Performing Model: {best_model}</strong><br>
                            Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)<br>
                            This model is recommended for predictions.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confusion Matrix for best model
                        st.markdown("### Confusion Matrix - Best Model")
                        best_model_obj = results[best_model]['model']
                        best_predictions = results[best_model]['predictions']
                        
                        cm = confusion_matrix(y_test, best_predictions)
                        fig = px.imshow(
                            cm,
                            text_auto=True,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=['No Disease', 'Disease'],
                            y=['No Disease', 'Disease'],
                            title=f"Confusion Matrix - {best_model}",
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Classification report
                        st.markdown("### Detailed Classification Report")
                        report = classification_report(y_test, best_predictions, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format('{:.3f}'), use_container_width=True)
                        
                        # Feature Importance for tree-based models
                        if best_model in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
                            st.markdown("### Feature Importance Analysis")
                            
                            if hasattr(best_model_obj, 'feature_importances_'):
                                importances = best_model_obj.feature_importances_
                                feature_names = X.columns
                                
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=True)
                                
                                fig = px.bar(
                                    importance_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title=f"Top Features - {best_model}",
                                    color='Importance',
                                    color_continuous_scale='RdYlGn'
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Predictions
    with tab_prediction:
        st.markdown("### 🔮 Patient Risk Assessment")
        
        if not st.session_state.get('models_trained', False):
            st.markdown("""
            <div class="info-box">
                ⚠️ No trained models found. Please go to the "Model Training" tab and train models first.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("#### Enter Patient Clinical Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
                sex = st.selectbox("Gender", ['M', 'F'])
                cp = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
            
            with col2:
                resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
                cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
                fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
            with col3:
                resting_ecg = st.selectbox("Resting ECG Results", ['Normal', 'ST', 'LVH'])
                max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
                exercise_angina = st.selectbox("Exercise Induced Angina", ['N', 'Y'], format_func=lambda x: "Yes" if x=='Y' else "No")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-3.0, max_value=6.0, value=0.0, step=0.1)
            
            with col2:
                st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])
            
            with col3:
                # Model selection for prediction
                available_models = list(st.session_state.trained_models.keys())
                if available_models:
                    prediction_model = st.selectbox("Select Model for Prediction", available_models)
            
            if st.button("🩺 Assess Heart Disease Risk", use_container_width=True):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Sex': [sex],
                    'ChestPainType': [cp],
                    'RestingBP': [resting_bp],
                    'Cholesterol': [cholesterol],
                    'FastingBS': [fasting_bs],
                    'RestingECG': [resting_ecg],
                    'MaxHR': [max_hr],
                    'ExerciseAngina': [exercise_angina],
                    'Oldpeak': [oldpeak],
                    'ST_Slope': [st_slope]
                })
                
                # Encode categorical variables
                for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
                    if col in st.session_state.encoders:
                        encoder = st.session_state.encoders[col]
                        try:
                            input_data[col] = encoder.transform(input_data[col])
                        except:
                            input_data[col] = 0
                
                # Ensure all features are present
                for col in st.session_state.X_columns:
                    if col not in input_data.columns:
                        input_data[col] = 0
                
                input_data = input_data[st.session_state.X_columns]
                
                # Scale features
                input_scaled = st.session_state.scaler.transform(input_data)
                
                # Make prediction
                model = st.session_state.trained_models[prediction_model]
                prediction = model.predict(input_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_scaled)[0][1]
                else:
                    probability = None
                
                # Display results
                st.markdown("---")
                st.markdown("### 📋 Assessment Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="warning-message">
                            <strong>⚠️ HIGH RISK DETECTED</strong><br>
                            The AI model predicts an elevated risk of heart disease.<br><br>
                            <strong>Recommended Actions:</strong><br>
                            • 🏥 Schedule immediate consultation with a cardiologist<br>
                            • 📊 Complete cardiac workup (ECG, Echo, Stress test)<br>
                            • 💊 Consider preventive medications if indicated<br>
                            • 🍎 Start heart-healthy diet immediately<br>
                            • 🏃 Begin supervised exercise program
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-message">
                            <strong>✅ LOW RISK DETECTED</strong><br>
                            The AI model predicts low risk of heart disease.<br><br>
                            <strong>Preventive Recommendations:</strong><br>
                            • 🍏 Maintain healthy diet and lifestyle<br>
                            • 🏃 Regular physical activity (150 minutes/week)<br>
                            • 📅 Annual health check-ups<br>
                            • ❤️ Monitor blood pressure and cholesterol<br>
                            • 🚭 Avoid smoking and limit alcohol
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if probability is not None:
                        risk_level = probability * 100
                        
                        # Risk meter
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_level,
                            title={'text': "Risk Score", 'font': {'size': 24}},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "#dc3545" if risk_level > 50 else "#28a745"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 30], 'color': '#d4edda'},
                                    {'range': [30, 70], 'color': '#fff3cd'},
                                    {'range': [70, 100], 'color': '#f8d7da'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': risk_level
                                }
                            }
                        ))
                        fig.update_layout(height=350, margin=dict(t=50, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 10px;">
                            <span style="font-size: 1.2rem; font-weight: bold;">
                                Probability of Heart Disease: {risk_level:.1f}%
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed risk factors
                st.markdown("### 📊 Key Risk Factors Analysis")
                
                risk_factors = []
                if age > 55:
                    risk_factors.append(("Age > 55", "Increased risk with age", "high"))
                if resting_bp > 140:
                    risk_factors.append(("High Blood Pressure", f"{resting_bp} mm Hg > 140", "high"))
                elif resting_bp > 120:
                    risk_factors.append(("Elevated BP", f"{resting_bp} mm Hg", "moderate"))
                if cholesterol > 240:
                    risk_factors.append(("High Cholesterol", f"{cholesterol} mg/dl > 240", "high"))
                elif cholesterol > 200:
                    risk_factors.append(("Borderline Cholesterol", f"{cholesterol} mg/dl", "moderate"))
                if max_hr < 100:
                    risk_factors.append(("Low Max HR", f"{max_hr} bpm < 100", "high"))
                if oldpeak > 1.5:
                    risk_factors.append(("Significant ST Depression", f"Oldpeak: {oldpeak}", "high"))
                
                if risk_factors:
                    for risk_factor, details, severity in risk_factors:
                        color = "#dc3545" if severity == "high" else "#ffc107"
                        st.markdown(f"""
                        <div style="border-left: 3px solid {color}; padding: 10px; margin: 5px 0; background: #f8f9fa;">
                            <strong>{risk_factor}</strong><br>
                            {details}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("✅ No major risk factors identified based on the entered data.")
                
                # Disclaimer
                st.markdown("---")
                st.markdown("""
                <div style="background: #e7f3ff; padding: 1rem; border-radius: 8px; font-size: 0.9rem;">
                    <strong>⚠️ Medical Disclaimer:</strong> This is an AI-powered screening tool for educational purposes only. 
                    It does not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified 
                    healthcare provider for medical decisions.
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>❤️ Heart Disease Prediction System | Powered by Machine Learning</p>
    <p style="font-size: 0.8rem;">© 2024 - Clinical Decision Support Tool</p>
</div>
""", unsafe_allow_html=True)
