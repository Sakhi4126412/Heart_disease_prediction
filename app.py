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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    /* Main background color */
    .stApp {
        background-color: #f0f7f0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2c5f2d 0%, #4a8b4a 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #2c5f2d;
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9f8 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #d4e6d4;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c5f2d;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2c5f2d;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #4a8b4a;
        transform: translateY(-2px);
    }
    
    /* Success message */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Warning message */
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>Advanced Machine Learning for Early Detection & Prevention</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Data Source")
    
    # Data import section
    uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'], help="Upload your heart disease dataset")
    
    use_sample = st.checkbox("Use sample data (default dataset)", value=True)
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! ({len(data)} records)")
        use_sample = False
    elif use_sample:
        # Reading from the provided heart.csv content
        from io import StringIO
        data = pd.read_csv(StringIO(st.session_state.get('heart_data', '')))
        st.info(f"📈 Using default dataset ({len(data)} records)")
    else:
        st.warning("⚠️ Please upload data or use sample data")
        st.stop()
    
    st.markdown("---")
    
    st.markdown("### 🤖 Model Selection")
    
    # Model selection
    models_to_train = st.multiselect(
        "Choose ML Models for Comparison",
        options=['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 
                 'Random Forest', 'Gradient Boosting', 'SVM', 'XGBoost'],
        default=['Logistic Regression', 'Random Forest', 'XGBoost']
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Model Parameters")
    
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", value=42, step=1)
    
    st.markdown("---")
    
    st.markdown("### 📈 About")
    st.markdown("""
    This system uses machine learning to predict the likelihood of heart disease based on clinical parameters.
    
    **Features used:**
    - Age, Sex, Chest Pain Type
    - Resting BP, Cholesterol
    - Fasting BS, Resting ECG
    - Max HR, Exercise Angina
    - Oldpeak, ST Slope
    
    **Target:** Heart Disease (0 = No, 1 = Yes)
    """)

# Main content
if 'data' in locals() or use_sample:
    
    # Initialize session state for data if not exists
    if 'heart_data' not in st.session_state and use_sample:
        # Store the CSV content
        import base64
        st.session_state['heart_data'] = """
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
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
"""
        data = pd.read_csv(StringIO(st.session_state['heart_data']))
    
    # Data preprocessing function
    def preprocess_data(df):
        df_processed = df.copy()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                label_encoders[col] = le
        
        # Handle missing values
        df_processed = df_processed.dropna()
        
        # Separate features and target
        target_col = 'HeartDisease' if 'HeartDisease' in df_processed.columns else 'target'
        if 'HeartDisease' in df_processed.columns:
            X = df_processed.drop('HeartDisease', axis=1)
            y = df_processed['HeartDisease']
        else:
            X = df_processed
            y = None
        
        return X, y, label_encoders
    
    # Train models function
    def train_models(X_train, X_test, y_train, y_test, models_list, random_state):
        models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
            'SVM': SVC(random_state=random_state, probability=True),
            'XGBoost': XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        }
        
        results = {}
        trained_models = {}
        
        for model_name in models_list:
            if model_name in models:
                model = models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # ROC-AUC
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                else:
                    roc_auc = None
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'roc_auc': roc_auc,
                    'model': model,
                    'predictions': y_pred
                }
                trained_models[model_name] = model
        
        return results, trained_models
    
    # Display data overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Features</div>
        </div>
        """.format(len(data.columns) - 1), unsafe_allow_html=True)
    
    with col3:
        if 'HeartDisease' in data.columns:
            disease_count = data['HeartDisease'].sum()
            disease_pct = (disease_count / len(data)) * 100
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Heart Disease Prevalence</div>
            </div>
            """.format(disease_pct), unsafe_allow_html=True)
    
    # Data preview
    with st.expander("📊 Data Preview"):
        st.dataframe(data.head(10))
        
        # Basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numerical Features Statistics**")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            st.dataframe(data[numeric_cols].describe())
        
        with col2:
            st.markdown("**Categorical Features**")
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                st.write(f"**{col}:** {data[col].nunique()} unique values")
    
    # Visualization section
    st.markdown("---")
    st.markdown("### 📊 Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Feature Distribution", "Correlation Analysis", "Target Analysis"])
    
    with tab1:
        # Feature distributions
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        selected_feature = st.selectbox("Select feature to visualize", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(data, x=selected_feature, color='HeartDisease' if 'HeartDisease' in data.columns else None,
                              title=f"Distribution of {selected_feature}",
                              color_discrete_sequence=['#2c5f2d', '#ff6b6b'])
            fig.update_layout(bargap=0.1, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'HeartDisease' in data.columns:
                fig = px.box(data, x='HeartDisease', y=selected_feature, 
                           title=f"{selected_feature} by Heart Disease",
                           color='HeartDisease', 
                           color_discrete_sequence=['#2c5f2d', '#ff6b6b'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlation Matrix")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'HeartDisease' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Target distribution
                target_counts = data['HeartDisease'].value_counts()
                fig = px.pie(values=target_counts.values, 
                            names=['No Disease', 'Disease'],
                            title="Heart Disease Distribution",
                            color_discrete_sequence=['#2c5f2d', '#ff6b6b'],
                            hole=0.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Target by sex
                if 'Sex' in data.columns:
                    sex_disease = pd.crosstab(data['Sex'], data['HeartDisease'])
                    fig = px.bar(sex_disease, 
                                title="Heart Disease by Sex",
                                barmode='group',
                                color_discrete_sequence=['#2c5f2d', '#ff6b6b'])
                    st.plotly_chart(fig, use_container_width=True)
    
    # Model Training Section
    st.markdown("---")
    st.markdown("### 🤖 Model Training & Comparison")
    
    if st.button("🚀 Train Selected Models", use_container_width=True):
        with st.spinner("Training models... This may take a few moments."):
            # Preprocess data
            X, y, encoders = preprocess_data(data)
            
            if y is None:
                st.error("Target column 'HeartDisease' not found in the dataset!")
                st.stop()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test, models_to_train, random_state)
            
            # Display results
            st.markdown("### 📈 Model Performance Comparison")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[m]['accuracy'] for m in results],
                'CV Score (Mean)': [results[m]['cv_mean'] for m in results],
                'CV Score (Std)': [results[m]['cv_std'] for m in results],
                'ROC-AUC': [results[m]['roc_auc'] if results[m]['roc_auc'] else 0 for m in results]
            })
            
            # Sort by accuracy
            results_df = results_df.sort_values('Accuracy', ascending=False)
            
            # Display metrics table
            st.dataframe(
                results_df.style.format({
                    'Accuracy': '{:.3f}',
                    'CV Score (Mean)': '{:.3f}',
                    'CV Score (Std)': '{:.3f}',
                    'ROC-AUC': '{:.3f}'
                }).background_gradient(subset=['Accuracy'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig = px.bar(results_df, x='Model', y='Accuracy', 
                           title='Model Accuracy Comparison',
                           color='Accuracy',
                           color_continuous_scale='RdYlGn',
                           text='Accuracy')
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ROC-AUC comparison
                fig = px.bar(results_df, x='Model', y='ROC-AUC',
                           title='ROC-AUC Score Comparison',
                           color='ROC-AUC',
                           color_continuous_scale='RdYlGn',
                           text='ROC-AUC')
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Best model selection
            best_model_name = results_df.iloc[0]['Model']
            best_accuracy = results_df.iloc[0]['Accuracy']
            
            st.markdown(f"""
            <div class="success-message">
                <strong>🏆 Best Performing Model: {best_model_name}</strong><br>
                Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics for best model
            st.markdown("### 📋 Detailed Classification Report")
            
            selected_model = st.selectbox("Select model to view detailed metrics", results_df['Model'].tolist())
            
            if selected_model in results:
                model = results[selected_model]['model']
                predictions = results[selected_model]['predictions']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, predictions)
                    fig = px.imshow(cm, 
                                   text_auto=True,
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['No Disease', 'Disease'],
                                   y=['No Disease', 'Disease'],
                                   title=f"Confusion Matrix - {selected_model}",
                                   color_continuous_scale='Blues')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Classification Report as DataFrame
                    report = classification_report(y_test, predictions, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format('{:.3f}'), use_container_width=True)
            
            # Feature Importance (for tree-based models)
            st.markdown("### 🌟 Feature Importance Analysis")
            
            if 'Random Forest' in trained_models or 'Gradient Boosting' in trained_models or 'XGBoost' in trained_models:
                feature_importance_model = st.selectbox(
                    "Select model for feature importance",
                    [m for m in ['Random Forest', 'Gradient Boosting', 'XGBoost'] if m in trained_models]
                )
                
                if feature_importance_model in trained_models:
                    model = trained_models[feature_importance_model]
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = X.columns
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=True)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature',
                                   orientation='h',
                                   title=f"Feature Importance - {feature_importance_model}",
                                   color='Importance',
                                   color_continuous_scale='RdYlGn')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Save models to session state for prediction
            st.session_state['trained_models'] = trained_models
            st.session_state['scaler'] = scaler
            st.session_state['encoders'] = encoders
            st.session_state['features'] = X.columns.tolist()
    
    # Prediction Section
    st.markdown("---")
    st.markdown("### 🔮 Make Predictions")
    
    if 'trained_models' in st.session_state:
        st.markdown("Enter patient information to predict heart disease risk:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", ['M', 'F'])
            cp = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
        
        with col2:
            resting_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
            cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        
        with col3:
            resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
            max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
            exercise_angina = st.selectbox("Exercise Induced Angina", ['N', 'Y'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=6.0, value=0.0, step=0.1)
        
        with col2:
            st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])
        
        with col3:
            prediction_model = st.selectbox("Select Model for Prediction", list(st.session_state['trained_models'].keys()))
        
        if st.button("🩺 Predict Heart Disease Risk", use_container_width=True):
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
                if col in st.session_state['encoders']:
                    encoder = st.session_state['encoders'][col]
                    input_data[col] = encoder.transform(input_data[col])
            
            # Ensure columns match
            for col in st.session_state['features']:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            input_data = input_data[st.session_state['features']]
            
            # Scale features
            input_scaled = st.session_state['scaler'].transform(input_data)
            
            # Make prediction
            model = st.session_state['trained_models'][prediction_model]
            prediction = model.predict(input_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_scaled)[0][1]
            else:
                probability = None
            
            # Display results
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="warning-message">
                        <strong>⚠️ High Risk of Heart Disease</strong><br>
                        The model predicts an elevated risk of heart disease.<br>
                        Please consult with a healthcare provider for proper evaluation.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-message">
                        <strong>✅ Low Risk of Heart Disease</strong><br>
                        The model predicts low risk of heart disease.<br>
                        Continue maintaining a healthy lifestyle.
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if probability is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{probability*100:.1f}%</div>
                        <div class="metric-label">Probability of Heart Disease</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Risk meter
            if probability is not None:
                risk_level = probability * 100
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_level,
                    title = {'text': "Risk Assessment Meter"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "#d4edda"},
                            {'range': [30, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#f8d7da"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_level
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if prediction == 1:
                st.markdown("""
                - 🏥 Schedule an appointment with a cardiologist
                - 🍎 Adopt a heart-healthy diet (Mediterranean diet recommended)
                - 🏃‍♂️ Start a regular exercise routine (consult with doctor first)
                - 📊 Monitor blood pressure and cholesterol regularly
                - 🚭 Avoid smoking and limit alcohol consumption
                - 🧘 Practice stress management techniques
                """)
            else:
                st.markdown("""
                - ✅ Continue healthy lifestyle habits
                - 🍏 Maintain balanced diet rich in fruits and vegetables
                - 🏃 Regular physical activity (150 minutes/week)
                - 📅 Annual health check-ups
                - 💤 Get adequate sleep (7-8 hours)
                - 🧠 Manage stress through meditation or hobbies
                """)
    else:
        st.info("👈 Train models first to make predictions!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>⚠️ Medical Disclaimer: This application is for educational purposes only. 
    Always consult with a qualified healthcare provider for medical decisions.</p>
    <p>© 2024 Heart Disease Prediction System | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
