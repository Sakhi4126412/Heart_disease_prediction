# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
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

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
    }
    .disclaimer {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "❤️ Prediction", "⚡ Quick Prediction", "📊 Visualizations", "🧪 Model Comparison", "ℹ️ About"]
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "**Heart Disease Prediction System**\n\n"
    "Using Machine Learning to predict heart disease risk based on clinical parameters."
)

# Load default dataset from the provided CSV content
@st.cache_data
def load_default_data():
    # Create dataframe from the provided heart.csv data
    data = pd.read_csv('heart.csv') if 'heart.csv' in __import__('os').listdir('.') else None
    
    if data is None:
        # Fallback sample data
        data = pd.DataFrame({
            'Age': [40, 49, 37, 48, 54],
            'Sex': ['M', 'F', 'M', 'F', 'M'],
            'ChestPainType': ['ATA', 'NAP', 'ATA', 'ASY', 'NAP'],
            'RestingBP': [140, 160, 130, 138, 150],
            'Cholesterol': [289, 180, 283, 214, 195],
            'FastingBS': [0, 0, 0, 0, 0],
            'RestingECG': ['Normal', 'Normal', 'ST', 'Normal', 'Normal'],
            'MaxHR': [172, 156, 98, 108, 122],
            'ExerciseAngina': ['N', 'N', 'N', 'Y', 'N'],
            'Oldpeak': [0, 1, 0, 1.5, 0],
            'ST_Slope': ['Up', 'Flat', 'Up', 'Flat', 'Up'],
            'HeartDisease': [0, 1, 0, 1, 0]
        })
    return data

# File upload section
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.sidebar.success(f"✅ Loaded {len(data)} records!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        data = load_default_data()
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.sidebar.info("📊 Using default dataset")
else:
    # Try to load heart.csv from the same directory
    try:
        data = pd.read_csv('heart.csv')
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.sidebar.info("📊 Using heart.csv dataset")
    except:
        data = load_default_data()
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.sidebar.info("📊 Using sample dataset")

# Data preprocessing function with proper encoding
def preprocess_data(df):
    df_processed = df.copy()
    
    # Define mappings for categorical columns
    mappings = {
        'Sex': {'M': 1, 'F': 0},
        'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3},
        'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
        'ExerciseAngina': {'N': 0, 'Y': 1},
        'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
    }
    
    # Apply mappings
    for col, mapping in mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
    
    # Handle any remaining non-numeric columns
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].dtype in ['int64', 'float64']:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Separate features and target
    target_col = 'HeartDisease' if 'HeartDisease' in df_processed.columns else 'target'
    if target_col in df_processed.columns:
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
    else:
        X = df_processed
        y = None
    
    # Ensure all columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

# Optimized model training with hyperparameter tuning
@st.cache_resource
def train_optimized_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimized models with best parameters
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, 
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        'SVM': SVC(
            random_state=42, 
            probability=True,
            C=1.0,
            kernel='rbf',
            gamma='scale'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5
        )
    }
    
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # ROC-AUC score
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        trained_models[name] = model
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'roc_auc': roc_auc,
            'model': model,
            'predictions': y_pred,
            'y_test': y_test
        }
    
    return trained_models, results

# Train models if data is available
if st.session_state.data_loaded and st.session_state.data is not None:
    try:
        X_scaled, y, scaler, feature_names = preprocess_data(st.session_state.data)
        if y is not None:
            if st.session_state.models is None:
                models, results = train_optimized_models(X_scaled, y)
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.scaler = scaler
                st.session_state.feature_names = feature_names
                st.session_state.models_trained = True
            else:
                models = st.session_state.models
                results = st.session_state.results
                scaler = st.session_state.scaler
                feature_names = st.session_state.feature_names
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        st.session_state.models_trained = False

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction System</h1>
        <p>Advanced Machine Learning for Early Detection & Prevention</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This system uses **Machine Learning** to predict the risk of heart disease based on 
        clinical parameters. Early detection can significantly improve patient outcomes.
        
        **Key Features:**
        - 🔮 Real-time risk prediction
        - ⚡ Quick prediction with basic parameters
        - 📊 Interactive visualizations
        - 🤖 Multiple optimized ML models
        - 💡 Personalized recommendations
        """)
        
        st.markdown("### 🧠 Machine Learning Models")
        st.markdown("""
        Four optimized algorithms are implemented:
        
        1. **Logistic Regression** - Baseline statistical model
        2. **Random Forest** - Ensemble learning with 200 trees
        3. **SVM** - Support Vector Machine with RBF kernel
        4. **Gradient Boosting** - Advanced boosting algorithm
        """)
    
    with col2:
        st.markdown("### 📊 Dataset Information")
        if st.session_state.data is not None:
            st.markdown(f"""
            **Total Records:** {len(st.session_state.data)}  
            **Features:** {len(st.session_state.data.columns) - 1} clinical parameters  
            **Target Variable:** HeartDisease (0 = No Disease, 1 = Disease)
            
            **Clinical Features:**
            - Age, Sex, Chest Pain Type
            - Resting BP, Cholesterol
            - Fasting BS, Resting ECG
            - Max HR, Exercise Angina
            - Oldpeak, ST Slope
            """)
            
            if 'HeartDisease' in st.session_state.data.columns:
                disease_count = st.session_state.data['HeartDisease'].sum()
                healthy_count = len(st.session_state.data) - disease_count
                st.markdown(f"""
                **Target Distribution:**
                - ❤️ Disease Cases: {disease_count} ({disease_count/len(st.session_state.data)*100:.1f}%)
                - 💚 Healthy Cases: {healthy_count} ({healthy_count/len(st.session_state.data)*100:.1f}%)
                """)
    
    st.markdown("---")
    st.markdown("### 🚀 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Input Data")
        st.markdown("Enter patient clinical parameters")
    
    with col2:
        st.markdown("#### 2️⃣ Model Prediction")
        st.markdown("AI models analyze the data and predict risk level")
    
    with col3:
        st.markdown("#### 3️⃣ Get Results")
        st.markdown("Receive risk assessment and health recommendations")

# ==================== PREDICTION PAGE (Detailed) ====================
elif page == "❤️ Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Detailed Heart Disease Risk Prediction</h1>
        <p>Enter complete patient information for accurate risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models are not trained yet. Please check the data and refresh the page.")
    else:
        st.markdown("### 📝 Patient Clinical Data")
        
        # Create input columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", 
                             ["ATA", "NAP", "ASY", "TA"],
                             help="ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic, TA: Typical Angina")
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        
        with col2:
            cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
            max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
        
        with col3:
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-3.0, max_value=6.0, value=1.0, step=0.1)
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        
        # Convert categorical to numeric
        sex_num = 1 if sex == "Male" else 0
        cp_num = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[cp]
        fasting_bs_num = 1 if fasting_bs == "Yes" else 0
        resting_ecg_num = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg]
        exercise_angina_num = 1 if exercise_angina == "Yes" else 0
        st_slope_num = {"Up": 0, "Flat": 1, "Down": 2}[st_slope]
        
        # Model selection
        st.markdown("---")
        selected_model = st.selectbox(
            "Select Model for Prediction",
            ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"]
        )
        
        if st.button("🔮 Predict Heart Disease Risk", use_container_width=True):
            try:
                # Create input array with all features
                input_data = np.array([[age, sex_num, cp_num, resting_bp, cholesterol, 
                                        fasting_bs_num, resting_ecg_num, max_hr, 
                                        exercise_angina_num, oldpeak, st_slope_num]])
                
                # Scale input
                input_scaled = st.session_state.scaler.transform(input_data)
                
                # Make prediction
                model = st.session_state.models[selected_model]
                prediction = model.predict(input_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_scaled)[0][1]
                else:
                    probability = None
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="risk-high">
                            <h2>⚠️ HIGH RISK</h2>
                            <p>Heart Disease Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="risk-low">
                            <h2>✅ LOW RISK</h2>
                            <p>No Heart Disease Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if probability is not None:
                        risk_level = probability * 100
                        
                        # Risk meter
                        fig = go.Figure(go.Indicator(
                            mode="gauge",
                            value=risk_level,
                            title={"text": "Risk Level"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "darkred" if risk_level > 70 else "orange" if risk_level > 40 else "green"},
                                "steps": [
                                    {"range": [0, 40], "color": "#d4edda"},
                                    {"range": [40, 70], "color": "#fff3cd"},
                                    {"range": [70, 100], "color": "#f8d7da"}
                                ]
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Determine risk level text
                        if risk_level > 70:
                            risk_text = "🔴 High Risk"
                        elif risk_level > 40:
                            risk_text = "🟡 Medium Risk"
                        else:
                            risk_text = "🟢 Low Risk"
                        
                        st.markdown(f"<h3 style='text-align: center'>{risk_text}</h3>", unsafe_allow_html=True)
                
                # Health Recommendations
                st.markdown("---")
                st.markdown("## 💡 Health Recommendations")
                
                if prediction == 1:
                    st.markdown("""
                    ### ⚠️ High Risk - Recommended Actions:
                    
                    **Immediate Steps:**
                    - 🏥 Schedule an appointment with a cardiologist immediately
                    - 📊 Get a complete cardiac evaluation (ECG, Echo, Stress Test)
                    - 💊 Discuss preventive medications with your doctor
                    
                    **Lifestyle Changes:**
                    - 🍎 Adopt a heart-healthy Mediterranean diet
                    - 🏃 Start a supervised exercise program (30 mins/day, 5 days/week)
                    - 🚭 Quit smoking and limit alcohol consumption
                    - 📉 Monitor blood pressure and cholesterol regularly
                    - 🧘 Practice stress management techniques (meditation, yoga)
                    """)
                else:
                    st.markdown("""
                    ### ✅ Low Risk - Preventive Recommendations:
                    
                    **Maintain Healthy Habits:**
                    - 🍏 Continue balanced diet rich in fruits, vegetables, and whole grains
                    - 🏃 Regular physical activity (150 minutes moderate exercise/week)
                    - 💤 Get 7-8 hours of quality sleep
                    - 🧠 Manage stress through hobbies and relaxation
                    
                    **Preventive Care:**
                    - 📅 Annual health check-ups
                    - ❤️ Monitor blood pressure (target <120/80)
                    - 📊 Check cholesterol levels every 4-6 years
                    - 🩸 Screen for diabetes if at risk
                    """)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        
        # Disclaimer after prediction
        st.markdown("---")
        st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ Medical Disclaimer:</strong> This is an AI-powered screening tool for educational purposes only. 
            It does not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified 
            healthcare provider for medical decisions. The predictions are based on machine learning models and 
            should be used as a reference only, not as a definitive diagnosis.
        </div>
        """, unsafe_allow_html=True)

# ==================== QUICK PREDICTION PAGE ====================
elif page == "⚡ Quick Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Quick Heart Disease Risk Assessment</h1>
        <p>Enter basic health parameters for a quick risk check</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models are not trained yet. Please wait for data processing.")
    else:
        st.markdown("### 📝 Quick Health Check")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=50, key="quick_age")
            sex = st.selectbox("Gender", ["Male", "Female"], key="quick_sex")
            heart_rate = st.number_input("Heart Rate (beats per minute)", min_value=40, max_value=200, value=75, key="quick_hr")
            cholesterol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=200, key="quick_chol")
        
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, key="quick_weight")
            # BMI calculation
            height_m = st.number_input("Height (m) - for BMI calculation", min_value=1.0, max_value=2.5, value=1.7, step=0.01, key="quick_height")
            bmi = weight_kg / (height_m ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
            # Additional basic parameters
            blood_pressure = st.selectbox("Blood Pressure Category", 
                                         ["Normal (<120/80)", "Elevated (120-129/80)", "High BP Stage 1 (130-139/80-89)", 
                                          "High BP Stage 2 (140+/90+)", "Hypertensive Crisis (>180/120)"])
            exercise = st.selectbox("Regular Exercise", ["Yes", "No"], key="quick_exercise")
        
        st.markdown("---")
        selected_quick_model = st.selectbox(
            "Select Model for Quick Prediction",
            ["Random Forest", "Logistic Regression", "Gradient Boosting", "SVM"],
            key="quick_model"
        )
        
        if st.button("⚡ Get Quick Risk Assessment", use_container_width=True, key="quick_predict"):
            try:
                # Convert basic inputs to match model features
                sex_num = 1 if sex == "Male" else 0
                exercise_num = 1 if exercise == "Yes" else 0
                
                # Blood pressure to numeric mapping
                bp_mapping = {
                    "Normal (<120/80)": 110,
                    "Elevated (120-129/80)": 125,
                    "High BP Stage 1 (130-139/80-89)": 135,
                    "High BP Stage 2 (140+/90+)": 150,
                    "Hypertensive Crisis (>180/120)": 180
                }
                estimated_bp = bp_mapping[blood_pressure]
                
                # Estimate other parameters based on quick inputs
                estimated_max_hr = 220 - age  # Standard formula
                
                # Default values for parameters not provided
                cp_num = 1  # Default to NAP (Non-anginal Pain)
                fasting_bs_num = 0  # Default to No
                resting_ecg_num = 0  # Default to Normal
                oldpeak = 0.5
                st_slope_num = 1  # Default to Flat
                
                # Create input array
                input_data = np.array([[age, sex_num, cp_num, estimated_bp, cholesterol, 
                                        fasting_bs_num, resting_ecg_num, estimated_max_hr, 
                                        exercise_num, oldpeak, st_slope_num]])
                
                # Scale input
                input_scaled = st.session_state.scaler.transform(input_data)
                
                # Make prediction
                model = st.session_state.models[selected_quick_model]
                prediction = model.predict(input_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_scaled)[0][1]
                else:
                    probability = None
                
                # Display quick results
                st.markdown("---")
                st.markdown("## ⚡ Quick Assessment Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="risk-high">
                            <h2>⚠️ HIGH RISK</h2>
                            <p>Possible Heart Disease Risk Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        **Based on your quick assessment:**
                        - Age: {} years
                        - {} • Heart Rate: {} bpm
                        - Cholesterol: {} mg/dl
                        - BMI: {:.1f}
                        
                        **🚨 Recommendation:** Please consult a healthcare professional for a complete evaluation.
                        """.format(age, sex, heart_rate, cholesterol, bmi))
                    else:
                        st.markdown("""
                        <div class="risk-low">
                            <h2>✅ LOW RISK</h2>
                            <p>Low Risk of Heart Disease Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        **Based on your quick assessment:**
                        - Age: {} years
                        - {} • Heart Rate: {} bpm
                        - Cholesterol: {} mg/dl
                        - BMI: {:.1f}
                        
                        **💚 Good News:** Your quick assessment shows low risk. Continue healthy habits!
                        """.format(age, sex, heart_rate, cholesterol, bmi))
                
                with col2:
                    if probability is not None:
                        risk_level = probability * 100
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge",
                            value=risk_level,
                            title={"text": "Risk Score"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "darkred" if risk_level > 70 else "orange" if risk_level > 40 else "green"},
                                "steps": [
                                    {"range": [0, 40], "color": "#d4edda"},
                                    {"range": [40, 70], "color": "#fff3cd"},
                                    {"range": [70, 100], "color": "#f8d7da"}
                                ]
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if risk_level > 70:
                            st.markdown("🔴 **Risk Level: High** - Schedule a checkup soon")
                        elif risk_level > 40:
                            st.markdown("🟡 **Risk Level: Medium** - Monitor your health closely")
                        else:
                            st.markdown("🟢 **Risk Level: Low** - Maintain healthy lifestyle")
                
                # Quick health tips
                st.markdown("---")
                st.markdown("### 💡 Quick Health Tips")
                
                if bmi > 30:
                    st.warning("⚠️ Your BMI indicates obesity. Consider consulting a nutritionist.")
                elif bmi > 25:
                    st.info("📊 Your BMI is in overweight range. Regular exercise can help.")
                else:
                    st.success("✅ Your BMI is in healthy range. Keep it up!")
                
                if cholesterol > 240:
                    st.warning("⚠️ High cholesterol detected. Consider dietary changes.")
                elif cholesterol > 200:
                    st.info("📊 Borderline cholesterol. Monitor your diet.")
                
                if heart_rate > 100:
                    st.warning("⚠️ Elevated heart rate. Consider relaxation techniques.")
                elif heart_rate < 60 and age < 60:
                    st.info("📊 Low heart rate. Consult a doctor if you experience symptoms.")
                
            except Exception as e:
                st.error(f"Error during quick prediction: {str(e)}")
        
        # Disclaimer for quick prediction
        st.markdown("---")
        st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ Quick Assessment Disclaimer:</strong> This quick assessment uses estimated values based on limited inputs 
            and is for screening purposes only. Results may not be as accurate as the detailed prediction. 
            Always consult a healthcare professional for proper medical advice and diagnosis.
        </div>
        """, unsafe_allow_html=True)

# ==================== VISUALIZATIONS PAGE ====================
elif page == "📊 Visualizations":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Visualizations</h1>
        <p>Explore patterns and insights in heart disease data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        df_viz = st.session_state.data
        
        # Target Distribution
        if 'HeartDisease' in df_viz.columns:
            st.markdown("### 🎯 Target Variable Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                target_counts = df_viz['HeartDisease'].value_counts()
                fig = px.pie(values=target_counts.values, names=['No Disease', 'Disease'],
                            title="Heart Disease Distribution",
                            color_discrete_sequence=['#2ecc71', '#e74c3c'],
                            hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(x=['No Disease', 'Disease'], y=target_counts.values,
                            title="Count of Cases",
                            color=['No Disease', 'Disease'],
                            color_discrete_sequence=['#2ecc71', '#e74c3c'],
                            text=target_counts.values)
                fig.update_layout(xaxis_title="Heart Disease", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
            # Age Distribution
            if 'Age' in df_viz.columns:
                st.markdown("### 📅 Age Distribution by Heart Disease")
                fig = px.histogram(df_viz, x='Age', color='HeartDisease', nbins=30,
                                  title="Age Distribution by Heart Disease Status",
                                  color_discrete_sequence=['#2ecc71', '#e74c3c'])
                fig.update_layout(barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)
            
            # Cholesterol Distribution
            if 'Cholesterol' in df_viz.columns:
                st.markdown("### 🩸 Cholesterol Levels by Heart Disease")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.box(df_viz, x='HeartDisease', y='Cholesterol', color='HeartDisease',
                                title="Cholesterol Distribution",
                                color_discrete_sequence=['#2ecc71', '#e74c3c'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Heatmap
            st.markdown("### 🔥 Correlation Heatmap")
            numeric_cols = df_viz.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = df_viz[numeric_cols].corr()
                fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                                color_continuous_scale='RdBu_r',
                                title="Feature Correlation Matrix")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Max Heart Rate Distribution
            if 'MaxHR' in df_viz.columns:
                st.markdown("### 📈 Maximum Heart Rate Distribution")
                fig = px.histogram(df_viz, x='MaxHR', color='HeartDisease', 
                                  title="Maximum Heart Rate Distribution",
                                  color_discrete_sequence=['#2ecc71', '#e74c3c'],
                                  nbins=30)
                st.plotly_chart(fig, use_container_width=True)
            
            # Chest Pain Type Analysis
            if 'ChestPainType' in df_viz.columns:
                st.markdown("### 💔 Chest Pain Type Analysis")
                cp_counts = pd.crosstab(df_viz['ChestPainType'], df_viz['HeartDisease'], normalize='index') * 100
                fig = px.bar(cp_counts, title="Heart Disease Rate by Chest Pain Type",
                            labels={'value': 'Percentage (%)', 'ChestPainType': 'Chest Pain Type'},
                            color_discrete_sequence=['#2ecc71', '#e74c3c'],
                            barmode='group')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data loaded. Please upload a dataset.")
    
    # Disclaimer for visualizations
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> These visualizations are based on the provided dataset and are for informational 
        and educational purposes only. They do not constitute medical advice or diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ==================== MODEL COMPARISON PAGE ====================
elif page == "🧪 Model Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Model Performance Comparison</h1>
        <p>Evaluate and compare different machine learning models</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models are not trained yet. Please check the data and refresh the page.")
    elif st.session_state.results is not None:
        st.markdown("### 📊 Model Performance Metrics")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': list(st.session_state.results.keys()),
            'Accuracy': [st.session_state.results[model]['accuracy'] for model in st.session_state.results],
            'CV Score': [st.session_state.results[model]['cv_mean'] for model in st.session_state.results],
            'ROC-AUC': [st.session_state.results[model]['roc_auc'] if st.session_state.results[model]['roc_auc'] else 0 for model in st.session_state.results]
        })
        
        # Sort by accuracy
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                comparison_df.style.format({'Accuracy': '{:.3f}', 'CV Score': '{:.3f}', 'ROC-AUC': '{:.3f}'})
                .background_gradient(subset=['Accuracy'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        with col2:
            fig = px.bar(comparison_df, x='Model', y='Accuracy',
                        title="Model Accuracy Comparison",
                        color='Accuracy', color_continuous_scale='RdYlGn',
                        text='Accuracy')
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # ROC-AUC Comparison
        st.markdown("### 📈 ROC-AUC Score Comparison")
        fig = px.bar(comparison_df, x='Model', y='ROC-AUC',
                    title="ROC-AUC Score Comparison",
                    color='ROC-AUC', color_continuous_scale='RdYlGn',
                    text='ROC-AUC')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrices for each model
        st.markdown("### 📈 Confusion Matrices")
        
        tabs = st.tabs(list(st.session_state.results.keys()))
        
        for idx, (model_name, model_data) in enumerate(st.session_state.results.items()):
            with tabs[idx]:
                col1, col2 = st.columns(2)
                
                with col1:
                    cm = confusion_matrix(model_data['y_test'], model_data['predictions'])
                    fig = px.imshow(cm, text_auto=True,
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=['No Disease', 'Disease'],
                                   y=['No Disease', 'Disease'],
                                   title=f"{model_name} - Confusion Matrix",
                                   color_continuous_scale='Blues')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Classification report
                    report = classification_report(model_data['y_test'], model_data['predictions'], 
                                                  target_names=['No Disease', 'Disease'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format('{:.3f}'), use_container_width=True)
        
        # Feature Importance for Random Forest
        if 'Random Forest' in st.session_state.models:
            st.markdown("### 🌟 Feature Importance Analysis")
            
            rf_model = st.session_state.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Random Forest Feature Importance",
                        color='Importance', color_continuous_scale='RdYlGn')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Recommendations
        st.markdown("### 💡 Model Selection Recommendation")
        best_model = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        
        recommendations = {
            "Random Forest": f"🏆 **Recommended Model: Random Forest**\n\nAccuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)\n\nRandom Forest performs best on this dataset due to its ability to handle complex interactions and provide feature importance insights.",
            "Gradient Boosting": f"🏆 **Recommended Model: Gradient Boosting**\n\nAccuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)\n\nGradient Boosting offers excellent predictive power by combining multiple weak learners.",
            "Logistic Regression": f"🏆 **Recommended Model: Logistic Regression**\n\nAccuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)\n\nLogistic Regression offers good performance with high interpretability.",
            "SVM": f"🏆 **Recommended Model: SVM**\n\nAccuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)\n\nSVM performs well with the RBF kernel for complex decision boundaries."
        }
        
        st.success(recommendations.get(best_model, f"🏆 Best Model: {best_model} with {best_accuracy:.3f} accuracy"))
    
    # Disclaimer for model comparison
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> Model performance metrics are based on the provided dataset and cross-validation. 
        Real-world performance may vary. These results are for educational and comparison purposes only.
    </div>
    """, unsafe_allow_html=True)

# ==================== ABOUT PAGE ====================
else:
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About This Project</h1>
        <p>Heart Disease Prediction System - Data Science Final Year Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Project Objective")
        st.markdown("""
        This project uses machine learning to predict heart disease risk using clinical parameters.
        Early detection can significantly improve patient outcomes and reduce healthcare costs.
        """)
        
        st.markdown("### 🛠️ Technologies Used")
        st.markdown("""
        - **Frontend:** Streamlit
        - **Machine Learning:** scikit-learn
        - **Data Processing:** Pandas, NumPy
        - **Visualization:** Plotly
        - **Models:** Logistic Regression, Random Forest, SVM, Gradient Boosting
        """)
        
        st.markdown("### 📊 Dataset Features")
        st.markdown("""
        The dataset contains 11 clinical features:
        - **Age:** Age in years
        - **Sex:** Male/Female
        - **ChestPainType:** ATA, NAP, ASY, TA
        - **RestingBP:** Resting blood pressure
        - **Cholesterol:** Serum cholesterol
        - **FastingBS:** Fasting blood sugar
        - **RestingECG:** Resting ECG results
        - **MaxHR:** Maximum heart rate
        - **ExerciseAngina:** Exercise induced angina
        - **Oldpeak:** ST depression
        - **ST_Slope:** ST slope
        """)
    
    with col2:
        st.markdown("### 🤖 Machine Learning Models")
        st.markdown("""
        Four optimized models are implemented:
        
        **1. Logistic Regression**
        - Simple and interpretable
        - Optimized with C=1.0, liblinear solver
        
        **2. Random Forest**
        - 200 trees, max_depth=15
        - Handles non-linear relationships
        - Provides feature importance
        
        **3. SVM**
        - RBF kernel, C=1.0
        - Effective for complex decision boundaries
        
        **4. Gradient Boosting**
        - 150 estimators, learning_rate=0.1
        - Advanced boosting algorithm
        """)
        
        st.markdown("### 📈 Model Evaluation Metrics")
        st.markdown("""
        Models are evaluated using:
        - Accuracy Score
        - Cross-Validation (5-fold)
        - ROC-AUC Score
        - Confusion Matrix
        - Classification Report
        """)
        
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
        **Data Science Final Year Project**
        
        This system demonstrates the application of machine learning in healthcare
        for early heart disease detection and prevention.
        """)
    
    st.markdown("---")
    st.markdown("### 📚 References")
    st.markdown("""
    - UCI Machine Learning Repository - Heart Disease Dataset
    - American Heart Association Guidelines
    - scikit-learn Documentation
    - Streamlit Documentation
    """)
    
    # Final disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ IMPORTANT MEDICAL DISCLAIMER:</strong> This system is for educational and research purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. Never disregard professional 
        medical advice or delay seeking it because of information provided by this system. The predictions made by 
        this system should not be used as the sole basis for any medical decision. Always consult with a qualified 
        healthcare provider for proper medical evaluation and treatment.
    </div>
    """, unsafe_allow_html=True)

# Global disclaimer at the bottom of every page
st.markdown("---")
st.markdown("""
<div style="background: #f8d7da; padding: 0.8rem; border-radius: 8px; margin-top: 1rem; text-align: center; font-size: 0.85rem;">
    <strong>⚠️ Medical Disclaimer:</strong> This is an AI-powered screening tool for educational purposes only. 
    Always consult with a qualified healthcare provider for medical decisions.
</div>
""", unsafe_allow_html=True)
