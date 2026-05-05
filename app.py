# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .feature-importance {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "❤️ Prediction", "📊 Visualizations", "🧪 Model Comparison", "ℹ️ About"]
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "**Heart Disease Prediction System**\n\n"
    "Using Machine Learning to predict heart disease risk based on clinical parameters."
)

# Load default dataset function
@st.cache_data
def load_default_data():
    # Create sample data if no file uploaded
    data = pd.DataFrame({
        'age': [52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
        'sex': [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
        'cp': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        'trestbps': [125, 140, 130, 132, 148, 140, 120, 130, 140, 128],
        'chol': [212, 203, 256, 234, 284, 206, 234, 284, 294, 308],
        'fbs': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        'restecg': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'thalach': [168, 155, 150, 140, 142, 155, 146, 138, 112, 145],
        'exang': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        'oldpeak': [1.2, 1.5, 2.3, 0.8, 1.4, 1.6, 0.5, 2.0, 1.8, 2.2],
        'slope': [2, 1, 2, 2, 1, 2, 2, 1, 2, 2],
        'ca': [0, 0, 1, 0, 1, 0, 0, 2, 0, 1],
        'thal': [2, 3, 2, 2, 3, 2, 2, 3, 2, 2],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
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
        st.sidebar.success(f"✅ Loaded {len(data)} records!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        data = load_default_data()
        st.session_state.data = data
        st.sidebar.info("Using default dataset")
else:
    data = load_default_data()
    st.session_state.data = data
    st.sidebar.info("📊 Using default dataset")

# Data preprocessing function
def preprocess_data(df):
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.fillna(df_processed.median())
    
    # Separate features and target
    if 'target' in df_processed.columns:
        X = df_processed.drop('target', axis=1)
        y = df_processed['target']
    else:
        X = df_processed
        y = None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

# Train models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        trained_models[name] = model
        results[name] = {
            'accuracy': accuracy,
            'model': model,
            'predictions': y_pred,
            'y_test': y_test
        }
    
    return trained_models, results, X_train, X_test, y_train, y_test

# Train models if not already trained
if st.session_state.data is not None:
    X_scaled, y, scaler, feature_names = preprocess_data(st.session_state.data)
    if y is not None:
        models, results, X_train, X_test, y_train, y_test = train_models(X_scaled, y)

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
        - 📊 Interactive visualizations
        - 🤖 Multiple ML models
        - 💡 Personalized recommendations
        """)
        
        st.markdown("### 🧠 Machine Learning Models")
        st.markdown("""
        Three powerful algorithms are implemented:
        
        1. **Logistic Regression** - Baseline statistical model
        2. **Random Forest** - Ensemble learning for better accuracy
        3. **Support Vector Machine (SVM)** - Effective for high-dimensional data
        """)
    
    with col2:
        st.markdown("### 📊 Dataset Information")
        st.markdown(f"""
        **Total Records:** {len(st.session_state.data)}  
        **Features:** {len(st.session_state.data.columns) - 1} clinical parameters  
        **Target Variable:** Heart Disease (0 = No, 1 = Yes)
        
        **Clinical Features:**
        - Age, Sex, Chest Pain Type
        - Resting Blood Pressure, Cholesterol
        - Fasting Blood Sugar, Resting ECG
        - Max Heart Rate, Exercise Angina
        - ST Depression, Slope, CA, Thal
        """)
        
        if 'target' in st.session_state.data.columns:
            disease_count = st.session_state.data['target'].sum()
            healthy_count = len(st.session_state.data) - disease_count
            st.markdown(f"""
            **Target Distribution:**
            - ❤️ Disease: {disease_count} ({disease_count/len(st.session_state.data)*100:.1f}%)
            - 💚 Healthy: {healthy_count} ({healthy_count/len(st.session_state.data)*100:.1f}%)
            """)
    
    st.markdown("---")
    st.markdown("### 🚀 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Input Data")
        st.markdown("Enter patient clinical parameters through the sidebar")
    
    with col2:
        st.markdown("#### 2️⃣ Model Prediction")
        st.markdown("AI models analyze the data and predict risk level")
    
    with col3:
        st.markdown("#### 3️⃣ Get Results")
        st.markdown("Receive risk assessment and health recommendations")
    
    st.markdown("---")
    st.info("💡 **Tip:** Navigate to the 'Prediction' page to test the system with patient data!")

# ==================== PREDICTION PAGE ====================
elif page == "❤️ Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Risk Prediction</h1>
        <p>Enter patient information for risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Patient Clinical Data")
    
    # Create input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
    
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    
    # Convert categorical to numeric
    sex_num = 1 if sex == "Male" else 0
    cp_num = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
    fbs_num = 1 if fbs == "Yes" else 0
    restecg_num = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}[restecg]
    exang_num = 1 if exang == "Yes" else 0
    slope_num = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
    thal_num = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]
    
    # Model selection
    st.markdown("---")
    selected_model = st.selectbox(
        "Select Model for Prediction",
        ["Logistic Regression", "Random Forest", "SVM"]
    )
    
    if st.button("🔮 Predict Heart Disease Risk", use_container_width=True):
        # Create input array
        input_data = np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num, 
                                restecg_num, thalach, exang_num, oldpeak, slope_num, ca, thal_num]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        model = models[selected_model]
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
                
                # Determine risk level color
                if risk_level > 70:
                    risk_color = "🔴 High Risk"
                    risk_class = "risk-high"
                elif risk_level > 40:
                    risk_color = "🟡 Medium Risk"
                    risk_class = "risk-medium"
                else:
                    risk_color = "🟢 Low Risk"
                    risk_class = "risk-low"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h3>Risk Score</h3>
                    <h1>{risk_level:.1f}%</h1>
                    <p>{risk_color}</p>
                </div>
                """, unsafe_allow_html=True)
                
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
        
        # Explain Prediction with Feature Importance
        st.markdown("---")
        st.markdown("## 🧠 Understanding the Prediction")
        
        if selected_model == "Random Forest":
            # Get feature importance from Random Forest
            feature_importance = models['Random Forest'].feature_importances_
            
            # Create feature importance dataframe
            feature_names_list = feature_names
            importance_df = pd.DataFrame({
                'Feature': feature_names_list,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Top Influencing Features")
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title="Feature Importance",
                           color='Importance', color_continuous_scale='RdYlGn')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Key Factors")
                top_features = importance_df.head(5)
                for _, row in top_features.iterrows():
                    st.markdown(f"- **{row['Feature']}**: {row['Importance']:.2%} influence")
        
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
            
            **Medications (if prescribed):**
            - Blood pressure medications
            - Cholesterol-lowering statins
            - Antiplatelet therapy (aspirin)
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
            
            **Healthy Lifestyle Tips:**
            - Maintain healthy weight (BMI 18.5-24.9)
            - Limit processed foods and added sugars
            - Stay hydrated (8-10 glasses water daily)
            - Build strong social connections
            """)

# ==================== VISUALIZATIONS PAGE ====================
elif page == "📊 Visualizations":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Visualizations</h1>
        <p>Explore patterns and insights in heart disease data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'target' in st.session_state.data.columns:
        df_viz = st.session_state.data
        
        # Target Distribution
        st.markdown("### 🎯 Target Variable Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            target_counts = df_viz['target'].value_counts()
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
        st.markdown("### 📅 Age Distribution by Heart Disease")
        fig = px.histogram(df_viz, x='age', color='target', nbins=30,
                          title="Age Distribution by Heart Disease Status",
                          color_discrete_sequence=['#2ecc71', '#e74c3c'],
                          labels={'target': 'Heart Disease', 'age': 'Age'})
        fig.update_layout(barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
        
        # Cholesterol Distribution
        st.markdown("### 🩸 Cholesterol Levels by Heart Disease")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df_viz, x='target', y='chol', color='target',
                        title="Cholesterol Distribution",
                        color_discrete_sequence=['#2ecc71', '#e74c3c'],
                        labels={'target': 'Heart Disease', 'chol': 'Cholesterol'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.violin(df_viz, x='target', y='chol', color='target',
                           title="Cholesterol Violin Plot",
                           color_discrete_sequence=['#2ecc71', '#e74c3c'],
                           box=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        
        # Select only numeric columns
        numeric_cols = df_viz.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_viz[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlation Matrix")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional Visualizations
        st.markdown("### 📈 Additional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Max Heart Rate Distribution
            fig = px.histogram(df_viz, x='thalach', color='target', 
                              title="Maximum Heart Rate Distribution",
                              color_discrete_sequence=['#2ecc71', '#e74c3c'],
                              nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Chest Pain Type Analysis
            cp_counts = pd.crosstab(df_viz['cp'], df_viz['target'], normalize='index') * 100
            fig = px.bar(cp_counts, title="Heart Disease Rate by Chest Pain Type",
                        labels={'value': 'Percentage (%)', 'cp': 'Chest Pain Type'},
                        color_discrete_sequence=['#2ecc71', '#e74c3c'],
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("### 📊 Age vs Max Heart Rate")
        fig = px.scatter(df_viz, x='age', y='thalach', color='target',
                        title="Age vs Maximum Heart Rate",
                        color_discrete_sequence=['#2ecc71', '#e74c3c'],
                        size='chol', hover_data=['chol', 'trestbps'])
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Target column not found in dataset. Cannot generate visualizations.")

# ==================== MODEL COMPARISON PAGE ====================
elif page == "🧪 Model Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Model Performance Comparison</h1>
        <p>Evaluate and compare different machine learning models</p>
    </div>
    """, unsafe_allow_html=True)
    
    if y is not None:
        st.markdown("### 📊 Model Performance Metrics")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results]
        })
        
        # Sort by accuracy
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                comparison_df.style.format({'Accuracy': '{:.3f}'})
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
        
        # Confusion Matrices for each model
        st.markdown("### 📈 Confusion Matrices")
        
        tabs = st.tabs(list(results.keys()))
        
        for idx, (model_name, model_data) in enumerate(results.items()):
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
        
        # Cross-validation comparison
        st.markdown("### 🔄 Cross-Validation Results")
        st.info("Note: To get cross-validation scores, you can use cross_val_score from sklearn.")
        
        # Feature Importance for Random Forest
        st.markdown("### 🌟 Random Forest Feature Importance")
        
        rf_model = models['Random Forest']
        feature_importance = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
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
        
        if best_model == "Random Forest":
            st.success(f"🏆 **Recommended Model: Random Forest**\n\n"
                      f"Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)\n\n"
                      f"Random Forest performs best on this dataset due to its ability to handle "
                      f"complex interactions between features and its robustness to overfitting. "
                      f"It also provides feature importance insights.")
        elif best_model == "Logistic Regression":
            st.info(f"🏆 **Recommended Model: Logistic Regression**\n\n"
                   f"Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)\n\n"
                   f"Logistic Regression offers good performance with high interpretability. "
                   f"It's a good choice when you need to explain predictions to clinicians.")
        else:
            st.info(f"🏆 **Recommended Model: SVM**\n\n"
                   f"Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)\n\n"
                   f"SVM performs well on this dataset, especially with the RBF kernel. "
                   f"It's effective in high-dimensional spaces.")

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
        This project aims to develop a machine learning-based system for predicting heart disease 
        risk using clinical parameters. Early detection can significantly improve patient outcomes 
        and reduce healthcare costs.
        """)
        
        st.markdown("### 🛠️ Technologies Used")
        st.markdown("""
        - **Frontend:** Streamlit
        - **Machine Learning:** scikit-learn
        - **Data Processing:** Pandas, NumPy
        - **Visualization:** Matplotlib, Seaborn, Plotly
        - **Deployment:** Streamlit Cloud
        """)
        
        st.markdown("### 📊 Dataset Information")
        st.markdown("""
        The dataset contains 13 clinical features:
        - **Age:** Age in years
        - **Sex:** Male/Female
        - **CP:** Chest pain type
        - **Trestbps:** Resting blood pressure
        - **Chol:** Serum cholesterol
        - **Fbs:** Fasting blood sugar
        - **Restecg:** Resting ECG results
        - **Thalach:** Maximum heart rate
        - **Exang:** Exercise induced angina
        - **Oldpeak:** ST depression
        - **Slope:** Slope of peak exercise ST segment
        - **CA:** Number of major vessels
        - **Thal:** Thalassemia
        """)
    
    with col2:
        st.markdown("### 🤖 Machine Learning Models")
        st.markdown("""
        Three models are implemented for comparison:
        
        **1. Logistic Regression**
        - Simple and interpretable
        - Good baseline model
        - Provides probability estimates
        
        **2. Random Forest**
        - Ensemble learning method
        - Handles non-linear relationships
        - Provides feature importance
        
        **3. Support Vector Machine (SVM)**
        - Effective in high dimensions
        - Uses RBF kernel
        - Good for complex decision boundaries
        """)
        
        st.markdown("### 📈 Model Performance")
        st.markdown("""
        Models are evaluated using:
        - Accuracy
        - Confusion Matrix
        - Classification Report
        - Cross-Validation
        """)
        
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
        **Data Science Final Year Project**
        
        This system is designed as a comprehensive solution for heart disease prediction,
        demonstrating the application of machine learning in healthcare.
        """)
    
    st.markdown("---")
    st.markdown("### 📚 References")
    st.markdown("""
    - UCI Machine Learning Repository - Heart Disease Dataset
    - American Heart Association Guidelines
    - scikit-learn Documentation
    - Streamlit Documentation
    """)
    
    st.info("💡 **Note:** This system is for educational purposes only. Always consult healthcare professionals for medical advice.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>❤️ Heart Disease Prediction System | Data Science Final Year Project</p>
    <p style="font-size: 0.8rem;">Powered by Machine Learning | For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)
