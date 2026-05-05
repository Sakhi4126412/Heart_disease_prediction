# ==========================================
# ❤️ HEART DISEASE PREDICTION STREAMLIT APP
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ==============================
# MEDICAL THEME STYLING
# ==============================
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}
.main {
    background-color: #f4f8fb;
}
h1 {
    color: #d62828;
    text-align: center;
}
h2, h3 {
    color: #003049;
}
.stButton>button {
    background-color: #0077b6;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
.stSidebar {
    background-color: #e3f2fd;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.title("❤️ Heart Disease Prediction System")
st.markdown("### 🏥 AI-powered Clinical Risk Assessment Tool")

# ==============================
# FILE UPLOAD
# ==============================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("heart.csv")  # default dataset

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ==============================
# PREPROCESSING
# ==============================
def preprocess_data(df):
    df = df.copy()

    # Handle invalid values
    if 'Cholesterol' in df.columns:
        df['Cholesterol'] = df['Cholesterol'].replace(0, df['Cholesterol'].median())
    if 'RestingBP' in df.columns:
        df['RestingBP'] = df['RestingBP'].replace(0, df['RestingBP'].median())

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns

# ==============================
# MODEL SELECTION
# ==============================
st.sidebar.header("🤖 Select Machine Learning Model")

model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM",
        "XGBoost"
    ]
)

def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    elif name == "KNN":
        return KNeighborsClassifier()
    elif name == "Decision Tree":
        return DecisionTreeClassifier()
    elif name == "Random Forest":
        return RandomForestClassifier()
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier()
    elif name == "SVM":
        return SVC(probability=True)
    elif name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# ==============================
# TRAIN MODEL
# ==============================
X, y, scaler, feature_names = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = get_model(model_name)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.success(f"📈 Accuracy: {accuracy:.2f}")

# ==============================
# USER INPUT
# ==============================
st.header("🧾 Enter Patient Details")

col1, col2, col3 = st.columns(3)

def user_input():
    age = col1.slider("Age", 20, 80, 50)
    sex = col1.selectbox("Sex", ["M", "F"])
    cp = col1.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])

    bp = col2.slider("Resting BP", 80, 200, 120)
    chol = col2.slider("Cholesterol", 100, 600, 200)
    fbs = col2.selectbox("Fasting Blood Sugar >120", [0, 1])

    restecg = col3.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxhr = col3.slider("Max Heart Rate", 60, 200, 150)
    exang = col3.selectbox("Exercise Angina", ["Y", "N"])

    oldpeak = col2.slider("Oldpeak", -2.5, 6.0, 1.0)
    slope = col3.selectbox("ST Slope", ["Up", "Flat", "Down"])

    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": cp,
        "RestingBP": bp,
        "Cholesterol": chol,
        "FastingBS": fbs,
        "RestingECG": restecg,
        "MaxHR": maxhr,
        "ExerciseAngina": exang,
        "Oldpeak": oldpeak,
        "ST_Slope": slope
    }

    return pd.DataFrame([data])

input_df = user_input()

# ==============================
# PROCESS INPUT
# ==============================
def process_input(input_df, df, scaler):
    full_df = pd.concat([df.drop("HeartDisease", axis=1), input_df], axis=0)
    full_df = pd.get_dummies(full_df, drop_first=True)

    full_df = full_df.tail(1)
    full_df = scaler.transform(full_df)

    return full_df

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict Heart Disease"):

    processed_input = process_input(input_df, df, scaler)

    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\nProbability: {probability:.2f}")

# ==============================
# MODEL COMPARISON
# ==============================
st.header("📊 Model Accuracy Comparison")

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boost": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results[name] = accuracy_score(y_test, pred)

st.bar_chart(pd.DataFrame(results, index=["Accuracy"]).T)
