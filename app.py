import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("Heart Disease Risk Predictor")
st.write("Enter patient data below to predict heart disease risk.")

st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 8px;
        right: 12px;
        font-size: 12px;
        color: gray;
        opacity: 0.6;
    }
    </style>
    <div class="watermark">Fahad · CodeAlpha ML Intern</div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type", options=[
        ("Typical Angina", 0), ("Atypical Angina", 1),
        ("Non-anginal Pain", 2), ("Asymptomatic", 3)
    ], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Resting ECG Results", options=[
        ("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)
    ], format_func=lambda x: x[0])[1]

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
        ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)
    ], format_func=lambda x: x[0])[1]
    ca = st.selectbox("Number of Major Vessels Colored (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[
        ("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)
    ], format_func=lambda x: x[0])[1]

if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]],
                               columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                        'thalach','exang','oldpeak','slope','ca','thal'])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Higher risk of heart disease detected — probability: {probability*100:.1f}%")
    else:
        st.success(f"✅ Lower risk of heart disease — probability: {probability*100:.1f}%")

    st.caption("Disclaimer: This is a machine learning demo for educational purposes, not a medical diagnosis.")
