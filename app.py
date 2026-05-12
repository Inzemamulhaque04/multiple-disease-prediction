import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Multiple Disease Prediction", page_icon="🏥", layout="wide")

MODEL_DIR = "models"

@st.cache_resource
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

heart_model      = load_model("heart_model.pkl")
diabetes_model   = load_model("diabetes_model.pkl")
parkinsons_model = load_model("parkinsons_model.pkl")

st.sidebar.title("🏥 Disease Prediction")
st.sidebar.markdown("---")

# ── Session state for navigation ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

PAGES = ["🏠 Home", "❤️ Heart Disease", "🩸 Diabetes", "🧠 Parkinson's Disease"]
page = st.sidebar.radio(
    "Select a disease to predict:",
    PAGES,
    index=PAGES.index(st.session_state.page),
)
st.session_state.page = page

st.sidebar.markdown("---")
st.sidebar.info("**About**\n\nThis app uses Machine Learning (Random Forest) to predict the likelihood of three diseases based on medical parameters.")

# HOME
if page == "🏠 Home":
    st.title("🏥 Multiple Disease Prediction System")
    st.markdown("### Using Machine Learning to assist early disease detection")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### ❤️ Heart Disease")
        st.markdown("Predicts the risk of heart disease based on 13 clinical features such as age, cholesterol, blood pressure, and ECG results.")
        if st.button("Go to Heart Disease →", key="btn_heart", use_container_width=True):
            st.session_state.page = "❤️ Heart Disease"
            st.rerun()
    with col2:
        st.markdown("### 🩸 Diabetes")
        st.markdown("Predicts diabetes using 8 features from the PIMA Indians dataset including glucose level, BMI, and insulin.")
        if st.button("Go to Diabetes →", key="btn_diabetes", use_container_width=True):
            st.session_state.page = "🩸 Diabetes"
            st.rerun()
    with col3:
        st.markdown("### 🧠 Parkinson's Disease")
        st.markdown("Detects Parkinson's disease from 22 biomedical voice measurement features such as jitter, shimmer, and HNR.")
        if st.button("Go to Parkinson's →", key="btn_park", use_container_width=True):
            st.session_state.page = "🧠 Parkinson's Disease"
            st.rerun()
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a qualified medical professional for diagnosis and treatment.")

# HEART
elif page == "❤️ Heart Disease":
    st.title("❤️ Heart Disease Prediction")
    st.markdown("Fill in the patient's clinical details below.")
    st.markdown("---")
    if heart_model is None:
        st.error("Model not found! Please run `python notebooks/train_models.py` first.")
        st.stop()
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp  = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3], help="0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 120)
        chol     = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    with col2:
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
        exang   = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col3:
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope   = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
        ca      = st.selectbox("Major Vessels Colored by Fluoroscopy (0–3)", [0, 1, 2, 3])
        thal    = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])
    st.markdown("---")
    if st.button("🔍 Predict Heart Disease", use_container_width=True):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(features)[0]
        proba      = heart_model.predict_proba(features)[0]
        if prediction == 1:
            st.error(f"🚨 **High risk of Heart Disease detected!** (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ **Low risk of Heart Disease.** (Confidence: {proba[0]*100:.1f}%)")
        st.info("💡 Please consult a cardiologist for professional medical advice.")

# DIABETES
elif page == "🩸 Diabetes":
    st.title("🩸 Diabetes Prediction")
    st.markdown("Fill in the patient's health details below.")
    st.markdown("---")
    if diabetes_model is None:
        st.error("Model not found! Please run `python notebooks/train_models.py` first.")
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        pregnancies    = st.number_input("Number of Pregnancies", 0, 20, 1)
        glucose        = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80)
        bmi     = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age     = st.number_input("Age", 1, 120, 33)
    st.markdown("---")
    if st.button("🔍 Predict Diabetes", use_container_width=True):
        features   = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(features)[0]
        proba      = diabetes_model.predict_proba(features)[0]
        if prediction == 1:
            st.error(f"🚨 **The patient is likely Diabetic!** (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ **The patient is likely not Diabetic.** (Confidence: {proba[0]*100:.1f}%)")
        st.info("💡 Please consult an endocrinologist for professional medical advice.")

# PARKINSONS
elif page == "🧠 Parkinson's Disease":
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("Fill in the patient's voice measurement features below.")
    st.markdown("---")
    if parkinsons_model is None:
        st.error("Model not found! Please run `python notebooks/train_models.py` first.")
        st.stop()
    col1, col2, col3 = st.columns(3)
    with col1:
        fo             = st.number_input("MDVP:Fo (Hz)", 50.0, 300.0, 119.99, step=0.01)
        fhi            = st.number_input("MDVP:Fhi (Hz)", 50.0, 600.0, 157.30, step=0.01)
        flo            = st.number_input("MDVP:Flo (Hz)", 50.0, 300.0, 74.99,  step=0.01)
        jitter_percent = st.number_input("MDVP:Jitter (%)", 0.0, 2.0, 0.00784, step=0.00001, format="%.5f")
        jitter_abs     = st.number_input("MDVP:Jitter (Abs)", 0.0, 0.001, 0.00007, step=0.000001, format="%.6f")
        rap            = st.number_input("MDVP:RAP", 0.0, 0.05, 0.00370, step=0.00001, format="%.5f")
        ppq            = st.number_input("MDVP:PPQ", 0.0, 0.05, 0.00554, step=0.00001, format="%.5f")
        ddp            = st.number_input("Jitter:DDP", 0.0, 0.1, 0.01109, step=0.00001, format="%.5f")
    with col2:
        shimmer    = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.04374, step=0.00001, format="%.5f")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 3.0, 0.426, step=0.001, format="%.3f")
        apq3       = st.number_input("Shimmer:APQ3", 0.0, 0.1, 0.02182, step=0.00001, format="%.5f")
        apq5       = st.number_input("Shimmer:APQ5", 0.0, 0.2, 0.03130, step=0.00001, format="%.5f")
        apq        = st.number_input("MDVP:APQ", 0.0, 0.2, 0.02971, step=0.00001, format="%.5f")
        dda        = st.number_input("Shimmer:DDA", 0.0, 0.5, 0.06545, step=0.00001, format="%.5f")
    with col3:
        nhr     = st.number_input("NHR", 0.0, 0.5, 0.02211, step=0.00001, format="%.5f")
        hnr     = st.number_input("HNR", 0.0, 40.0, 21.033, step=0.001, format="%.3f")
        rpde    = st.number_input("RPDE", 0.0, 1.0, 0.414783, step=0.000001, format="%.6f")
        dfa     = st.number_input("DFA", 0.0, 1.0, 0.815285, step=0.000001, format="%.6f")
        spread1 = st.number_input("Spread1", -10.0, 0.0, -4.813031, step=0.000001, format="%.6f")
        spread2 = st.number_input("Spread2", 0.0, 1.0, 0.266482, step=0.000001, format="%.6f")
        d2      = st.number_input("D2", 0.0, 5.0, 2.301442, step=0.000001, format="%.6f")
        ppe     = st.number_input("PPE", 0.0, 1.0, 0.284654, step=0.000001, format="%.6f")
    st.markdown("---")
    if st.button("🔍 Predict Parkinson's Disease", use_container_width=True):
        features = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                               shimmer, shimmer_db, apq3, apq5, apq, dda,
                               nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        prediction = parkinsons_model.predict(features)[0]
        proba      = parkinsons_model.predict_proba(features)[0]
        if prediction == 1:
            st.error(f"🚨 **Signs of Parkinson's Disease detected!** (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ **No signs of Parkinson's Disease detected.** (Confidence: {proba[0]*100:.1f}%)")
        st.info("💡 Please consult a neurologist for professional medical advice.")
