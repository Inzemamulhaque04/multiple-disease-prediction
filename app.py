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
st.sidebar.info(
    "**About**\n\n"
    "This app uses Machine Learning (Random Forest) to predict the likelihood "
    "of three diseases based on key medical parameters."
)

# ── HOME ───────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏥 Multiple Disease Prediction System")
    st.markdown("### Using Machine Learning to assist early disease detection")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ❤️ Heart Disease")
        st.markdown(
            "Predicts the risk of heart disease using **6 key clinical features**: "
            "Age, Chest Pain Type, Resting Blood Pressure, Cholesterol, "
            "Max Heart Rate, and Exercise-Induced Angina."
        )
        if st.button("Go to Heart Disease →", key="btn_heart", use_container_width=True):
            st.session_state.page = "❤️ Heart Disease"
            st.rerun()

    with col2:
        st.markdown("### 🩸 Diabetes")
        st.markdown(
            "Predicts diabetes using **5 essential health metrics**: "
            "Age, BMI, Insulin Level, Blood Pressure, and Glucose Level."
        )
        if st.button("Go to Diabetes →", key="btn_diabetes", use_container_width=True):
            st.session_state.page = "🩸 Diabetes"
            st.rerun()

    with col3:
        st.markdown("### 🧠 Parkinson's Disease")
        st.markdown(
            "Detects Parkinson's disease from **6 biomedical voice features**: "
            "Age, MDVP:Fo(Hz), MDVP:Jitter(%), MDVP:Shimmer, NHR, and RPDE."
        )
        if st.button("Go to Parkinson's →", key="btn_park", use_container_width=True):
            st.session_state.page = "🧠 Parkinson's Disease"
            st.rerun()

    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** This tool is for educational purposes only. "
        "Always consult a qualified medical professional for diagnosis and treatment."
    )

# ── HEART DISEASE ──────────────────────────────────────────────────────────────
elif page == "❤️ Heart Disease":
    st.title("❤️ Heart Disease Prediction")
    st.markdown("Fill in the patient's clinical details below.")
    st.markdown("---")

    if heart_model is None:
        st.error("Model not found! Please run `python train_models.py` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=45,
            help="Patient's age in years"
        )
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Typical Angina",
                1: "1 – Atypical Angina",
                2: "2 – Non-Anginal Pain",
                3: "3 – Asymptomatic",
            }[x],
            help="Type of chest pain experienced by the patient"
        )

    with col2:
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80, max_value=220, value=120,
            help="Resting blood pressure measured on admission"
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100, max_value=600, value=200,
            help="Serum cholesterol in mg/dl"
        )

    with col3:
        thalach = st.number_input(
            "Max Heart Rate Achieved (bpm)",
            min_value=60, max_value=250, value=150,
            help="Maximum heart rate achieved during exercise"
        )
        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether exercise induced angina (chest pain)"
        )

    st.markdown("---")

    if st.button("🔍 Predict Heart Disease", use_container_width=True):
        # Feature order: Age, ChestPain, RestingBP, Cholesterol, MaxHR, ExerciseAngina
        features   = np.array([[age, cp, trestbps, chol, thalach, exang]])
        prediction = heart_model.predict(features)[0]
        proba      = heart_model.predict_proba(features)[0]

        if prediction == 1:
            st.error(f"🚨 **High risk of Heart Disease detected!**  (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ **Low risk of Heart Disease.**  (Confidence: {proba[0]*100:.1f}%)")

        st.info("💡 Please consult a cardiologist for professional medical advice.")

# ── DIABETES ───────────────────────────────────────────────────────────────────
elif page == "🩸 Diabetes":
    st.title("🩸 Diabetes Prediction")
    st.markdown("Fill in the patient's health details below.")
    st.markdown("---")

    if diabetes_model is None:
        st.error("Model not found! Please run `python train_models.py` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=33,
            help="Patient's age in years"
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Body Mass Index"
        )

    with col2:
        insulin = st.number_input(
            "Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80,
            help="2-hour serum insulin level"
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70,
            help="Diastolic blood pressure"
        )

    with col3:
        glucose = st.number_input(
            "Glucose Level (mg/dL)", min_value=0, max_value=300, value=120,
            help="Plasma glucose concentration (2-hour oral glucose tolerance test)"
        )

    st.markdown("---")

    if st.button("🔍 Predict Diabetes", use_container_width=True):
        # Feature order: Age, BMI, Insulin, BloodPressure, Glucose
        features   = np.array([[age, bmi, insulin, blood_pressure, glucose]])
        prediction = diabetes_model.predict(features)[0]
        proba      = diabetes_model.predict_proba(features)[0]

        if prediction == 1:
            st.error(f"🚨 **The patient is likely Diabetic!**  (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ **The patient is likely not Diabetic.**  (Confidence: {proba[0]*100:.1f}%)")

        st.info("💡 Please consult an endocrinologist for professional medical advice.")

# ── PARKINSON'S DISEASE ────────────────────────────────────────────────────────
elif page == "🧠 Parkinson's Disease":
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("Fill in the patient's voice measurement features below.")
    st.markdown("---")

    if parkinsons_model is None:
        st.error("Model not found! Please run `python train_models.py` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=60,
            help="Patient's age in years"
        )
        fo = st.number_input(
            "MDVP:Fo (Hz)", min_value=50.0, max_value=300.0,
            value=119.99, step=0.01, format="%.2f",
            help="Average vocal fundamental frequency"
        )

    with col2:
        jitter_percent = st.number_input(
            "MDVP:Jitter (%)", min_value=0.0, max_value=2.0,
            value=0.00784, step=0.00001, format="%.5f",
            help="Percentage measure of variation in fundamental frequency"
        )
        shimmer = st.number_input(
            "MDVP:Shimmer", min_value=0.0, max_value=1.0,
            value=0.04374, step=0.00001, format="%.5f",
            help="Measure of variation in amplitude"
        )

    with col3:
        nhr = st.number_input(
            "NHR (Noise-to-Harmonics Ratio)", min_value=0.0, max_value=0.5,
            value=0.02211, step=0.00001, format="%.5f",
            help="Ratio of noise to tonal components in the voice"
        )
        rpde = st.number_input(
            "RPDE (Recurrence Period Density Entropy)", min_value=0.0, max_value=1.0,
            value=0.414783, step=0.000001, format="%.6f",
            help="Nonlinear dynamical complexity measure"
        )

    st.markdown("---")

    if st.button("🔍 Predict Parkinson's Disease", use_container_width=True):
        # Feature order: Age, Fo, Jitter(%), Shimmer, NHR, RPDE
        features   = np.array([[age, fo, jitter_percent, shimmer, nhr, rpde]])
        prediction = parkinsons_model.predict(features)[0]
        proba      = parkinsons_model.predict_proba(features)[0]

        if prediction == 1:
            st.error(f"🚨 **Signs of Parkinson's Disease detected!**  (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ **No signs of Parkinson's Disease detected.**  (Confidence: {proba[0]*100:.1f}%)")

        st.info("💡 Please consult a neurologist for professional medical advice.")
