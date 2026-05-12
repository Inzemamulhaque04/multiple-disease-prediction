import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8fafc; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    /* Result boxes */
    .result-positive {
        background: #fef2f2;
        border: 2px solid #fca5a5;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #991b1b;
        font-size: 18px;
        font-weight: 600;
    }
    .result-negative {
        background: #f0fdf4;
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #166534;
        font-size: 18px;
        font-weight: 600;
    }

    /* Input styling */
    .stNumberInput input, .stSelectbox select {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin: 20px 0 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #e2e8f0;
    }

    h1 { color: #0f172a; }
    h2, h3 { color: #1e293b; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "diabetes":   "saved_models/diabetes_model.sav",
        "heart":      "saved_models/heart_disease_model.sav",
        "parkinsons": "saved_models/parkinsons_model.sav",
    }
    for name, path in model_files.items():
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            models[name] = None
    return models

models = load_models()

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:40px;'>🏥</div>
        <div style='font-size:18px; font-weight:700; color:#f1f5f9; margin-top:8px;'>
            Disease Predictor
        </div>
        <div style='font-size:12px; color:#94a3b8; margin-top:4px;'>
            AI-Powered Health Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Home", "Diabetes", "Heart Disease", "Parkinson's"],
        icons=["house-heart", "droplet", "heart-pulse", "activity"],
        default_index=0,
        styles={
            "container":        {"padding": "8px", "background-color": "transparent"},
            "icon":             {"color": "#94a3b8", "font-size": "16px"},
            "nav-link":         {
                "font-size": "14px", "color": "#cbd5e1",
                "border-radius": "8px", "margin": "2px 0",
            },
            "nav-link-selected": {
                "background-color": "#3b82f6", "color": "white",
                "font-weight": "600",
            },
        },
    )

    st.markdown("""
    <div style='padding: 16px; background: rgba(255,255,255,0.05);
                border-radius: 10px; margin-top: 20px;'>
        <div style='font-size:11px; color:#94a3b8; line-height:1.6;'>
            ⚠️ <strong style='color:#fbbf24;'>Disclaimer</strong><br>
            This tool is for educational purposes only and is not a substitute
            for professional medical advice.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Home":
    st.markdown("<h1 style='margin-bottom:4px;'>Multiple Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:16px; margin-bottom:32px;'>An AI-powered system to predict risk of common diseases using machine learning.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='card'>
            <div style='font-size:36px; margin-bottom:12px;'>💉</div>
            <h3 style='margin:0 0 8px;'>Diabetes</h3>
            <p style='color:#64748b; font-size:14px; margin:0;'>
                Predict diabetes risk using 8 key health indicators from the PIMA Indian Diabetes dataset.
            </p>
            <div style='margin-top:16px; font-size:13px; color:#3b82f6; font-weight:600;'>
                → Select from sidebar
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <div style='font-size:36px; margin-bottom:12px;'>❤️</div>
            <h3 style='margin:0 0 8px;'>Heart Disease</h3>
            <p style='color:#64748b; font-size:14px; margin:0;'>
                Analyse 13 clinical parameters to assess the likelihood of heart disease.
            </p>
            <div style='margin-top:16px; font-size:13px; color:#3b82f6; font-weight:600;'>
                → Select from sidebar
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card'>
            <div style='font-size:36px; margin-bottom:12px;'>🧠</div>
            <h3 style='margin:0 0 8px;'>Parkinson's</h3>
            <p style='color:#64748b; font-size:14px; margin:0;'>
                Detect Parkinson's disease using 22 biomedical voice measurement features.
            </p>
            <div style='margin-top:16px; font-size:13px; color:#3b82f6; font-weight:600;'>
                → Select from sidebar
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='margin-top: 8px;'>
        <h3 style='margin-top:0;'>🤖 How it works</h3>
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:12px;'>
            <div style='text-align:center; padding:16px; background:#f8fafc; border-radius:10px;'>
                <div style='font-size:24px;'>1️⃣</div>
                <div style='font-weight:600; margin:8px 0 4px;'>Select Disease</div>
                <div style='font-size:13px; color:#64748b;'>Choose which disease to predict from the sidebar</div>
            </div>
            <div style='text-align:center; padding:16px; background:#f8fafc; border-radius:10px;'>
                <div style='font-size:24px;'>2️⃣</div>
                <div style='font-weight:600; margin:8px 0 4px;'>Enter Details</div>
                <div style='font-size:13px; color:#64748b;'>Fill in the health parameters in the form</div>
            </div>
            <div style='text-align:center; padding:16px; background:#f8fafc; border-radius:10px;'>
                <div style='font-size:24px;'>3️⃣</div>
                <div style='font-weight:600; margin:8px 0 4px;'>Get Result</div>
                <div style='font-size:13px; color:#64748b;'>Click predict to see the AI's assessment</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIABETES PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Diabetes":
    st.markdown("<h1>💉 Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Enter the patient's health information below to predict diabetes risk.</p>", unsafe_allow_html=True)

    with st.form("diabetes_form"):
        st.markdown("<div class='section-header'>Patient Information</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1,
                                          help="Number of times pregnant")
            SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                                            help="Triceps skin fold thickness")
            DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0,
                                                        max_value=3.0, value=0.5, format="%.3f",
                                                        help="Likelihood of diabetes based on family history")

        with col2:
            Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=110,
                                      help="Plasma glucose concentration")
            Insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=80,
                                      help="2-Hour serum insulin")
            Age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

        with col3:
            BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70,
                                            help="Diastolic blood pressure")
            BMI = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f",
                                  help="Body Mass Index")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

    if submitted:
        if models["diabetes"] is None:
            st.error("⚠️ Model file not found. Please train and save the model first (see notebooks/).")
        else:
            features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                   Insulin, BMI, DiabetesPedigreeFunction, Age]])
            prediction = models["diabetes"].predict(features)
            proba = models["diabetes"].predict_proba(features)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction[0] == 1:
                st.markdown(f"""
                <div class='result-positive'>
                    ⚠️ High Risk: The model predicts this person <strong>may have Diabetes</strong><br>
                    <span style='font-size:14px; font-weight:400; opacity:0.8;'>
                        Confidence: {proba[1]*100:.1f}% — Please consult a medical professional.
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-negative'>
                    ✅ Low Risk: The model predicts this person is <strong>unlikely to have Diabetes</strong><br>
                    <span style='font-size:14px; font-weight:400; opacity:0.8;'>
                        Confidence: {proba[0]*100:.1f}% — Maintain a healthy lifestyle!
                    </span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEART DISEASE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Heart Disease":
    st.markdown("<h1>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Enter clinical measurements to assess heart disease risk.</p>", unsafe_allow_html=True)

    with st.form("heart_form"):
        st.markdown("<div class='section-header'>Clinical Parameters</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            age      = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
            restecg  = st.selectbox("Resting ECG Results",
                                    options=[0, 1, 2],
                                    format_func=lambda x: {0:"Normal", 1:"ST-T wave abnormality", 2:"Left ventricular hypertrophy"}[x])
            oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
            ca       = st.number_input("Major Vessels Coloured (0–3)", min_value=0, max_value=3, value=0)

        with col2:
            sex      = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            chol     = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
            thalach  = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
            slope    = st.selectbox("Slope of Peak Exercise ST",
                                    options=[0, 1, 2],
                                    format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x])

        with col3:
            cp       = st.selectbox("Chest Pain Type",
                                    options=[0, 1, 2, 3],
                                    format_func=lambda x: {0:"Typical angina", 1:"Atypical angina",
                                                           2:"Non-anginal pain", 3:"Asymptomatic"}[x])
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL",
                                    options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            exang    = st.selectbox("Exercise Induced Angina",
                                    options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            thal     = st.selectbox("Thalassemia",
                                    options=[0, 1, 2, 3],
                                    format_func=lambda x: {0:"Normal", 1:"Fixed defect",
                                                           2:"Reversable defect", 3:"Unknown"}[x])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

    if submitted:
        if models["heart"] is None:
            st.error("⚠️ Model file not found. Please train and save the model first (see notebooks/).")
        else:
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]])
            prediction = models["heart"].predict(features)
            proba = models["heart"].predict_proba(features)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction[0] == 1:
                st.markdown(f"""
                <div class='result-positive'>
                    ⚠️ High Risk: The model predicts this person <strong>may have Heart Disease</strong><br>
                    <span style='font-size:14px; font-weight:400; opacity:0.8;'>
                        Confidence: {proba[1]*100:.1f}% — Please consult a cardiologist.
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-negative'>
                    ✅ Low Risk: The model predicts this person is <strong>unlikely to have Heart Disease</strong><br>
                    <span style='font-size:14px; font-weight:400; opacity:0.8;'>
                        Confidence: {proba[0]*100:.1f}% — Keep up the healthy habits!
                    </span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PARKINSON'S PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Parkinson's":
    st.markdown("<h1>🧠 Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Enter biomedical voice measurements to detect Parkinson's disease.</p>", unsafe_allow_html=True)

    with st.form("parkinsons_form"):
        st.markdown("<div class='section-header'>Voice Frequency Measures</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            fo        = st.number_input("MDVP:Fo(Hz) — Average vocal freq.", min_value=50.0, max_value=300.0, value=119.99, format="%.3f")
            fhi       = st.number_input("MDVP:Fhi(Hz) — Max vocal freq.",    min_value=50.0, max_value=700.0, value=157.30, format="%.3f")
            flo       = st.number_input("MDVP:Flo(Hz) — Min vocal freq.",    min_value=50.0, max_value=300.0, value=74.99,  format="%.3f")
            Jitter_percent = st.number_input("MDVP:Jitter(%)",               min_value=0.0,  max_value=2.0,   value=0.005,  format="%.5f")
            Jitter_Abs     = st.number_input("MDVP:Jitter(Abs)",             min_value=0.0,  max_value=0.1,   value=0.00004,format="%.5f")
            RAP            = st.number_input("MDVP:RAP",                     min_value=0.0,  max_value=0.1,   value=0.003,  format="%.5f")
            PPQ            = st.number_input("MDVP:PPQ",                     min_value=0.0,  max_value=0.1,   value=0.003,  format="%.5f")
            DDP            = st.number_input("Jitter:DDP",                   min_value=0.0,  max_value=0.1,   value=0.009,  format="%.5f")

        with col2:
            Shimmer        = st.number_input("MDVP:Shimmer",                 min_value=0.0,  max_value=1.0,   value=0.026,  format="%.5f")
            Shimmer_dB     = st.number_input("MDVP:Shimmer(dB)",             min_value=0.0,  max_value=5.0,   value=0.241,  format="%.3f")
            APQ3           = st.number_input("Shimmer:APQ3",                 min_value=0.0,  max_value=0.5,   value=0.013,  format="%.5f")
            APQ5           = st.number_input("Shimmer:APQ5",                 min_value=0.0,  max_value=0.5,   value=0.016,  format="%.5f")
            APQ            = st.number_input("MDVP:APQ",                     min_value=0.0,  max_value=0.5,   value=0.024,  format="%.5f")
            DDA            = st.number_input("Shimmer:DDA",                  min_value=0.0,  max_value=0.5,   value=0.040,  format="%.5f")
            NHR            = st.number_input("NHR — Noise-to-harmonic ratio",min_value=0.0,  max_value=0.5,   value=0.012,  format="%.5f")
            HNR            = st.number_input("HNR — Harmonic-to-noise ratio",min_value=0.0,  max_value=40.0,  value=21.03,  format="%.3f")

        with col3:
            RPDE           = st.number_input("RPDE",                         min_value=0.0,  max_value=1.0,   value=0.414,  format="%.6f")
            DFA            = st.number_input("DFA",                          min_value=0.0,  max_value=1.0,   value=0.815,  format="%.6f")
            spread1        = st.number_input("Spread1",                      min_value=-10.0,max_value=0.0,   value=-4.81,  format="%.6f")
            spread2        = st.number_input("Spread2",                      min_value=0.0,  max_value=1.0,   value=0.266,  format="%.6f")
            D2             = st.number_input("D2",                           min_value=0.0,  max_value=5.0,   value=2.30,   format="%.6f")
            PPE            = st.number_input("PPE",                          min_value=0.0,  max_value=1.0,   value=0.284,  format="%.6f")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Predict Parkinson's Risk")

    if submitted:
        if models["parkinsons"] is None:
            st.error("⚠️ Model file not found. Please train and save the model first (see notebooks/).")
        else:
            features = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                   Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                                   RPDE, DFA, spread1, spread2, D2, PPE]])
            prediction = models["parkinsons"].predict(features)
            proba = models["parkinsons"].predict_proba(features)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction[0] == 1:
                st.markdown(f"""
                <div class='result-positive'>
                    ⚠️ High Risk: The model predicts this person <strong>may have Parkinson's Disease</strong><br>
                    <span style='font-size:14px; font-weight:400; opacity:0.8;'>
                        Confidence: {proba[1]*100:.1f}% — Please consult a neurologist.
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-negative'>
                    ✅ Low Risk: The model predicts this person is <strong>unlikely to have Parkinson's Disease</strong><br>
                    <span style='font-size:14px; font-weight:400; opacity:0.8;'>
                        Confidence: {proba[0]*100:.1f}% — No signs detected.
                    </span>
                </div>""", unsafe_allow_html=True)
