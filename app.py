"""
Cardio Guard – Heart Disease Prediction
Streamlit Web Application
Dataset: heart_statlog_cleveland_hungary_final.csv
Features: age, sex, chest_pain_type, resting_bp_s, cholesterol,
          fasting_blood_sugar, resting_ecg, max_heart_rate,
          exercise_angina, oldpeak, st_slope
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from streamlit_option_menu import option_menu

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio Guard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar: glassmorphism dark background ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1c0d0d 0%, #180818 60%, #0d1020 100%) !important;
    border-right: 1px solid rgba(239,83,80,.15) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Brand block */
.sb-brand {
    padding: 2rem 1.5rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,.07);
    text-align: center;
}
.sb-logo {
    width: 64px; height: 64px; border-radius: 50%;
    background: rgba(239,83,80,.18);
    border: 1px solid rgba(239,83,80,.35);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto .9rem;
    box-shadow: 0 0 28px rgba(239,83,80,.25), inset 0 1px 0 rgba(255,255,255,.08);
    font-size: 28px; line-height: 64px; text-align: center;
}
.sb-title {
    font-size: 1.3rem; font-weight: 700; color: #ffffff;
    letter-spacing: -.01em; margin: 0;
}
.sb-subtitle {
    font-size: .75rem; color: rgba(255,255,255,.45);
    margin: .3rem 0 0; letter-spacing: .04em; text-transform: uppercase;
}

/* Nav items */
.sb-nav { padding: 1.5rem 1rem; }
.sb-nav-label {
    font-size: .65rem; font-weight: 600; color: rgba(255,255,255,.3);
    text-transform: uppercase; letter-spacing: .1em;
    padding: 0 .5rem; margin-bottom: .6rem;
}

/* ── Nav buttons: target Streamlit's actual rendered selectors ── */
[data-testid="stSidebar"] .nav-wrap-active,
[data-testid="stSidebar"] .nav-wrap-inactive {
    padding: 0 .65rem;
    margin-bottom: .3rem !important;
}

/* Kill Streamlit's white default on ALL button variants */
[data-testid="stSidebar"] .nav-wrap-active [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] .nav-wrap-inactive [data-testid="stBaseButton-secondary"] {
    width: 100% !important;
    border-radius: 10px !important;
    font-size: .88rem !important;
    font-weight: 500 !important;
    padding: .72rem 1rem !important;
    transition: background .2s, border-color .2s, color .2s !important;
    outline: none !important;
}

/* INACTIVE — dark translucent glass */
[data-testid="stSidebar"] .nav-wrap-inactive [data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,.04) !important;
    color: rgba(255,255,255,.55) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .nav-wrap-inactive [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,.09) !important;
    color: rgba(255,255,255,.9) !important;
    border-color: rgba(255,255,255,.16) !important;
}

/* ACTIVE — red glass glow */
[data-testid="stSidebar"] .nav-wrap-active [data-testid="stBaseButton-secondary"] {
    background: rgba(239,83,80,.15) !important;
    color: #ff8a80 !important;
    border: 1px solid rgba(239,83,80,.35) !important;
    box-shadow: 0 0 18px rgba(239,83,80,.18), inset 0 1px 0 rgba(255,255,255,.06) !important;
}

/* Override the <p> inside the button */
[data-testid="stSidebar"] .nav-wrap-inactive [data-testid="stBaseButton-secondary"] p {
    color: rgba(255,255,255,.55) !important;
    font-size: .88rem !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] .nav-wrap-active [data-testid="stBaseButton-secondary"] p {
    color: #ff8a80 !important;
    font-size: .88rem !important; font-weight: 500 !important;
}

/* Status card */
.sb-status {
    margin: .9rem .75rem .6rem;
    border-radius: 14px; padding: .95rem 1.1rem;
    border: 1px solid rgba(102,187,106,.2);
    background: rgba(102,187,106,.06);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
}
.sb-status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #66bb6a; display: inline-block;
    margin-right: .5rem; box-shadow: 0 0 6px #66bb6a;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(102,187,106,.5); }
    70%  { box-shadow: 0 0 0 5px rgba(102,187,106,.0); }
    100% { box-shadow: 0 0 0 0 rgba(102,187,106,.0); }
}
.sb-status-label {
    font-size: .7rem; font-weight: 600; color: #66bb6a;
    text-transform: uppercase; letter-spacing: .08em;
}
.sb-status-model {
    font-size: .95rem; font-weight: 700; color: #ffffff;
    margin: .4rem 0 .2rem; line-height: 1.2;
}
.sb-status-feat {
    font-size: .72rem; color: rgba(255,255,255,.4);
}
.sb-status-err {
    border-color: rgba(239,83,80,.3);
    background: rgba(239,83,80,.08);
}
.sb-status-err .sb-status-label { color: #ef5350; }

/* Stats row */
.sb-stats {
    display: flex; gap: .5rem; margin: 0 1rem 1.5rem;
}
.sb-stat {
    flex: 1;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
    border-radius: 10px; padding: .6rem .4rem; text-align: center;
}
.sb-stat-val { font-size: 1rem; font-weight: 700; color: #ef5350; }
.sb-stat-lbl { font-size: .62rem; color: rgba(255,255,255,.35); margin-top: .1rem; text-transform: uppercase; letter-spacing: .05em; }

/* Footer */
.sb-footer {
    padding: 1rem 1.5rem 1.5rem;
    border-top: 1px solid rgba(255,255,255,.07);
    margin-top: auto;
}
.sb-footer-text { font-size: .7rem; color: rgba(255,255,255,.25); line-height: 1.6; }
.sb-footer-text a { color: rgba(239,83,80,.7); text-decoration: none; }
.sb-footer-text a:hover { color: #ef5350; }

/* ── Main content ── */
.main-header {
    background: linear-gradient(135deg, #b71c1c 0%, #ef5350 60%, #26c6da 100%);
    padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.5rem; color: white;
}
.main-header h1 { color: white !important; margin: 0; font-size: 2rem; }
.main-header p  { color: rgba(255,255,255,.88); margin: .4rem 0 0; }
.result-danger  { background:#ffebee; border-left:5px solid #ef5350; padding:1.2rem 1.5rem; border-radius:10px; margin-top:1rem; }
.result-warning { background:#fff8e1; border-left:5px solid #ffa726; padding:1.2rem 1.5rem; border-radius:10px; margin-top:1rem; }
.result-success { background:#e8f5e9; border-left:5px solid #66bb6a; padding:1.2rem 1.5rem; border-radius:10px; margin-top:1rem; }
.result-danger h2  { color:#c62828 !important; }
.result-warning h2 { color:#e65100 !important; }
.result-success h2 { color:#2e7d32 !important; }
.section-title {
    font-size:.9rem; font-weight:700; color:#ef5350; text-transform:uppercase;
    letter-spacing:.06em; margin:1.4rem 0 .6rem;
    padding-bottom:.35rem; border-bottom:2px solid #ef5350;
}
.disclaimer { background:#f5f5f5; border-radius:8px; padding:.8rem 1rem;
              font-size:.8rem; color:#757575; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m  = joblib.load("models/best_model.pkl")
    fn = joblib.load("models/feature_names.pkl")
    with open("models/best_model_name.txt") as f:
        name = f.read().strip()
    return m, fn, name

try:
    model, feature_names, best_model_name = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Dropdown option maps  (display label → numeric value) ────
SEX_OPTIONS    = {"Male": 1, "Female": 0}

CP_OPTIONS     = {
    "Typical Angina":     1,
    "Atypical Angina":    2,
    "Non-Anginal Pain":   3,
    "Asymptomatic":       4,
}

FBS_OPTIONS    = {"No  (≤ 120 mg/dl)": 0, "Yes  (> 120 mg/dl)": 1}

RESTECG_OPTIONS = {
    "Normal":                        0,
    "ST-T Wave Abnormality":         1,
    "Left Ventricular Hypertrophy":  2,
}

EXANG_OPTIONS  = {"No": 0, "Yes": 1}

SLOPE_OPTIONS  = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}

# ── Sidebar ───────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "Predict"

with st.sidebar:

    feat_count = len(feature_names) if model_loaded else 0
    model_name = best_model_name if model_loaded else "Not loaded"

    # Brand block
    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-logo">&#10084;&#65039;</div>'
        '<p class="sb-title">Cardio Guard</p>'
        '<p class="sb-subtitle">Heart Disease Predictor</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Nav label
    st.markdown(
        '<div style="font-size:.65rem;font-weight:600;color:rgba(255,255,255,.3);'
        'text-transform:uppercase;letter-spacing:.1em;padding:.8rem 1rem .4rem;">'
        'Navigation</div>',
        unsafe_allow_html=True,
    )

    # Nav buttons using st.button + session_state
    for label, icon in [("Predict","🔮"),("Analytics","📊"),("About","ℹ️")]:
        active = st.session_state.page == label
        wrap_cls = "nav-wrap-active" if active else "nav-wrap-inactive"
        st.markdown(f'<div class="{wrap_cls}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Status card
    if model_loaded:
        status_html = (
            '<div class="sb-status">'
            '<div><span class="sb-status-dot"></span>'
            '<span class="sb-status-label">Model Active</span></div>'
            '<div class="sb-status-model">' + model_name + '</div>'
            '<div class="sb-status-feat">' + str(feat_count) + ' features &nbsp;&middot;&nbsp; 1,190 samples</div>'
            '</div>'
        )
    else:
        status_html = (
            '<div class="sb-status sb-status-err">'
            '<div><span class="sb-status-label">&#9888; Model not found</span></div>'
            '<div class="sb-status-feat" style="margin-top:.3rem">Run train_model.py first</div>'
            '</div>'
        )
    st.markdown(status_html, unsafe_allow_html=True)

    # Stats row
    st.markdown(
        '<div class="sb-stats">'
        '<div class="sb-stat"><div class="sb-stat-val">92.4%</div><div class="sb-stat-lbl">Accuracy</div></div>'
        '<div class="sb-stat"><div class="sb-stat-val">97.3%</div><div class="sb-stat-lbl">AUC-ROC</div></div>'
        '<div class="sb-stat"><div class="sb-stat-val">6</div><div class="sb-stat-lbl">Models</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown(
        '<div class="sb-footer">'
        '<div class="sb-footer-text">'
        'Dataset: <a href="https://www.kaggle.com/datasets/mahfuzhossain/heart-statlog-cleveland-hungary-final" target="_blank">Mahfuz Hossain / Kaggle</a><br>'
        '</div></div>',
        unsafe_allow_html=True,
    )

# Assign page OUTSIDE the sidebar block so it's accessible to all page sections
page = st.session_state.page

# ══════════════════════════════════════════════════════════════
# PAGE — PREDICT
# ══════════════════════════════════════════════════════════════
if page == "Predict":

    st.markdown("""
    <div class="main-header">
      <h1>❤️ Cardio Guard</h1>
      <p>Fill in the patient's clinical details and click <strong>Predict Risk</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model not loaded. Please run `python train_model.py` first.")
        st.stop()

    with st.form("prediction_form"):

        # ── Demographics ──────────────────────────────────────
        st.markdown('<p class="section-title">👤 Demographics</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120,
                                  value=54, step=1, help="Patient's age in years")
        with c2:
            sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()),
                                     help="Biological sex of the patient")

        # ── Symptoms ──────────────────────────────────────────
        st.markdown('<p class="section-title">🫀 Symptoms & Pain</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            cp_label = st.selectbox(
                "Chest Pain Type", list(CP_OPTIONS.keys()),
                help="1=Typical Angina  2=Atypical  3=Non-Anginal  4=Asymptomatic")
        with c2:
            exang_label = st.selectbox(
                "Exercise-Induced Angina", list(EXANG_OPTIONS.keys()),
                help="Does the patient get chest pain during exercise?")
        with c3:
            oldpeak = st.number_input(
                "ST Depression (Oldpeak)", min_value=0.0, max_value=10.0,
                value=1.0, step=0.1, format="%.1f",
                help="ST depression induced by exercise relative to rest")

        # ── Vitals & Lab ──────────────────────────────────────
        st.markdown('<p class="section-title">💉 Vitals & Lab Results</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)", min_value=50, max_value=300,
                value=130, step=1, help="Resting BP measured on hospital admission")
        with c2:
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dl)", min_value=50, max_value=700,
                value=246, step=1, help="Total serum cholesterol level")
        with c3:
            fbs_label = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl", list(FBS_OPTIONS.keys()),
                help="Is fasting blood sugar above 120 mg/dl?")

        # ── ECG & Stress Test ─────────────────────────────────
        st.markdown('<p class="section-title">📈 ECG & Stress Test</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            restecg_label = st.selectbox(
                "Resting ECG Results", list(RESTECG_OPTIONS.keys()),
                help="0=Normal  1=ST-T Abnormality  2=LV Hypertrophy")
        with c2:
            max_hr = st.number_input(
                "Max Heart Rate Achieved (bpm)", min_value=50, max_value=250,
                value=150, step=1, help="Maximum heart rate during stress test")
        with c3:
            slope_label = st.selectbox(
                "Slope of Peak ST Segment", list(SLOPE_OPTIONS.keys()),
                help="1=Upsloping  2=Flat  3=Downsloping")

        st.divider()
        submitted = st.form_submit_button(
            "🔍  Predict Heart Disease Risk",
            use_container_width=True, type="primary")

    # ── Result ────────────────────────────────────────────────
    if submitted:
        # Map labels → numeric in exact model feature order
        input_values = {
            "age":                age,
            "sex":                SEX_OPTIONS[sex_label],
            "chest_pain_type":    CP_OPTIONS[cp_label],
            "resting_bp_s":       resting_bp,
            "cholesterol":        cholesterol,
            "fasting_blood_sugar":FBS_OPTIONS[fbs_label],
            "resting_ecg":        RESTECG_OPTIONS[restecg_label],
            "max_heart_rate":     max_hr,
            "exercise_angina":    EXANG_OPTIONS[exang_label],
            "oldpeak":            oldpeak,
            "st_slope":           SLOPE_OPTIONS[slope_label],
        }

        X     = np.array([input_values[f] for f in feature_names]).reshape(1, -1)
        proba = float(model.predict_proba(X)[0][1])
        pred  = int(model.predict(X)[0])
        pct   = round(proba * 100, 1)

        st.divider()
        st.subheader("📋 Prediction Result")

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Probability",  f"{pct}%")
        m2.metric("Prediction",        "Heart Disease ❗" if pred == 1 else "No Disease ✅")
        m3.metric("Model Used",        best_model_name)

        if pct >= 70:
            risk, css = "🔴 High Risk", "result-danger"
            msg = "High probability of heart disease detected. Please consult a cardiologist immediately."
        elif pct >= 40:
            risk, css = "🟡 Moderate Risk", "result-warning"
            msg = "Moderate risk detected. Consider scheduling a medical check-up soon."
        else:
            risk, css = "🟢 Low Risk", "result-success"
            msg = "Low probability of heart disease. Keep maintaining a healthy lifestyle!"

        st.markdown(f"""
        <div class="{css}">
          <h2>{risk}</h2>
          <p style="margin:.4rem 0 0">{msg}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"#### Risk Level: {pct}%")
        st.progress(proba)

        with st.expander("📋 View all entered values"):
            display_rows = [
                ("Age",                        f"{age} years"),
                ("Sex",                        sex_label),
                ("Chest Pain Type",            cp_label),
                ("Resting Blood Pressure",     f"{resting_bp} mm Hg"),
                ("Serum Cholesterol",          f"{cholesterol} mg/dl"),
                ("Fasting Blood Sugar >120",   fbs_label),
                ("Resting ECG Results",        restecg_label),
                ("Max Heart Rate Achieved",    f"{max_hr} bpm"),
                ("Exercise-Induced Angina",    exang_label),
                ("ST Depression (Oldpeak)",    f"{oldpeak}"),
                ("Slope of Peak ST Segment",   slope_label),
            ]
            st.dataframe(
                pd.DataFrame(display_rows, columns=["Parameter", "Value"]),
                use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="disclaimer">
          ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only.
          Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE — ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "Analytics":

    st.markdown("""
    <div class="main-header">
      <h1>📊 Model Analytics & EDA</h1>
      <p>Visual insights generated from the Heart Disease Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    IMAGES = [
        ("Target Class Distribution",          "static/images/target_dist.png"),
        ("Age Distribution by Diagnosis",      "static/images/age_distribution.png"),
        ("Feature Correlation Heatmap",        "static/images/correlation_heatmap.png"),
        ("Model Performance Comparison",       "static/images/model_comparison.png"),
        ("Confusion Matrix (Best Model)",      "static/images/confusion_matrix.png"),
        ("ROC Curves – All Models",            "static/images/roc_curves.png"),
        ("Feature Importances (Random Forest)","static/images/feature_importance.png"),
    ]

    if not all(os.path.exists(p) for _, p in IMAGES):
        st.warning("⚠️ Charts not found. Run `python train_model.py` to generate them.")

    # Row 1: two small side by side
    c1, c2 = st.columns(2)
    for col, (title, path) in zip([c1, c2], IMAGES[:2]):
        with col:
            st.subheader(title)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info("Not yet generated.")

    # Rest: full width
    for title, path in IMAGES[2:]:
        st.subheader(title)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Not yet generated.")
        st.divider()


# ══════════════════════════════════════════════════════════════
# PAGE — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("""
    <div class="main-header">
      <h1>ℹ️ About Cardio Guard</h1>
      <p>Heart Disease Predictor</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Objective
    Cardio Guard predicts the likelihood of heart disease from 11 clinical parameters
    using multiple ML models, with the best selected automatically by AUC-ROC score.

    ### 📦 Dataset
    - **Source:** [Heart Disease Dataset by Mahfuz Hossain](https://www.kaggle.com/datasets/mahfuzhossain/heart-statlog-cleveland-hungary-final) — Kaggle
    - **File:** heart_statlog_cleveland_hungary_final.csv
    - **Samples:** 1,190  ·  **Features:** 11  ·  **Target:** Binary

    ### 📋 Features Used
    | Feature | Description |
    |---|---|
    | age | Patient age in years |
    | sex | Biological sex (Male / Female) |
    | chest_pain_type | Type of chest pain (1–4) |
    | resting_bp_s | Resting blood pressure (mm Hg) |
    | cholesterol | Serum cholesterol (mg/dl) |
    | fasting_blood_sugar | Fasting blood sugar > 120 mg/dl |
    | resting_ecg | Resting ECG results (0–2) |
    | max_heart_rate | Max heart rate achieved (bpm) |
    | exercise_angina | Exercise-induced angina |
    | oldpeak | ST depression (exercise vs rest) |
    | st_slope | Slope of peak ST segment (1–3) |

    ### 🤖 Models Trained
    Logistic Regression · Random Forest · Gradient Boosting · SVM · KNN · Decision Tree

    ### 🛠️ Tech Stack
    Python · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · **Streamlit**

    ### 📋 Rubric Coverage
    | Component | Status |
    |---|---|
    | Proposal & Planning | ✅ README.md |
    | Data Preprocessing & EDA | ✅ 7 charts auto-generated |
    | ML Model Development | ✅ 6 models trained |
    | Evaluation & Analysis | ✅ ROC, CM, CV, Classification Report |
    | Web Interface | ✅ This Streamlit app |
    | Final Report | 📝 Write separately |
    | Presentation & Demo | 🎥 Record screen demo |
    """)
    st.divider()
    st.caption("❤️ Cardio Guard · Created by Rabail Sarwar & Huzaifa Rehman Qureshi")