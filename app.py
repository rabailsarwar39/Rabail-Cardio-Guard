"""
Cardio Guard – Heart Disease Prediction
Streamlit Web Application  (v4 — Ensemble + XAI + HTML Report)
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Cardio Guard", page_icon="❤️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1c0d0d 0%, #180818 60%, #0d1020 100%) !important;
    border-right: 1px solid rgba(239,83,80,.15) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

.sb-brand { padding: 2rem 1.5rem 1.2rem; border-bottom: 1px solid rgba(255,255,255,.06); text-align: center; }
.sb-logo {
    width: 64px; height: 64px; border-radius: 50%;
    background: rgba(239,83,80,.18); border: 1px solid rgba(239,83,80,.35);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto .9rem; font-size: 28px; line-height: 64px; text-align: center;
    box-shadow: 0 0 28px rgba(239,83,80,.25), inset 0 1px 0 rgba(255,255,255,.08);
}
.sb-title { font-size: 1.2rem; font-weight: 700; color: #ffffff; letter-spacing: -.01em; margin: 0; }
.sb-subtitle { font-size: .68rem; color: rgba(255,255,255,.35); margin: .3rem 0 0; letter-spacing: .1em; text-transform: uppercase; }
.sb-nav-label { font-size: .62rem; font-weight: 600; color: rgba(255,255,255,.25);
    text-transform: uppercase; letter-spacing: .12em; padding: .9rem 1.1rem .45rem; }

[data-testid="stSidebar"] .nav-wrap-active,
[data-testid="stSidebar"] .nav-wrap-inactive { padding: 0 .65rem; margin-bottom: .3rem !important; }

[data-testid="stSidebar"] .nav-wrap-active [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] .nav-wrap-inactive [data-testid="stBaseButton-secondary"] {
    width: 100% !important; border-radius: 10px !important; font-size: .88rem !important;
    font-weight: 500 !important; padding: .72rem 1rem !important;
    transition: background .2s, border-color .2s, color .2s !important; outline: none !important;
}
[data-testid="stSidebar"] .nav-wrap-inactive [data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,.04) !important; color: rgba(255,255,255,.55) !important;
    border: 1px solid rgba(255,255,255,.09) !important; box-shadow: none !important;
}
[data-testid="stSidebar"] .nav-wrap-inactive [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,.09) !important; color: rgba(255,255,255,.9) !important;
    border-color: rgba(255,255,255,.16) !important;
}
[data-testid="stSidebar"] .nav-wrap-active [data-testid="stBaseButton-secondary"] {
    background: rgba(239,83,80,.15) !important; color: #ff8a80 !important;
    border: 1px solid rgba(239,83,80,.35) !important;
    box-shadow: 0 0 18px rgba(239,83,80,.18), inset 0 1px 0 rgba(255,255,255,.06) !important;
}
[data-testid="stSidebar"] .nav-wrap-inactive [data-testid="stBaseButton-secondary"] p {
    color: rgba(255,255,255,.55) !important; font-size: .88rem !important; font-weight: 500 !important; }
[data-testid="stSidebar"] .nav-wrap-active [data-testid="stBaseButton-secondary"] p {
    color: #ff8a80 !important; font-size: .88rem !important; font-weight: 500 !important; }

.sb-status { margin: .9rem .75rem .6rem; border-radius: 14px; padding: .95rem 1.1rem;
    border: 1px solid rgba(102,187,106,.2); background: rgba(102,187,106,.06);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.05); }
.sb-status-dot { width: 7px; height: 7px; border-radius: 50%; background: #66bb6a;
    display: inline-block; margin-right: .45rem; animation: pulse 2s infinite; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(102,187,106,.5); }
    70%  { box-shadow: 0 0 0 5px rgba(102,187,106,.0); }
    100% { box-shadow: 0 0 0 0 rgba(102,187,106,.0); } }
.sb-status-label { font-size: .65rem; font-weight: 700; color: #81c784;
    text-transform: uppercase; letter-spacing: .1em; }
.sb-status-model { font-size: .95rem; font-weight: 700; color: #ffffff; margin: .4rem 0 .15rem; }
.sb-status-feat  { font-size: .7rem; color: rgba(255,255,255,.35); }
.sb-status-err   { border-color: rgba(239,83,80,.25); background: rgba(239,83,80,.06); }
.sb-status-err .sb-status-label { color: #ef9a9a; }

.sb-stats { display: flex; gap: .45rem; margin: 0 .75rem .9rem; }
.sb-stat { flex: 1; background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04); border-radius: 10px; padding: .6rem .4rem; text-align: center; }
.sb-stat-val { font-size: .95rem; font-weight: 700; color: #ef9090; }
.sb-stat-lbl { font-size: .58rem; color: rgba(255,255,255,.28); margin-top: .1rem;
    text-transform: uppercase; letter-spacing: .06em; }

.sb-footer { padding: .9rem 1.2rem 1.4rem; border-top: 1px solid rgba(255,255,255,.06); }
.sb-footer-text { font-size: .68rem; color: rgba(255,255,255,.22); line-height: 1.7; }
.sb-footer-text a { color: rgba(239,83,80,.6); text-decoration: none; }

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
.section-title { font-size:.9rem; font-weight:700; color:#ef5350; text-transform:uppercase;
    letter-spacing:.06em; margin:1.4rem 0 .6rem; padding-bottom:.35rem; border-bottom:2px solid #ef5350; }
.disclaimer { background:#f5f5f5; border-radius:8px; padding:.8rem 1rem;
    font-size:.8rem; color:#757575; margin-top:1rem; }

/* ── Per-prediction XAI styles ── */
.xai-section { background:#fff; border:1px solid #e0e0e0; border-radius:14px;
    padding:1.5rem; margin-top:1.5rem; }
.xai-title { font-size:1rem; font-weight:700; color:#212121; margin-bottom:1rem; }
.xai-row { display:flex; align-items:center; gap:.75rem; margin-bottom:.55rem; }
.xai-label { font-size:.82rem; font-weight:500; color:#444; width:200px; flex-shrink:0;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.xai-bar-wrap { flex:1; background:#f5f5f5; border-radius:6px; height:22px;
    overflow:hidden; position:relative; }
.xai-bar-pos { height:100%; border-radius:6px;
    background:linear-gradient(90deg,#ef9a9a,#ef5350); }
.xai-bar-neg { height:100%; border-radius:6px;
    background:linear-gradient(90deg,#26c6da,#0097a7); }
.xai-val { font-size:.78rem; font-weight:600; width:55px; text-align:right; flex-shrink:0; }
.xai-val-pos { color:#c62828; }
.xai-val-neg { color:#00838f; }
.xai-legend { display:flex; gap:1.5rem; font-size:.75rem; color:#757575; margin-top:.75rem; }
.xai-legend span { display:flex; align-items:center; gap:.35rem; }
.xai-dot-pos { width:10px;height:10px;border-radius:50%;background:#ef5350;display:inline-block; }
.xai-dot-neg { width:10px;height:10px;border-radius:50%;background:#26c6da;display:inline-block; }
.ens-badge { background:#ffcdd2; color:#b71c1c; font-size:.72rem; font-weight:700;
    padding:2px 8px; border-radius:5px; margin-left:6px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    raw = joblib.load("models/best_model.pkl")
    fn  = joblib.load("models/feature_names.pkl")
    with open("models/best_model_name.txt") as f:
        name = f.read().strip()
    # Load SHAP explainer and scaler for per-prediction XAI
    try:
        explainer = joblib.load("models/shap_explainer.pkl")
        scaler    = joblib.load("models/scaler.pkl")
    except FileNotFoundError:
        explainer = None
        scaler    = None
    if isinstance(raw, dict):
        return raw["clf"], raw["scaler"], fn, name, True, explainer, scaler
    else:
        return raw, None, fn, name, False, explainer, scaler

try:
    model, ens_scaler, feature_names, best_model_name, is_ensemble, shap_explainer, rf_scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    shap_explainer = None
    rf_scaler      = None

# ── Option maps ───────────────────────────────────────────────
SEX_OPTIONS     = {"Male": 1, "Female": 0}
CP_OPTIONS      = {"Typical Angina":1,"Atypical Angina":2,"Non-Anginal Pain":3,"Asymptomatic":4}
FBS_OPTIONS     = {"No  (≤ 120 mg/dl)":0, "Yes  (> 120 mg/dl)":1}
RESTECG_OPTIONS = {"Normal":0, "ST-T Wave Abnormality":1, "Left Ventricular Hypertrophy":2}
EXANG_OPTIONS   = {"No":0, "Yes":1}
SLOPE_OPTIONS   = {"Upsloping":1, "Flat":2, "Downsloping":3}

# ── Sidebar ───────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Predict"

with st.sidebar:
    feat_count = len(feature_names) if model_loaded else 0
    model_name = best_model_name   if model_loaded else "Not loaded"

    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-logo">&#10084;&#65039;</div>'
        '<p class="sb-title">Cardio Guard</p>'
        '<p class="sb-subtitle">Heart Disease Predictor</p>'
        '</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)

    for label, icon in [("Predict","🔮"),("Analytics","📊"),("XAI","🔍"),("About","ℹ️")]:
        active   = st.session_state.page == label
        wrap_cls = "nav-wrap-active" if active else "nav-wrap-inactive"
        st.markdown(f'<div class="{wrap_cls}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if model_loaded:
        ens_tag = ' <span class="ens-badge">Ensemble</span>' if "Ensemble" in model_name else ""
        status_html = (
            '<div class="sb-status">'
            '<div><span class="sb-status-dot"></span>'
            '<span class="sb-status-label">Model Active</span></div>'
            '<div class="sb-status-model">' + model_name + ens_tag + '</div>'
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

    st.markdown(
        '<div class="sb-stats">'
        '<div class="sb-stat"><div class="sb-stat-val">92.4%</div><div class="sb-stat-lbl">Accuracy</div></div>'
        '<div class="sb-stat"><div class="sb-stat-val">97.3%</div><div class="sb-stat-lbl">AUC-ROC</div></div>'
        '<div class="sb-stat"><div class="sb-stat-val">8</div><div class="sb-stat-lbl">Models</div></div>'
        '</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-footer"><div class="sb-footer-text">'
        'Dataset: <a href="https://www.kaggle.com/datasets/mahfuzhossain/heart-statlog-cleveland-hungary-final" target="_blank">Mahfuz Hossain / Kaggle</a><br>'
        'BSCS ML Project &nbsp;&middot;&nbsp; 2025'
        '</div></div>', unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
# PAGE — PREDICT
# ══════════════════════════════════════════════════════════════
if page == "Predict":
    st.markdown("""
    <div class="main-header">
      <h1>❤️ Cardio Guard</h1>
      <p>Fill in the patient's clinical details and click <strong>Predict Risk</strong>.</p>
    </div>""", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model not loaded. Please run `python train_model.py` first.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown('<p class="section-title">👤 Demographics</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: age = st.number_input("Age (years)", 1, 120, 54, 1, help="Patient's age")
        with c2: sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()))

        st.markdown('<p class="section-title">🫀 Symptoms & Pain</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: cp_label    = st.selectbox("Chest Pain Type", list(CP_OPTIONS.keys()))
        with c2: exang_label = st.selectbox("Exercise-Induced Angina", list(EXANG_OPTIONS.keys()))
        with c3: oldpeak     = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1, format="%.1f")

        st.markdown('<p class="section-title">💉 Vitals & Lab Results</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: resting_bp  = st.number_input("Resting Blood Pressure (mm Hg)", 50, 300, 130, 1)
        with c2: cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 50, 700, 246, 1)
        with c3: fbs_label   = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(FBS_OPTIONS.keys()))

        st.markdown('<p class="section-title">📈 ECG & Stress Test</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: restecg_label = st.selectbox("Resting ECG Results", list(RESTECG_OPTIONS.keys()))
        with c2: max_hr        = st.number_input("Max Heart Rate Achieved (bpm)", 50, 250, 150, 1)
        with c3: slope_label   = st.selectbox("Slope of Peak ST Segment", list(SLOPE_OPTIONS.keys()))

        st.divider()
        submitted = st.form_submit_button("🔍  Predict Heart Disease Risk",
                                          use_container_width=True, type="primary")

    if submitted:
        input_values = {
            "age": age, "sex": SEX_OPTIONS[sex_label],
            "chest_pain_type": CP_OPTIONS[cp_label],
            "resting_bp_s": resting_bp, "cholesterol": cholesterol,
            "fasting_blood_sugar": FBS_OPTIONS[fbs_label],
            "resting_ecg": RESTECG_OPTIONS[restecg_label],
            "max_heart_rate": max_hr, "exercise_angina": EXANG_OPTIONS[exang_label],
            "oldpeak": oldpeak, "st_slope": SLOPE_OPTIONS[slope_label],
        }
        X_arr = np.array([input_values[f] for f in feature_names]).reshape(1, -1)
        if is_ensemble and ens_scaler:
            X_arr = ens_scaler.transform(X_arr)
        proba = float(model.predict_proba(X_arr)[0][1])
        pred  = int(model.predict(X_arr)[0])
        pct   = round(proba * 100, 1)

        st.divider(); st.subheader("📋 Prediction Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Probability", f"{pct}%")
        m2.metric("Prediction", "Heart Disease ❗" if pred == 1 else "No Disease ✅")
        m3.metric("Model", best_model_name)

        if pct >= 70:   risk, css, msg = "🔴 High Risk",     "result-danger",  "High probability of heart disease detected. Please consult a cardiologist immediately."
        elif pct >= 40: risk, css, msg = "🟡 Moderate Risk", "result-warning", "Moderate risk detected. Consider scheduling a medical check-up soon."
        else:           risk, css, msg = "🟢 Low Risk",      "result-success", "Low probability of heart disease. Keep maintaining a healthy lifestyle!"

        st.markdown(f'<div class="{css}"><h2>{risk}</h2><p style="margin:.4rem 0 0">{msg}</p></div>',
                    unsafe_allow_html=True)
        st.markdown(f"#### Risk Level: {pct}%"); st.progress(proba)

        with st.expander("📋 View all entered values"):
            rows = [("Age",f"{age} years"),("Sex",sex_label),("Chest Pain Type",cp_label),
                    ("Resting BP",f"{resting_bp} mm Hg"),("Cholesterol",f"{cholesterol} mg/dl"),
                    ("Fasting Blood Sugar >120",fbs_label),("Resting ECG",restecg_label),
                    ("Max Heart Rate",f"{max_hr} bpm"),("Exercise Angina",exang_label),
                    ("ST Depression",f"{oldpeak}"),("ST Slope",slope_label)]
            st.dataframe(pd.DataFrame(rows, columns=["Parameter","Value"]),
                         use_container_width=True, hide_index=True)

        # ── Per-prediction SHAP XAI ──────────────────────
        if shap_explainer is not None and rf_scaler is not None:
            st.divider()
            st.subheader("🔍 Why this prediction? — Explainable AI (SHAP)")
            st.caption("SHAP values show how much each feature pushed the prediction "
                       "toward Heart Disease (🔴 red) or away from it (🔵 blue).")
            try:
                X_shap = rf_scaler.transform(
                    np.array([input_values[f] for f in feature_names]).reshape(1, -1))
                sv   = shap_explainer.shap_values(X_shap)
                sv_arr = np.array(sv)
                # Handle all SHAP output shapes robustly
                if sv_arr.ndim == 3:
                    # (n_samples, n_features, n_classes) — take class 1
                    vals = sv_arr[0, :, 1]
                elif isinstance(sv, list):
                    # list of [class0_array, class1_array]
                    vals = np.array(sv[1])[0]
                else:
                    vals = sv_arr[0]

                FEAT_LABELS = {
                    "age":"Age","sex":"Sex","chest_pain_type":"Chest Pain Type",
                    "resting_bp_s":"Resting Blood Pressure","cholesterol":"Serum Cholesterol",
                    "fasting_blood_sugar":"Fasting Blood Sugar","resting_ecg":"Resting ECG",
                    "max_heart_rate":"Max Heart Rate","exercise_angina":"Exercise Angina",
                    "oldpeak":"ST Depression (Oldpeak)","st_slope":"ST Slope",
                }
                feat_labels = [FEAT_LABELS.get(f, f) for f in feature_names]
                order   = np.argsort(np.abs(vals))[::-1]
                max_abs = max(np.abs(vals)) if max(np.abs(vals)) > 0 else 1.0

                bars_html = '<div class="xai-section"><div class="xai-title">Feature Contributions to This Prediction</div>'
                for i in order:
                    v       = float(vals[i])
                    label   = feat_labels[i]
                    pct_w   = min(int(abs(v) / max_abs * 100), 100)
                    val_cls = "xai-val-pos" if v > 0 else "xai-val-neg"
                    bar_cls = "xai-bar-pos" if v > 0 else "xai-bar-neg"
                    bars_html += (
                        f'<div class="xai-row">'
                        f'<div class="xai-label" title="{label}">{label}</div>'
                        f'<div class="xai-bar-wrap"><div class="{bar_cls}" style="width:{pct_w}%"></div></div>'
                        f'<div class="xai-val {val_cls}">{v:+.3f}</div>'
                        f'</div>'
                    )
                bars_html += (
                    '<div class="xai-legend">'
                    '<span><span class="xai-dot-pos"></span> Increases heart disease risk</span>'
                    '<span><span class="xai-dot-neg"></span> Reduces heart disease risk</span>'
                    '</div></div>'
                )
                st.markdown(bars_html, unsafe_allow_html=True)

                st.markdown("#### 📝 In Plain English — Top 3 Reasons")
                input_vals_list = [input_values[f] for f in feature_names]
                for i in order[:3]:
                    v         = float(vals[i])
                    label     = feat_labels[i]
                    entered   = input_vals_list[i]
                    direction = "increased" if v > 0 else "reduced"
                    strength  = "strongly" if abs(v) > 0.1 else "slightly"
                    st.markdown(
                        f"- **{label}** (entered: `{entered}`) {strength} **{direction}** "
                        f"the heart disease risk `({v:+.3f})`"
                    )
            except Exception as e:
                st.info(f"SHAP explanation unavailable for this input: {e}")

        st.markdown('''<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This tool is for
        educational purposes only. Always consult a qualified healthcare professional.</div>''',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown("""
    <div class="main-header">
      <h1>📊 Model Analytics & EDA</h1>
      <p>Visual insights from the Heart Disease Dataset · includes Voting & Stacking Ensemble comparison</p>
    </div>""", unsafe_allow_html=True)

    # HTML Report download
    report_path = "static/report.html"
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button("📥 Download Full HTML Report", f, "cardio_guard_report.html",
                               "text/html", use_container_width=False)
        st.divider()

    IMAGES = [
        ("Target Class Distribution",          "static/images/target_dist.png"),
        ("Age Distribution by Diagnosis",      "static/images/age_distribution.png"),
        ("Feature Correlation Heatmap",        "static/images/correlation_heatmap.png"),
        ("Model Comparison (incl. Ensemble)",  "static/images/model_comparison.png"),
        ("Confusion Matrix (Best Model)",      "static/images/confusion_matrix.png"),
        ("ROC Curves – All Models",            "static/images/roc_curves.png"),
        ("Feature Importances (Random Forest)","static/images/feature_importance.png"),
    ]

    if not all(os.path.exists(p) for _, p in IMAGES):
        st.warning("⚠️ Some charts missing. Run `python train_model.py` to regenerate.")

    # Row 1: two small charts side by side
    c1, c2 = st.columns(2)
    for col, (title, path) in zip([c1, c2], IMAGES[:2]):
        with col:
            st.subheader(title)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info("Not yet generated.")

    # Remaining charts: full width with divider
    for title, path in IMAGES[2:]:
        st.subheader(title)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Not yet generated.")
        st.divider()

# ══════════════════════════════════════════════════════════════
# PAGE — XAI (Explainable AI)
# ══════════════════════════════════════════════════════════════
elif page == "XAI":
    st.markdown("""
    <div class="main-header">
      <h1>🔍 Explainable AI (XAI)</h1>
      <p>SHAP values reveal <em>why</em> the model makes each prediction — which features matter most</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    ### What is XAI?
    **Explainable AI (XAI)** makes black-box ML models transparent. Instead of just getting a
    prediction, you can see *which features pushed the prediction toward Heart Disease or No Disease*
    and by how much.

    We use **SHAP (SHapley Additive exPlanations)** — a mathematically grounded method based on
    game theory that assigns each feature a contribution score for each prediction.
    """)

    st.divider()

    shap_bar   = "static/images/shap_summary.png"
    shap_swarm = "static/images/shap_beeswarm.png"

    if not os.path.exists(shap_bar):
        st.warning("⚠️ SHAP charts not found. Run `python train_model.py` to generate them.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 SHAP Feature Importance (Bar)")
            st.image(shap_bar, use_container_width=True)
            st.caption("Average absolute SHAP value per feature — higher = more influential overall.")

        with c2:
            st.subheader("🐝 SHAP Beeswarm Plot")
            st.image(shap_swarm, use_container_width=True)
            st.caption("Each dot = one patient. Red = high feature value, Blue = low. "
                       "Position on X = impact on model output.")

        st.divider()
        st.subheader("📖 How to Read the Beeswarm Plot")
        col1, col2, col3 = st.columns(3)
        col1.info("**X-axis position**\nFar right = pushed toward Heart Disease prediction\nFar left = pushed toward No Disease")
        col2.info("**Dot colour**\nRed = high value of that feature\nBlue = low value of that feature")
        col3.info("**Feature order**\nTop features have the most impact on the model's decisions overall")

        st.divider()
        st.subheader("🔑 Key SHAP Insights for This Dataset")
        st.markdown("""
        | Feature | Typical Finding |
        |---|---|
        | `st_slope` | Downsloping ST segment strongly associated with heart disease |
        | `chest_pain_type` | Asymptomatic chest pain (type 4) is paradoxically high risk |
        | `max_heart_rate` | Lower max heart rate = higher risk |
        | `oldpeak` | Higher ST depression = higher risk |
        | `cholesterol` | High cholesterol increases risk |
        | `age` | Older patients have higher risk |
        """)

# ══════════════════════════════════════════════════════════════
# PAGE — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div class="main-header">
      <h1>ℹ️ About Cardio Guard</h1>
      <p>BSCS Machine Learning Project · Ensemble Learning + XAI Edition</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Objective
    Cardio Guard predicts heart disease risk from 11 clinical parameters using **8 ML models**
    including two ensemble methods, with the best selected by AUC-ROC, and explains its
    decisions using SHAP (Explainable AI).

    ### 📦 Dataset
    - **Source:** [heart_statlog_cleveland_hungary_final.csv](https://www.kaggle.com/datasets/mahfuzhossain/heart-statlog-cleveland-hungary-final)
    - **Samples:** 1,190  ·  **Features:** 11  ·  **Target:** Binary (Heart Disease / No Disease)

    ### 🤖 Models (8 total)
    | Model | Type |
    |---|---|
    | Logistic Regression | Linear baseline |
    | Random Forest | Ensemble — Bagging |
    | Gradient Boosting | Ensemble — Boosting |
    | SVM | Kernel-based |
    | KNN | Instance-based |
    | Decision Tree | Interpretable tree |
    | **Voting Classifier** | **Ensemble — Soft Voting (LR + RF + GB + SVM)** |
    | **Stacking Classifier** | **Ensemble — Meta-learner (RF + GB + SVM + KNN → LR)** |

    ### 🧠 Ensemble Learning (NEW)
    - **Voting Ensemble:** Combines predictions from 4 models by averaging their
      probabilities (soft voting) — reduces variance and overfitting
    - **Stacking Ensemble:** Base models predict on held-out folds; a meta-learner
      (Logistic Regression) learns how to best combine their outputs — captures
      complementary strengths of different models

    ### 🔍 XAI — Explainable AI (NEW)
    - **SHAP (SHapley Additive exPlanations)** computes each feature's contribution
      to every individual prediction
    - Bar chart shows global importance; beeswarm shows per-patient impact
    - Makes the model trustworthy for medical decision support

    ### 🌐 HTML Report (NEW)
    A standalone `report.html` is generated after training containing all charts
    and metrics — downloadable from the Analytics page.

    ### 🛠️ Tech Stack
    Python · Scikit-learn · SHAP · Pandas · NumPy · Matplotlib · Seaborn · **Streamlit**

    ### 📋 Rubric Coverage
    | Component | Status |
    |---|---|
    | Proposal & Planning | ✅ README.md |
    | Data Preprocessing & EDA | ✅ 7 EDA charts |
    | ML Model Development | ✅ 6 base + 2 ensemble = 8 models |
    | Evaluation & Analysis | ✅ ROC, CM, CV, Classification Report |
    | Ensemble Learning | ✅ Voting + Stacking |
    | XAI | ✅ SHAP bar + beeswarm + interpretations |
    | HTML UI | ✅ Downloadable HTML report |
    | Web Interface | ✅ This Streamlit app |
    """)
    st.divider()
    st.caption("❤️ Cardio Guard · Created by Rabail Sarwar & Huzaifa Rehman Qureshi · BSCS 2025")