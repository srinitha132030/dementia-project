import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os


st.set_page_config(
    page_title="Dementia Pattern Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main { background-color: #0f0f0f; }
.stApp { background-color: #0f0f0f; color: #e8e8e8; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }

.risk-card {
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    border: 1px solid #2a2a2a;
}
.risk-high   { background: linear-gradient(135deg, #2d0a0a, #1a0606); border-color: #8b2020; }
.risk-med    { background: linear-gradient(135deg, #2d1a06, #1a0f03); border-color: #8b5e1a; }
.risk-low    { background: linear-gradient(135deg, #0a2d0a, #061a06); border-color: #1a8b1a; }

.pattern-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin: 2px;
}
.badge-critical { background:#8b2020; color:#ffcccc; }
.badge-warning  { background:#8b5e1a; color:#ffd9a0; }
.badge-normal   { background:#1a5c1a; color:#b3ffb3; }

.metric-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.1em; }

.stButton > button {
    background: #1a1a2e !important;
    color: #7eb3ff !important;
    border: 1px solid #7eb3ff !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #7eb3ff !important;
    color: #0f0f0f !important;
}

.stNumberInput input, .stSelectbox select {
    background: #1a1a1a !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
}

.pattern-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 14px;
    margin: 4px 0;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.5;
}
.pattern-critical { background: #1f0808; border-left: 3px solid #cc3333; color: #ffcccc; }
.pattern-warning  { background: #1f1208; border-left: 3px solid #cc8833; color: #ffd9a0; }
.pattern-good     { background: #081f08; border-left: 3px solid #33cc33; color: #b3ffb3; }

.verdict-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; margin: 0; }
.verdict-score { font-family: 'IBM Plex Mono', monospace; font-size: 3.5rem; font-weight: 600; line-height: 1; }
.verdict-sub   { font-size: 0.85rem; color: #aaa; margin-top: 6px; }

hr.divider { border: none; border-top: 1px solid #2a2a2a; margin: 1.5rem 0; }

[data-testid="stSidebar"] {
    background-color: #111 !important;
    border-right: 1px solid #2a2a2a;
}
</style>
""", unsafe_allow_html=True)
THRESHOLDS = {
    "mmse":  {"normal": (24, 30),  "mild": (20, 23),  "impaired": (0, 19)},
    "cdr":   {"normal": (0, 0),    "mild": (0.5, 0.5),"impaired": (1, 3)},
    "nwbv":  {"normal": (0.72, 1), "mild": (0.68, 0.72), "impaired": (0, 0.68)},
    "educ":  {"normal": (12, 23),  "mild": (8, 11),   "impaired": (0, 7)},
    "ses":   {"normal": (1, 2),    "mild": (3, 3),    "impaired": (4, 5)},
    "age":   {"normal": (0, 74),   "mild": (75, 84),  "impaired": (85, 120)},
}

def classify_marker(name, value):
    t = THRESHOLDS.get(name, {})
    for level in ("impaired", "mild", "normal"):
        lo, hi = t.get(level, (None, None))
        if lo is not None and lo <= value <= hi:
            return level
    return "normal"

def detect_patterns(inputs):
    """
    Rule-based pattern detection.
    Each pattern: (type, description, weight_bonus)
    type: 'critical' | 'warning' | 'good'
    """
    patterns = []
    mmse, cdr, nwbv = inputs["mmse"], inputs["cdr"], inputs["nwbv"]
    age, educ, ses   = inputs["age"],  inputs["educ"], inputs["ses"]
    etiv, asf        = inputs["etiv"], inputs["asf"]

    
    if mmse < 20:
        patterns.append(("critical", f"MMSE {mmse} — severe cognitive impairment (threshold < 20)", 18))
    elif mmse < 24:
        patterns.append(("warning",  f"MMSE {mmse} — mild cognitive impairment (normal ≥ 24)", 10))
    else:
        patterns.append(("good",     f"MMSE {mmse} — within normal cognitive range", 0))

    if cdr >= 1:
        patterns.append(("critical", f"CDR {cdr} — clinical dementia confirmed by rating scale", 20))
    elif cdr == 0.5:
        patterns.append(("warning",  "CDR 0.5 — questionable dementia / very mild impairment", 12))
    else:
        patterns.append(("good",     "CDR 0 — no clinical dementia detected on rating scale", 0))

    if nwbv < 0.68:
        patterns.append(("critical", f"nWBV {nwbv:.3f} — significant brain atrophy below 0.68", 14))
    elif nwbv < 0.72:
        patterns.append(("warning",  f"nWBV {nwbv:.3f} — mild volume reduction (normal ≥ 0.72)", 7))
    else:
        patterns.append(("good",     f"nWBV {nwbv:.3f} — brain volume within normal range", 0))

    if age >= 85:
        patterns.append(("warning",  f"Age {age} — advanced age significantly elevates baseline risk", 8))
    elif age >= 75:
        patterns.append(("warning",  f"Age {age} — elevated age increases dementia susceptibility", 4))

    if educ <= 7:
        patterns.append(("warning",  f"Education {educ} yrs — low cognitive reserve from limited schooling", 5))
    elif educ >= 16:
        patterns.append(("good",     f"Education {educ} yrs — high cognitive reserve provides protection", 0))

    if ses >= 4:
        patterns.append(("warning",  f"SES {ses} — low socioeconomic status correlates with higher risk", 4))

    
    if mmse < 24 and cdr >= 0.5:
        patterns.append(("critical",
            "PATTERN: Low MMSE + elevated CDR — dual cognitive marker alarm, high-confidence indicator", 15))

    if nwbv < 0.70 and mmse < 26:
        patterns.append(("critical",
            "PATTERN: Brain atrophy + cognitive decline — structural and functional markers co-occurring", 12))

    if cdr >= 1 and nwbv < 0.72:
        patterns.append(("critical",
            "PATTERN: Clinical CDR + reduced volume — strong neurodegeneration signature", 10))

    if mmse < 24 and age >= 75:
        patterns.append(("critical",
            "PATTERN: Cognitive impairment in elderly — age amplifies severity, accelerated risk", 8))

    if educ <= 8 and ses >= 3 and mmse < 26:
        patterns.append(("warning",
            "PATTERN: Low education + low SES + reduced MMSE — compound socio-cognitive vulnerability", 6))

    if mmse >= 26 and cdr == 0 and nwbv >= 0.72:
        patterns.append(("good",
            "PATTERN: Strong triple-normal — MMSE, CDR, and nWBV all healthy — low risk profile", 0))

    if etiv < 1200:
        patterns.append(("warning",  f"eTIV {etiv} — smaller intracranial volume may indicate structural constraints", 3))

    return patterns


def compute_risk_score(inputs, patterns):
    mmse, cdr, nwbv = inputs["mmse"], inputs["cdr"], inputs["nwbv"]
    age, educ, ses   = inputs["age"],  inputs["educ"], inputs["ses"]

    base = 0
    base += max(0, (30 - mmse) / 30) * 35
    base += min(cdr / 3, 1) * 30
    base += max(0, (0.88 - nwbv) / 0.28) * 15
    base += max(0, (age - 55) / 45) * 8
    base += max(0, (16 - educ) / 16) * 5
    base += max(0, (ses - 1) / 4) * 4

    bonus = sum(p[2] for p in patterns)
    total = min(100, base + bonus * 0.3)
    return round(total)


def load_model_prediction(inputs, risk_score):
    """
    Tries to load your actual pkl model.
    Falls back gracefully to pattern-only score if files not found.
    Returns (prediction_label, confidence, source)
    """
    try:
        model  = joblib.load("dementia_model.pkl")
        scaler = joblib.load("scaler.pkl")
        gender_val = 1 if inputs["gender"] == "Male" else 0
        arr = np.array([[
            inputs["mrd"], gender_val, inputs["visit"],
            inputs["age"], inputs["educ"], inputs["ses"],
            inputs["mmse"], inputs["cdr"], inputs["etiv"],
            inputs["nwbv"], inputs["asf"]
        ]])
        scaled = scaler.transform(arr)
        pred   = model.predict(scaled)[0]
        # If model has predict_proba use it; else derive from risk_score
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled)[0][1] * 100
        else:
            prob = risk_score
        label = "Dementia Detected" if pred == 1 else "No Dementia"
        return label, round(prob), "ML Model + Pattern Analysis"
    except FileNotFoundError:
        label = "Dementia Detected" if risk_score >= 55 else \
                "Borderline / Uncertain" if risk_score >= 30 else "No Dementia"
        return label, risk_score, "Pattern Analysis Only"
    except Exception as e:
        label = "Dementia Detected" if risk_score >= 55 else \
                "Borderline / Uncertain" if risk_score >= 30 else "No Dementia"
        return label, risk_score, f"Pattern Analysis Only (model error: {e})"

def plot_radar(inputs):
    labels = ["MMSE", "CDR", "nWBV", "Age Risk", "Low Educ", "Low SES"]
    mmse_n  = max(0, min(1, (30  - inputs["mmse"])  / 30))
    cdr_n   = min(1, inputs["cdr"] / 3)
    nwbv_n  = max(0, min(1, (0.88 - inputs["nwbv"]) / 0.28))
    age_n   = max(0, min(1, (inputs["age"]  - 55)   / 45))
    educ_n  = max(0, min(1, (16   - inputs["educ"]) / 16))
    ses_n   = max(0, (inputs["ses"] - 1) / 4)
    values  = [mmse_n, cdr_n, nwbv_n, age_n, educ_n, ses_n]

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#888", fontsize=10, fontfamily="monospace")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], color="#444")
    ax.yaxis.grid(True, color="#2a2a2a", linewidth=0.5)
    ax.xaxis.grid(True, color="#2a2a2a", linewidth=0.5)

    risk = sum(values[:-1]) / (N)
    color = "#cc3333" if risk > 0.55 else "#cc8833" if risk > 0.3 else "#33cc33"

    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.2)

    return fig


def plot_bar_markers(inputs):
    markers = {
        "MMSE\n(0=worst)":    (inputs["mmse"],  0,  30, 24, 20),
        "CDR\n(0=best)":      (inputs["cdr"],   0,   3,  0.5, 1),
        "nWBV":               (inputs["nwbv"],  0.5, 1, 0.72, 0.68),
        "Educ\n(yrs)":        (inputs["educ"],  0,  23, 12,  8),
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.5))
    fig.patch.set_facecolor("#111")

    for ax, (label, (val, lo, hi, warn_thresh, bad_thresh)) in zip(axes, markers.items()):
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")

        norm_val = (val - lo) / (hi - lo) if hi != lo else 0
        norm_val = max(0, min(1, norm_val))

        # For CDR and nWBV, "bad" direction is different — we handle via risk_score
        bar_color = "#cc3333" if val <= bad_thresh or val >= bad_thresh \
                    else "#cc8833" if val <= warn_thresh or val >= warn_thresh \
                    else "#33cc33"
        # Simpler: colour by classify_marker
        lvl_map = {"critical": "#cc3333", "warning": "#cc8833", "normal": "#33cc33",
                   "impaired": "#cc3333", "mild": "#cc8833"}
        mk_name = label.split("\n")[0].lower().replace("(yrs)", "").strip()
        name_map = {"mmse": "mmse", "cdr": "cdr", "nwbv": "nwbv", "educ": "educ"}
        lvl = classify_marker(name_map.get(mk_name, mk_name), val)
        bar_color = lvl_map.get(lvl, "#33cc33")

        ax.barh(0, norm_val, height=0.4, color=bar_color, alpha=0.85)
        ax.barh(0, 1,        height=0.4, color="#2a2a2a", alpha=0.4, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{label}\n{val}", color="#e8e8e8", fontsize=9,
                     fontfamily="monospace", pad=4)

    fig.tight_layout(pad=1.5)
    return fig

with st.sidebar:
    st.markdown("## 🧠 Patient Data")
    st.markdown("---")

    st.markdown("*Cognitive Scores*")
    mmse  = st.number_input("MMSE (0–30)",  min_value=0.0, max_value=30.0, value=27.0, step=0.5,
                             help="Mini-Mental State Exam. Normal ≥ 24.")
    cdr   = st.number_input("CDR (0–3)",    min_value=0.0, max_value=3.0,  value=0.0,  step=0.5,
                             help="Clinical Dementia Rating. 0 = normal.")

    st.markdown("*Demographics*")
    age   = st.number_input("Age",          min_value=18,  max_value=120,  value=72)
    educ  = st.number_input("Education (yrs)", min_value=0, max_value=23,  value=12)
    ses   = st.number_input("SES (1=high, 5=low)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    gender = st.selectbox("Gender", ["Female", "Male"])

    st.markdown("*Brain Volume*")
    nwbv  = st.number_input("nWBV (0.5–1.0)", min_value=0.5, max_value=1.0, value=0.74, step=0.01,
                              help="Normalised Whole Brain Volume. Normal 0.72–0.88.")
    etiv  = st.number_input("eTIV",          min_value=800,  max_value=2500, value=1500)
    asf   = st.number_input("ASF",           min_value=0.8,  max_value=1.8,  value=1.2,  step=0.01)

    st.markdown("*Visit Info*")
    visit = st.number_input("Visit #",       min_value=1, max_value=20, value=1)
    mrd   = st.number_input("MR Delay",      min_value=0, value=0)

    st.markdown("---")
    run = st.button("🔍  Analyze Patterns", use_container_width=True)

st.markdown("# 🧠 Dementia Pattern Analyzer")
st.markdown("Clinical decision support — pattern-based risk detection")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

if not run:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#555;font-family:IBM Plex Mono,monospace;'>
        <div style='font-size:4rem;margin-bottom:1rem;'>⬅</div>
        <div style='font-size:1.1rem;'>Fill in patient data in the sidebar<br>and click <strong style='color:#7eb3ff'>Analyze Patterns</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


inputs = dict(mmse=mmse, cdr=cdr, nwbv=nwbv, age=age, educ=educ, ses=ses,
              etiv=etiv, asf=asf, visit=visit, mrd=mrd, gender=gender)

patterns    = detect_patterns(inputs)
risk_score  = compute_risk_score(inputs, patterns)
label, conf, source = load_model_prediction(inputs, risk_score)

critical_n = sum(1 for p in patterns if p[0] == "critical")
warning_n  = sum(1 for p in patterns if p[0] == "warning")
good_n     = sum(1 for p in patterns if p[0] == "good")


verdict_class = "risk-high" if risk_score >= 55 else "risk-med" if risk_score >= 30 else "risk-low"
verdict_color = "#cc3333"   if risk_score >= 55 else "#cc8833"  if risk_score >= 30 else "#33cc33"

st.markdown(f"""
<div class="risk-card {verdict_class}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
    <div>
      <p class="verdict-title" style="color:{verdict_color};">{label}</p>
      <p class="verdict-sub">{source}</p>
      <div style="margin-top:0.75rem;display:flex;gap:6px;flex-wrap:wrap;">
        <span class="pattern-badge badge-critical">{critical_n} critical patterns</span>
        <span class="pattern-badge badge-warning">{warning_n} warnings</span>
        <span class="pattern-badge badge-normal">{good_n} normal</span>
      </div>
    </div>
    <div style="text-align:right;">
      <p class="verdict-score" style="color:{verdict_color};">{conf}</p>
      <p style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#888;">RISK SCORE / 100</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown("### Key Markers")
c1, c2, c3, c4, c5 = st.columns(5)

def marker_html(val, label_top, status):
    color = {"impaired":"#cc3333","mild":"#cc8833","normal":"#33cc33"}.get(status, "#888")
    return f"""
    <div class='metric-box'>
      <div class='metric-value' style='color:{color};'>{val}</div>
      <div class='metric-label'>{label_top}</div>
      <div style='font-size:10px;color:{color};margin-top:4px;font-family:IBM Plex Mono,monospace;'>{status.upper()}</div>
    </div>"""

with c1: st.markdown(marker_html(mmse,  "MMSE",  classify_marker("mmse", mmse)),  unsafe_allow_html=True)
with c2: st.markdown(marker_html(cdr,   "CDR",   classify_marker("cdr",  cdr)),   unsafe_allow_html=True)
with c3: st.markdown(marker_html(f"{nwbv:.2f}", "nWBV", classify_marker("nwbv", nwbv)), unsafe_allow_html=True)
with c4: st.markdown(marker_html(age,   "Age",   classify_marker("age",  age)),   unsafe_allow_html=True)
with c5: st.markdown(marker_html(educ,  "Educ",  classify_marker("educ", educ)),  unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


col_p, col_r = st.columns([3, 2])

with col_p:
    st.markdown("### Detected Patterns")
    # Sort: critical first, then warning, then good
    order = {"critical": 0, "warning": 1, "good": 2}
    for ptype, pdesc, _ in sorted(patterns, key=lambda x: order[x[0]]):
        css = f"pattern-{ptype if ptype != 'good' else 'good'}"
        icon = "⚠" if ptype == "critical" else "△" if ptype == "warning" else "✓"
        st.markdown(f"""
        <div class="pattern-row {css}">
          <span style="font-size:16px;flex-shrink:0;">{icon}</span>
          <span>{pdesc}</span>
        </div>""", unsafe_allow_html=True)

with col_r:
    st.markdown("### Risk Radar")
    fig_r = plot_radar(inputs)
    st.pyplot(fig_r, use_container_width=True)
    plt.close(fig_r)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


st.markdown("### Marker Severity Bars")
fig_b = plot_bar_markers(inputs)
st.pyplot(fig_b, use_container_width=True)
plt.close(fig_b)


st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("### Clinical Interpretation")

if risk_score >= 55:
    st.error(f"""
*High Risk — Dementia indicators present*

{critical_n} critical pattern(s) detected. The combination of clinical markers (MMSE={mmse}, CDR={cdr}, nWBV={nwbv:.2f}) 
suggests significant cognitive impairment. Immediate neurological evaluation is recommended.

- Refer to specialist for comprehensive neuropsychological assessment
- Consider MRI for structural brain evaluation  
- Review medications, lifestyle factors, and comorbidities
- Establish care plan and support systems
""")
elif risk_score >= 30:
    st.warning(f"""
*Borderline Risk — Some markers elevated*

{warning_n} warning pattern(s) detected. Current scores suggest mild or questionable cognitive changes. 
This warrants close monitoring and follow-up.

- Schedule follow-up assessment in 6–12 months
- Track MMSE and CDR progression over time
- Encourage cognitive engagement and physical activity
- Review vascular and metabolic risk factors
""")
else:
    st.success(f"""
*Low Risk — Markers within normal range*

{good_n} normal pattern(s) confirmed. Current cognitive and structural markers are within acceptable 
ranges. Routine monitoring is sufficient at this stage.

- Continue routine cognitive health check-ups
- Maintain healthy lifestyle (exercise, diet, cognitive stimulation)
- Reassess at next scheduled visit
""")

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:11px;color:#444;font-family:IBM Plex Mono,monospace;'>"
    "This tool is for clinical decision support only and does not replace professional medical diagnosis. "
    "Pattern analysis based on OASIS longitudinal dataset thresholds.</p>",
    unsafe_allow_html=True
)