import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndoPrime Valor — Luxury Real Estate Predictor",
    page_icon="🏙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CUSTOM CSS & GOOGLE FONTS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep: #05070a;
    --emerald: #10b981;
    --emerald-deep: #064e3b;
    --text-white: #f9fafb;
    --text-dim: #9ca3af;
    --border-soft: rgba(255, 255, 255, 0.08);
    --glass: rgba(255, 255, 255, 0.03);
}

/* Base Styles */
.stApp {
    background-color: var(--bg-deep);
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
    color: var(--text-white);
    font-family: 'Outfit', sans-serif;
}

#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }

.block-container {
    max-width: 800px !important;
    padding: 2rem 1rem 5rem !important;
}

/* Typography */
h1, h2, h3, .brand-font {
    font-family: 'Playfair Display', serif !important;
}

body, p, label, .stWidgetLabel {
    font-family: 'Outfit', sans-serif !important;
}

/* Hero Section */
.hero-container {
    padding: 4rem 0 3rem;
    text-align: center;
}
.hero-tag {
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--emerald);
    margin-bottom: 1rem;
    display: inline-block;
}
.hero-main-title {
    font-size: 3.8rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 1.5rem;
    color: #ffffff;
}
.hero-main-title span {
    font-style: italic;
    color: var(--emerald);
}
.hero-desc {
    font-size: 1.1rem;
    color: var(--text-dim);
    max-width: 600px;
    margin: 0 auto;
    font-weight: 300;
}

/* Custom Section Headers */
.custom-header {
    margin: 3rem 0 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.custom-header h3 {
    font-size: 1rem;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    white-space: nowrap;
}
.custom-header .line {
    height: 1px;
    background: var(--border-soft);
    flex-grow: 1;
}

/* Widget Styling */
[data-testid="stSelectbox"] > div > div {
    background: #0d0f14 !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 4px !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--emerald) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
}

[data-testid="stNumberInput"] input {
    background: #0d0f14 !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--emerald) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
}

/* Radio Toggle Chips */
[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 12px !important;
}
[data-testid="stRadio"] label {
    background: var(--glass) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
    padding: 12px 24px !important;
    color: var(--text-dim) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    font-weight: 400 !important;
    flex: 1 !important;
    text-align: center !important;
}
[data-testid="stRadio"] label:hover {
    background: rgba(16, 185, 129, 0.05) !important;
    border-color: rgba(16, 185, 129, 0.3) !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: var(--emerald) !important;
    color: #ffffff !important;
    border-color: var(--emerald) !important;
    font-weight: 600 !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25) !important;
}
[data-testid="stRadio"] input { display: none !important; }

/* Perfected Sliders */
[data-testid="stSlider"] {
    margin-bottom: 2rem !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] + div,
[data-testid="stTickBar"] {
    display: none !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] {
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    background: #111827 !important; /* Extremely dark track */
    height: 4px !important;
    border-radius: 10px !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: var(--emerald) !important;
    height: 4px !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #ffffff !important;
    border: 3px solid var(--emerald) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    width: 20px !important;
    height: 20px !important;
    top: -8px !important;
    outline: none !important;
}

/* Checkbox Styling */
[data-testid="stCheckbox"] label span:first-child {
    background-color: #0d0f14 !important;
    border-color: var(--border-soft) !important;
    border-radius: 6px !important;
}
[data-testid="stCheckbox"] label:has(input:checked) span:first-child {
    background-color: var(--emerald) !important;
    border-color: var(--emerald) !important;
}

/* Action Button */
.stButton > button {
    width: 100% !important;
    background: var(--emerald) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1.25rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    margin-top: 2rem !important;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: #34d399 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 40px rgba(16, 185, 129, 0.3) !important;
}

/* Result Hero Output */
.result-card {
    background: linear-gradient(135deg, #111827 0%, #05070a 100%);
    border: 1px solid var(--emerald);
    border-radius: 32px;
    padding: 4.5rem 2rem;
    margin: 4rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 30px 60px rgba(0,0,0,0.4);
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.15), transparent 60%);
    pointer-events: none;
}
.res-pre {
    font-size: 0.8rem;
    letter-spacing: 0.4em;
    text-transform: uppercase;
    color: var(--emerald);
    margin-bottom: 2rem;
    font-weight: 600;
}
.res-val {
    font-size: 5.5rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 0.9;
    letter-spacing: -3px;
    margin-bottom: 0.5rem;
}
.res-val span {
    font-size: 2.2rem;
    color: var(--emerald);
    font-weight: 300;
    margin-right: 0.5rem;
}
.res-val-label {
    font-size: 1.8rem;
    color: #4b5563;
    font-weight: 300;
}
.res-sub {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.3rem;
    color: var(--text-dim);
    margin-top: 1.5rem;
}

/* Property Grid Recap */
.prop-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 3rem;
}
.prop-item {
    background: var(--glass);
    border: 1px solid var(--border-soft);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.prop-item:hover {
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.03);
}
.prop-icon {
    font-size: 1.6rem;
    margin-bottom: 0.75rem;
    display: block;
}
.prop-data {
    display: block;
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.25rem;
}
.prop-label {
    display: block;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4b5563;
}

.footer {
    text-align: center;
    padding: 5rem 0 3rem;
    color: #374151;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Adjust labels */
.stWidgetLabel p {
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

</style>
""", unsafe_allow_html=True)

# ── DATA & MODELS (IDENTICAL AS REQUESTED) ─────────────────────────────────────
CITY_PSF_MAP = {
    'Mumbai': 25000, 'Delhi': 18000, 'Bangalore': 15000,
    'Hyderabad': 12000, 'Chennai': 11000, 'Kolkata': 8000,
    'Pune': 10000, 'Ahmedabad': 7000, 'Surat': 6500,
    'Jaipur': 5500, 'Lucknow': 4500, 'Nagpur': 4800,
    'Indore': 5000, 'Bhopal': 4200, 'Visakhapatnam': 5800,
    'Vijayawada': 5200, 'Coimbatore': 5000, 'Kochi': 7500,
    'Chandigarh': 7000, 'Guwahati': 4000, 'Patna': 3800,
    'Ranchi': 3500, 'Vadodara': 5500, 'Rajkot': 4800,
    'Amritsar': 4500, 'Varanasi': 3800, 'Agra': 3500,
    'Mysuru': 5500, 'Thiruvananthapuram': 6000,
    'Noida': 9000, 'Gurgaon': 11000, 'Navi Mumbai': 14000,
    'Thane': 13000,
}
FURNISHING_MAP = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
LOCALITY_MAP   = {'Premium': 1, 'Mid-range': 2, 'Budget': 3}

@st.cache_resource
def load_artifacts():
    # Exact loading logic as requested
    model    = joblib.load('knn_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()

# ── HERO CONTENT ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-tag">Precision Analytics</div>
    <h1 class="hero-main-title">Calculate Your<br><span>Property's</span> True Worth</h1>
    <p class="hero-desc">Bespoke AI-driven valuations for India's most exclusive real estate markets.</p>
</div>
""", unsafe_allow_html=True)

# ── INPUT WORKFLOW ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="custom-header"><h3>Location Profile</h3><div class="line"></div></div>', unsafe_allow_html=True)
    city = st.selectbox("Select Target City", sorted(CITY_PSF_MAP.keys()),
                        index=sorted(CITY_PSF_MAP.keys()).index("Bangalore"))
    
    st.markdown('<div class="custom-header"><h3>Infrastructure</h3><div class="line"></div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        bedrooms = st.slider("Sleeping Quarters (BHK)", 1, 5, 3)
    with c2:
        bathrooms = st.slider("Bathrooms Available", 1.0, 5.0, 2.0, step=0.5)
    
    sqft_living = st.number_input("Total Living Area (Sq.Ft)", min_value=400, max_value=5000, value=1200, step=50)

    c3, c4 = st.columns(2)
    with c3:
        floors = st.slider("Position (Floor No.)", 1, 10, 3)
    with c4:
        age_years = st.slider("Structure Age (Years)", 0, 35, 5)

    st.markdown('<div class="custom-header"><h3>Finishing & Locality</h3><div class="line"></div></div>', unsafe_allow_html=True)
    
    cf, cl = st.columns(2)
    with cf:
        st.write("Interior Finish")
        furnishing_label = st.radio("Furnishing", list(FURNISHING_MAP.keys()), index=1, horizontal=True, label_visibility="collapsed")
    with cl:
        st.write("Locality Prestige")
        locality_label = st.radio("Locality", list(LOCALITY_MAP.keys()), index=1, horizontal=True, label_visibility="collapsed")

    st.write("Auxiliary Assets")
    a1, a2 = st.columns([2, 1])
    with a1:
        parking = st.radio("Allocated Parking Bays", [0, 1, 2], index=1, horizontal=True)
    with a2:
        st.write("&nbsp;") 
        lift = 1 if st.checkbox("Automated Lift Access", value=True) else 0

    predict_btn = st.button("Initiate Valuation →")

# ── PREDICTION & DYNAMIC OUTPUT ────────────────────────────────────────────────
if predict_btn:
    # Logic remains absolutely untouched
    inp = pd.DataFrame([{
        'city_base_psf': CITY_PSF_MAP[city],
        'bedrooms':      bedrooms,
        'bathrooms':     bathrooms,
        'sqft_living':   sqft_living,
        'floors':        floors,
        'age_years':     age_years,
        'furnishing':    FURNISHING_MAP[furnishing_label],
        'locality_tier': LOCALITY_MAP[locality_label],
        'parking':       parking,
        'lift':          lift,
    }])[FEATURES]

    price_lakhs  = np.expm1(model.predict(scaler.transform(inp))[0])
    price_crores = price_lakhs / 100

    # Hero Result Render
    st.markdown(f"""
<div class="result-card">
    <div class="res-pre">Estimated Valuation</div>
    <div class="res-val">
        <span>₹</span>{price_lakhs:,.2f}<span class="res-val-label">&nbsp;L</span>
    </div>
    <div class="res-sub">Approximately ₹ {price_crores:.2f} Crores</div>
</div>

<div class="custom-header"><h3>Property Archetype</h3><div class="line"></div></div>

<div class="prop-grid">
    <div class="prop-item">
        <span class="prop-icon">📍</span>
        <span class="prop-data">{city}</span>
        <span class="prop-label">Market</span>
    </div>
    <div class="prop-item">
        <span class="prop-icon">🏠</span>
        <span class="prop-data">{bedrooms} BHK</span>
        <span class="prop-label">Config</span>
    </div>
    <div class="prop-item">
        <span class="prop-icon">📏</span>
        <span class="prop-data">{sqft_living:,}</span>
        <span class="prop-label">Sq. Ft</span>
    </div>
    <div class="prop-item">
        <span class="prop-icon">🛁</span>
        <span class="prop-data">{bathrooms}</span>
        <span class="prop-label">Bath</span>
    </div>
    <div class="prop-item">
        <span class="prop-icon">🏙️</span>
        <span class="prop-data">Level {floors}</span>
        <span class="prop-label">Elevation</span>
    </div>
    <div class="prop-item">
        <span class="prop-icon">⏳</span>
        <span class="prop-data">{age_years} Yrs</span>
        <span class="prop-label">Vintage</span>
    </div>
    <div class="prop-item">
        <span class="prop-icon">🛋️</span>
        <span class="prop-data">{furnishing_label}</span>
        <span class="prop-label">Interior</span>
    </div>
    <div class="prop-item">
        <span class="prop-icon">💎</span>
        <span class="prop-data">{locality_label}</span>
        <span class="prop-label">Locality</span>
    </div>
    <div class="prop-item">
        <span class="prop-icon">🚗</span>
        <span class="prop-data">{parking} Slot{'s' if parking != 1 else ''}</span>
        <span class="prop-label">Parking</span>
    </div>
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 5rem 0; color: #374151; font-style: italic; font-weight: 300; font-family: 'Playfair Display', serif;">
        Customize property variables to generate a market valuation report.
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    İNDOPRİME VALOR © 2025 • LUXURY ANALYTICS SİSTEM
</div>
""", unsafe_allow_html=True)
