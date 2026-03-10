import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RajEstate — Premium Price Predictor",
    page_icon="🏰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CUSTOM CSS & GOOGLE FONTS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;500;600&family=Cormorant+Garamond:wght@400;600;700&display=swap');

:root {
    --bg-color: #0c0d12;
    --card-bg: #16181f;
    --accent: #d4af37; /* Gold */
    --accent-dim: #8b732d;
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --border: #2d313d;
}

/* Global Styles */
.stApp {
    background-color: var(--bg-color);
    background-image: 
        radial-gradient(circle at 0% 0%, rgba(212, 175, 55, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 100% 100%, rgba(212, 175, 55, 0.05) 0%, transparent 40%);
    color: var(--text-main);
}

#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }

h1, h2, h3, .hero-title {
    font-family: 'Cinzel', serif;
}

body, p, label, .stWidgetLabel {
    font-family: 'Inter', sans-serif !important;
}

.block-container {
    max-width: 850px !important;
    padding: 3rem 1.5rem 5rem !important;
}

/* Hero Section */
.hero {
    text-align: center;
    padding-bottom: 4rem;
    position: relative;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.hero-eyebrow {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.4em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 1.5rem;
    text-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.hero-title span {
    background: linear-gradient(to right, #fde68a, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    color: var(--text-muted);
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
}

/* Section Labels */
.section-header {
    margin-top: 3rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.section-header span {
    font-family: 'Cinzel', serif;
    font-size: 0.9rem;
    letter-spacing: 0.15em;
    color: var(--accent);
    white-space: nowrap;
}
.section-header::after {
    content: '';
    height: 1px;
    background: var(--border);
    flex-grow: 1;
}

/* Form Container */
.main-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

/* Styled Widgets */
[data-testid="stSelectbox"] > div > div {
    background: #1f222b !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: white !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--accent) !important;
}

[data-testid="stNumberInput"] input {
    background: #1f222b !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 0.6rem 1rem !important;
}

/* --- PREMIUM SLIDER OVERHAUL --- */
[data-testid="stSlider"] {
    margin-bottom: 2.5rem !important;
    padding: 0 !important;
}

/* A. THE HIT AREA (Transparent Container) */
[data-testid="stSlider"] div[data-baseweb="slider"] {
    background: transparent !important;
    border: none !important;
    height: 48px !important; /* Larger interactive zone for smooth dragging */
    padding: 0 !important;
    margin-top: 10px !important;
    display: flex !important;
    align-items: center !important;
}

/* B. THE TRACK (Consistency for all 4) */
[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    background: rgba(45, 49, 61, 0.5) !important; /* Translucent gunmetal */
    height: 4px !important;
    padding: 0 !important;
    border: none !important;
    border-radius: 10px !important;
}

/* C. THE ACTIVE FILL (Gold) */
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: var(--accent) !important;
    height: 4px !important;
}

/* D. THE THUMB (Perfectly Centered & Identical) */
[data-testid="stSlider"] [role="slider"] {
    background-color: #ffffff !important;
    border: 3px solid var(--accent) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4), 0 0 0 6px rgba(212, 175, 55, 0.1) !important;
    width: 22px !important;
    height: 22px !important;
    /* Calculated Centering: (22-4)/2 = 9px offset */
    top: -9px !important; 
    transition: transform 0.1s ease-out, box-shadow 0.2s ease !important;
    cursor: grab !important;
}
[data-testid="stSlider"] [role="slider"]:active {
    cursor: grabbing !important;
    transform: scale(1.1) !important;
}

/* E. ABSOLUTE LABEL SUPPRESSION (Nuclear Option) */
/* This hides everything except the track and thumb */
[data-testid="stSlider"] [data-baseweb="slider"] + div, 
[data-testid="stSlider"] [data-baseweb="slider"] + div > div,
[data-testid="stSlider"] [role="slider"] div,
[data-testid="stTickBar"],
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] [aria-valuetext] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* F. REMOVE GHOST LINE / BORDER ARTIFACTS */
[data-testid="stSlider"] div[role="presentation"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Fix for any potential lingering red text labels */
[data-testid="stSlider"] div {
    color: inherit; /* Prevent forced red from defaults */
}

/* Radio Chip Cards */
[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
}
[data-testid="stRadio"] label {
    background: #1f222b !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    color: var(--text-muted) !important;
    transition: all 0.3s ease !important;
    flex-grow: 1 !important;
    text-align: center !important;
    cursor: pointer !important;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--accent-dim) !important;
    transform: translateY(-2px);
}
[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(212, 175, 55, 0.1) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.1);
}
[data-testid="stRadio"] input { display: none !important; }

/* Checkbox Style */
[data-testid="stCheckbox"] label span:first-child {
    border-radius: 4px !important;
    background-color: #1f222b !important;
    border-color: var(--border) !important;
}
[data-testid="stCheckbox"] label:has(input:checked) span:first-child {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* Estimate Button */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #d4af37 0%, #a68a2d 100%) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.1rem !important;
    text-transform: uppercase !important;
    margin-top: 2rem !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #fde68a 0%, #d4af37 100%) !important;
    transform: scale(1.02) !important;
    box-shadow: 0 15px 30px rgba(212, 175, 55, 0.3) !important;
}

/* Result Visualization */
.result-hero {
    background: radial-gradient(circle at center, #1e212b 0%, #16181f 100%);
    border: 2px solid var(--accent);
    border-radius: 24px;
    padding: 4rem 2rem;
    margin: 3rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
}
.result-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: url('https://www.transparenttextures.com/patterns/carbon-fibre.png');
    opacity: 0.05;
    pointer-events: none;
}
.val-label {
    font-family: 'Cinzel', serif;
    font-size: 0.85rem;
    letter-spacing: 0.3em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.val-main {
    font-family: 'Inter', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.5rem;
    letter-spacing: -2px;
}
.val-currency {
    color: var(--accent);
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 300;
    margin-right: 0.5rem;
}
.val-unit {
    font-size: 2rem;
    color: var(--text-muted);
    font-weight: 300;
    margin-left: 0.5rem;
}
.val-sub {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    color: var(--text-muted);
    font-style: italic;
    margin-top: 1rem;
}

/* Property Summary Cards */
.summary-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-top: 2rem;
}
.info-card {
    background: #1f222b;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.info-card:hover {
    border-color: var(--accent-dim);
    background: #252834;
}
.info-icon {
    font-size: 1.5rem;
    margin-bottom: 0.8rem;
    display: block;
}
.info-val {
    display: block;
    font-size: 1.2rem;
    font-weight: 600;
    color: white;
    margin-bottom: 0.2rem;
}
.info-key {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
}

.footer-luxury {
    text-align: center;
    padding: 4rem 0 2rem;
    color: var(--border);
    font-family: 'Cinzel', serif;
    letter-spacing: 0.2em;
    font-size: 0.7rem;
}

/* Hide Labels for a cleaner look when we use sections */
.stWidgetLabel p {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}

</style>
""", unsafe_allow_html=True)

# ── DATA & MODELS ──────────────────────────────────────────────────────────────
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
    # Keep exact loading logic as requested
    model    = joblib.load('knn_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()

# ── HERO HEADER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Exquisite Analytics</div>
    <div class="hero-title">Discover Your <span>Estate's</span> Worth</div>
    <div class="hero-sub">Sophisticated Prediction for the Modern Indian Market</div>
</div>
""", unsafe_allow_html=True)

# ── INPUT WORKFLOW ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="section-header"><span>Location & Prestige</span></div>', unsafe_allow_html=True)
    city = st.selectbox("Select Target City", sorted(CITY_PSF_MAP.keys()),
                        index=sorted(CITY_PSF_MAP.keys()).index("Bangalore"))
    
    st.markdown('<div class="section-header"><span>Architectural Details</span></div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        bedrooms = st.slider("Number of Bedrooms", 1, 5, 3)
    with cc2:
        bathrooms = st.slider("Number of Bathrooms", 1.0, 5.0, 2.0, step=0.5)
    
    sqft_living = st.number_input("Living Area (sq ft)", min_value=400, max_value=5000, value=1200, step=50)

    cc3, cc4 = st.columns(2)
    with cc3:
        floors = st.slider("Floor Level", 1, 10, 3)
    with cc4:
        age_years = st.slider("Property Age (Years)", 0, 35, 5)

    st.markdown('<div class="section-header"><span>Lifestyle & Amenities</span></div>', unsafe_allow_html=True)
    
    col_f, col_l = st.columns(2)
    with col_f:
        st.write("Furnishing Status")
        furnishing_label = st.radio("Furnishing", list(FURNISHING_MAP.keys()), index=1, horizontal=True, label_visibility="collapsed")
    with col_l:
        st.write("Locality Category")
        locality_label = st.radio("Locality", list(LOCALITY_MAP.keys()), index=1, horizontal=True, label_visibility="collapsed")

    st.write("Additional Provisions")
    aa1, aa2 = st.columns([2, 1])
    with aa1:
        parking = st.radio("Dedicated Parking Spots", [0, 1, 2], index=1, horizontal=True)
    with aa2:
        st.write("&nbsp;") # Spacer
        lift = 1 if st.checkbox("Elevator Access", value=True) else 0

    predict_btn = st.button("Evaluate Property Value")

# ── PREDICTION LOGIC & HERO OUTPUT ─────────────────────────────────────────────
if predict_btn:
    # Exact processing logic preserved
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

    # Hero Result
    st.markdown(f"""
<div class="result-hero">
    <div class="val-label">Market Valuation</div>
    <div class="val-main">
        <span class="val-currency">₹</span>{price_lakhs:,.2f}<span class="val-unit">Lakhs</span>
    </div>
    <div class="val-sub">Equivalent to approximately <strong>₹ {price_crores:.2f} Crores</strong></div>
</div>

<div class="section-header"><span>Property Specification Recap</span></div>

<div class="summary-container">
    <div class="info-card">
        <span class="info-icon">🏢</span>
        <span class="info-val">{city}</span>
        <span class="info-key">Location</span>
    </div>
    <div class="info-card">
        <span class="info-icon">🛌</span>
        <span class="info-val">{bedrooms} BHK</span>
        <span class="info-key">Configuration</span>
    </div>
    <div class="info-card">
        <span class="info-icon">📐</span>
        <span class="info-val">{sqft_living:,}</span>
        <span class="info-key">Area (Sq.Ft)</span>
    </div>
    <div class="info-card">
        <span class="info-icon">🛁</span>
        <span class="info-val">{bathrooms}</span>
        <span class="info-key">Bathrooms</span>
    </div>
    <div class="info-card">
        <span class="info-icon">🏗</span>
        <span class="info-val">Level {floors}</span>
        <span class="info-key">Floor</span>
    </div>
    <div class="info-card">
        <span class="info-icon">⏳</span>
        <span class="info-val">{age_years} Yrs</span>
        <span class="info-key">Vintage</span>
    </div>
    <div class="info-card">
        <span class="info-icon">🛋</span>
        <span class="info-val">{furnishing_label}</span>
        <span class="info-key">Furnishing</span>
    </div>
    <div class="info-card">
        <span class="info-icon">📍</span>
        <span class="info-val">{locality_label}</span>
        <span class="info-key">Locality Category</span>
    </div>
    <div class="info-card">
        <span class="info-icon">🚗</span>
        <span class="info-val">{parking} Bay{'s' if parking != 1 else ''}</span>
        <span class="info-key">Parking</span>
    </div>
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 5rem 0; opacity: 0.3; font-style: italic;">
        Enter property details and initiate evaluation to see results.
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-luxury">
    RAJESTATE AI • PRECISION VALUATION • EST. 2025
</div>
""", unsafe_allow_html=True)
