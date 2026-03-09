import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="GharMol — India House Price Predictor",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
    background-color: #F7F5F0;
    color: #1a1a1a;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
    max-width: 780px !important;
    padding: 2.5rem 1.5rem 4rem !important;
    margin: 0 auto;
}

.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid #E2DDD6;
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #C17F3E;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero-title span { color: #C17F3E; }
.hero-sub {
    font-size: 0.9rem;
    font-weight: 300;
    color: #888;
    letter-spacing: 0.02em;
}

.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #C17F3E;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E2DDD6;
}

[data-testid="stSelectbox"] > div > div {
    background: #FFFFFF !important;
    border: 1.5px solid #E2DDD6 !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    color: #1a1a1a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #C17F3E !important;
    box-shadow: 0 0 0 3px rgba(193,127,62,0.12) !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: #E2DDD6 !important;
    height: 3px !important;
    border-radius: 99px !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    background: linear-gradient(90deg, #C17F3E, #E09E55) !important;
    height: 3px !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #FFFFFF !important;
    border: 2.5px solid #C17F3E !important;
    width: 18px !important;
    height: 18px !important;
    top: -7px !important;
    box-shadow: 0 2px 6px rgba(193,127,62,0.25) !important;
}

[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    border: 1.5px solid #E2DDD6 !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    color: #1a1a1a !important;
    padding: 0.55rem 0.9rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #C17F3E !important;
    box-shadow: 0 0 0 3px rgba(193,127,62,0.12) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background: #F7F5F0 !important;
    border: 1.5px solid #E2DDD6 !important;
    color: #888 !important;
    border-radius: 8px !important;
}

[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 0.5rem !important;
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] label {
    background: #FFFFFF !important;
    border: 1.5px solid #E2DDD6 !important;
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    color: #555 !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #FFF8F1 !important;
    border-color: #C17F3E !important;
    color: #C17F3E !important;
    font-weight: 500 !important;
}
[data-testid="stRadio"] input { display: none !important; }

.stButton > button {
    width: 100% !important;
    background: #1a1a1a !important;
    color: #F7F5F0 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-top: 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #C17F3E !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(193,127,62,0.3) !important;
}

.result-wrap {
    background: #1a1a1a;
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(193,127,62,0.2), transparent 70%);
    border-radius: 50%;
}
.result-eyebrow {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #C17F3E;
    margin-bottom: 1rem;
}
.result-amount {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 700;
    color: #F7F5F0;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-symbol {
    font-size: 1.8rem;
    color: #C17F3E;
    vertical-align: super;
    margin-right: 0.2rem;
}
.result-crores {
    font-size: 1rem;
    color: #888;
    font-weight: 300;
    margin-bottom: 1.5rem;
}
.result-badge {
    display: inline-block;
    background: rgba(193,127,62,0.15);
    border: 1px solid rgba(193,127,62,0.3);
    color: #C17F3E;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.3rem 1rem;
    border-radius: 99px;
    letter-spacing: 0.08em;
}

.summary-wrap {
    background: #FFFFFF;
    border: 1px solid #E2DDD6;
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1.5rem;
}
.summary-heading {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #BBB;
    margin-bottom: 1.5rem;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem 0.8rem;
}
.summary-item { text-align: center; }
.summary-val {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 0.2rem;
}
.summary-key {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #BBB;
}
.summary-divider {
    height: 1px;
    background: #F0EDE8;
    margin: 1.2rem 0;
}

.placeholder {
    border: 1.5px dashed #D9D4CC;
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
    color: #CCC;
    font-size: 0.9rem;
    font-weight: 300;
    margin: 2rem 0;
}

.footer {
    text-align: center;
    font-size: 0.72rem;
    color: #CCC;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #E2DDD6;
    letter-spacing: 0.05em;
}

label, [data-testid="stWidgetLabel"] p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #555 !important;
}
</style>
""", unsafe_allow_html=True)

# ── City map ───────────────────────────────────────────────────────────────────
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
    model    = joblib.load('knn_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Valuation</div>
    <div class="hero-title">What's your home<br>worth in <span>India?</span></div>
    <div class="hero-sub">KNN Regression &nbsp;·&nbsp; 33 Cities &nbsp;·&nbsp; 40,200 Samples &nbsp;·&nbsp; R² = 0.90</div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
city = st.selectbox("City", sorted(CITY_PSF_MAP.keys()),
                    index=sorted(CITY_PSF_MAP.keys()).index("Bangalore"),
                    label_visibility="collapsed")

st.markdown('<div class="section-label">🏗 Property Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    bedrooms  = st.slider("Bedrooms", 1, 5, 3)
with col2:
    bathrooms = st.slider("Bathrooms", 1.0, 5.0, 2.0, step=0.5)

sqft_living = st.number_input("Area (sq ft)", min_value=400, max_value=5000, value=1200, step=50)

col3, col4 = st.columns(2)
with col3:
    floors    = st.slider("Floor", 1, 10, 3)
with col4:
    age_years = st.slider("Property Age (yrs)", 0, 35, 5)

st.markdown('<div class="section-label">✨ Amenities</div>', unsafe_allow_html=True)
furnishing_label = st.radio("Furnishing", list(FURNISHING_MAP.keys()), index=1, horizontal=True)
locality_label   = st.radio("Locality", list(LOCALITY_MAP.keys()), index=1, horizontal=True)

col5, col6 = st.columns(2)
with col5:
    parking = st.radio("Parking spots", [0, 1, 2], index=1, horizontal=True)
with col6:
    lift = 1 if st.checkbox("Lift available", value=True) else 0

predict = st.button("Estimate Price →")

# ── Output ─────────────────────────────────────────────────────────────────────
if predict:
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

    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-eyebrow">Estimated Market Value</div>
        <div class="result-amount">
            <span class="result-symbol">₹</span>{price_lakhs:,.2f} L
        </div>
        <div class="result-crores">approximately ₹ {price_crores:.2f} Crores</div>
        <div class="result-badge">📍 {city}</div>
    </div>

    <div class="summary-wrap">
        <div class="summary-heading">Property Summary</div>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-val">{bedrooms} BHK</div>
                <div class="summary-key">Bedrooms</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{bathrooms}</div>
                <div class="summary-key">Bathrooms</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{sqft_living:,} sqft</div>
                <div class="summary-key">Area</div>
            </div>
        </div>
        <div class="summary-divider"></div>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-val">Floor {floors}</div>
                <div class="summary-key">Floor No.</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{age_years} yrs</div>
                <div class="summary-key">Age</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{furnishing_label}</div>
                <div class="summary-key">Furnishing</div>
            </div>
        </div>
        <div class="summary-divider"></div>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-val">{locality_label}</div>
                <div class="summary-key">Locality</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{parking} spot{'s' if parking != 1 else ''}</div>
                <div class="summary-key">Parking</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{'Yes ✓' if lift else 'No'}</div>
                <div class="summary-key">Lift</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="placeholder">
        Fill in the details above and click <strong>Estimate Price</strong>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    KNN · manhattan · k=11 · StandardScaler · log1p transform &nbsp;·&nbsp; GharMol © 2025
</div>
""", unsafe_allow_html=True)
