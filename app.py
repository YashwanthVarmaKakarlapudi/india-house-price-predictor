import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #f97316, #ec4899) !important;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f97316 0%, #ec4899 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: rgba(255,255,255,0.45);
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(249,115,22,0.15), rgba(236,72,153,0.15));
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin: 1rem 0;
}
.result-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.result-price-lakhs {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f97316, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.result-price-crores {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.2rem;
    color: rgba(255,255,255,0.55);
    margin-top: 0.4rem;
}
.result-city-badge {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    border: 1px solid rgba(167,139,250,0.4);
    color: #a78bfa;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-top: 1rem;
    letter-spacing: 0.05em;
}

/* Summary card */
.summary-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.summary-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}
.summary-item {
    text-align: center;
}
.summary-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
}
.summary-key {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.15rem;
}

/* Divider line */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(249,115,22,0.4), rgba(236,72,153,0.4), transparent);
    margin: 1.5rem 0;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #f97316, #ec4899) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(249,115,22,0.35) !important;
}

/* Info box */
.info-box {
    background: rgba(167,139,250,0.08);
    border-left: 3px solid #a78bfa;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.5rem;
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    line-height: 1.6;
}

/* Sidebar section labels */
.sidebar-section {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: rgba(249,115,22,0.8) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.2rem 0 0.4rem 0;
}

/* All text white in main area */
.main * { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── City → PSF map ──────────────────────────────────────────────────────────────
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

FURNISHING_MAP  = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
LOCALITY_MAP    = {'Premium': 1, 'Mid-range': 2, 'Budget': 3}

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load('knn_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.6rem;">🏠 Inputs</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">📍 Location</p>', unsafe_allow_html=True)
    city = st.selectbox("City", sorted(CITY_PSF_MAP.keys()), index=sorted(CITY_PSF_MAP.keys()).index("Bangalore"))

    st.markdown('<p class="sidebar-section">🏗️ Property Details</p>', unsafe_allow_html=True)
    bedrooms    = st.slider("Bedrooms", 1, 5, 3)
    bathrooms   = st.slider("Bathrooms", 1.0, 5.0, 2.0, step=0.5)
    sqft_living = st.number_input("Area (sq ft)", min_value=400, max_value=5000, value=1200, step=50)
    floors      = st.slider("Floor", 1, 10, 5)
    age_years   = st.slider("Property Age (years)", 0, 35, 8)

    st.markdown('<p class="sidebar-section">✨ Amenities</p>', unsafe_allow_html=True)
    furnishing_label = st.radio("Furnishing", list(FURNISHING_MAP.keys()), index=1, horizontal=True)
    locality_label   = st.radio("Locality", list(LOCALITY_MAP.keys()), index=1, horizontal=True)
    parking          = st.radio("Parking Spots", [0, 1, 2], index=1, horizontal=True)
    lift             = 1 if st.checkbox("Lift Available", value=True) else 0

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Price")

# ── Main ─────────────────────────────────────────────────────────────────────────
col_title, col_space = st.columns([3, 1])
with col_title:
    st.markdown('<p class="hero-title">India House Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">KNN Regression · 33 Cities · 40,200 Samples · R² = 0.90</p>', unsafe_allow_html=True)

st.markdown('<div class="info-box">Fill in the property details on the left sidebar and click <strong>Predict Price</strong> to get an instant estimate powered by a K-Nearest Neighbours model trained on Indian real estate data.</div>', unsafe_allow_html=True)

# ── Prediction ───────────────────────────────────────────────────────────────────
if predict_btn:
    city_base_psf   = CITY_PSF_MAP[city]
    furnishing_val  = FURNISHING_MAP[furnishing_label]
    locality_val    = LOCALITY_MAP[locality_label]

    input_df = pd.DataFrame([{
        'city_base_psf': city_base_psf,
        'bedrooms':      bedrooms,
        'bathrooms':     bathrooms,
        'sqft_living':   sqft_living,
        'floors':        floors,
        'age_years':     age_years,
        'furnishing':    furnishing_val,
        'locality_tier': locality_val,
        'parking':       parking,
        'lift':          lift,
    }])[FEATURES]

    scaled = scaler.transform(input_df)
    log_pred = model.predict(scaled)[0]
    price_lakhs  = np.expm1(log_pred)
    price_crores = price_lakhs / 100

    # Result card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Property Value</div>
        <div class="result-price-lakhs">₹ {price_lakhs:,.2f} L</div>
        <div class="result-price-crores">≈ ₹ {price_crores:.2f} Crores</div>
        <div class="result-city-badge">📍 {city}</div>
    </div>
    """, unsafe_allow_html=True)

    # Summary grid
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">Property Summary</div>
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
                <div class="summary-val">{sqft_living} sqft</div>
                <div class="summary-key">Area</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">Floor {floors}</div>
                <div class="summary-key">Floor No.</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{age_years} yrs</div>
                <div class="summary-key">Property Age</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{furnishing_label}</div>
                <div class="summary-key">Furnishing</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{locality_label}</div>
                <div class="summary-key">Locality</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{parking} spot{'s' if parking != 1 else ''}</div>
                <div class="summary-key">Parking</div>
            </div>
            <div class="summary-item">
                <div class="summary-val">{'Yes' if lift else 'No'}</div>
                <div class="summary-key">Lift</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Placeholder
    st.markdown("""
    <div style="
        border: 2px dashed rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        color: rgba(255,255,255,0.2);
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
    ">
        🔮 Your predicted price will appear here
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem; font-family:'DM Sans',sans-serif;">
KNN Regression · manhattan distance · k=11 · StandardScaler · log1p target transform
</p>
""", unsafe_allow_html=True)
