# 🏠 India House Price Predictor

A machine learning web app that predicts house prices across 33 Indian cities using K-Nearest Neighbours regression.

## Model Performance
- **Algorithm:** KNeighborsRegressor (k=11, distance weights, manhattan metric)
- **R² Score:** 0.90 (lakh scale: 0.89)
- **Training data:** 40,200 samples across 33 Indian cities
- **Target:** Price in ₹ Lakhs (log-transformed during training)

## Features Used
`city_base_psf`, `bedrooms`, `bathrooms`, `sqft_living`, `floors`, `age_years`, `furnishing`, `locality_tier`, `parking`, `lift`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud (Free)

1. Create a GitHub repo and push these files:
   - `app.py`
   - `knn_model.pkl`
   - `scaler.pkl`
   - `feature_names.pkl`
   - `requirements.txt`
   - `README.md`

2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → Sign in with GitHub

3. Click **New app** → Select your repo → Set main file to `app.py` → Click **Deploy**

That's it — live in ~2 minutes! 🚀

## Project Structure
```
├── app.py                  # Streamlit web app
├── knn_model.pkl           # Trained KNN model
├── scaler.pkl              # Fitted StandardScaler
├── feature_names.pkl       # Feature column order
├── requirements.txt        # Python dependencies
└── README.md
```
