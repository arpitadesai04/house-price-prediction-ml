from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from src.data_utils import load_data, prepare_data

MODEL_PATH = Path("models/house_price_pipeline.joblib")
DATA_PATH = Path("data/raw/Bengaluru_House_Data.csv")

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")
st.title("🏡 Bengaluru House Price Prediction")
st.write("Enter the property details to estimate its price using a trained Machine Learning model.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. First download the dataset and run `python train.py`.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Use training data to populate known locations and detect whether balcony was a model feature.
try:
    raw = load_data(DATA_PATH)
    cleaned, feature_cols = prepare_data(raw)
    locations = sorted(cleaned["location"].dropna().unique().tolist())
except Exception as exc:
    st.error(f"Could not read the training dataset: {exc}")
    st.stop()

st.subheader("Property details")
location = st.selectbox("Location", locations)
area = st.number_input("Total area (square feet)", min_value=100.0, max_value=50000.0, value=1200.0, step=50.0)
bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=20, value=2, step=1)
bath = st.number_input("Bathrooms", min_value=1.0, max_value=20.0, value=2.0, step=1.0)

input_data = {
    "location": location,
    "total_sqft_num": area,
    "bhk": float(bhk),
    "bath": float(bath)
}
if "balcony" in feature_cols:
    balcony = st.number_input("Balconies", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
    input_data["balcony"] = balcony

if st.button("Predict House Price", type="primary", use_container_width=True):
    row = pd.DataFrame([input_data])[feature_cols]
    prediction_lakh = float(model.predict(row)[0])
    st.success(f"Estimated price: ₹{prediction_lakh:,.2f} lakh")
    st.caption(f"Approximately ₹{prediction_lakh * 100000:,.0f}. This is a model estimate, not a formal property valuation.")

st.divider()
st.caption("Prediction quality depends on the dataset, its coverage and the property's actual condition.")
