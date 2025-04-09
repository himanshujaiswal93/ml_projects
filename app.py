import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('liquidity_model.pkl')

# Title of the app
st.title("🚀 Cryptocurrency Liquidity Predictor")
st.markdown("Predict the liquidity ratio based on trading volume, price volatility, and other features.")

# Inputs
volume_ma_7 = st.number_input("7-Day Moving Average of Volume", min_value=0.0)
price_volatility = st.number_input("7-Day Price Volatility", min_value=0.0)
liquidity_ratio = st.number_input("Liquidity Ratio (Volume / Price)", min_value=0.0)

# Prediction
if st.button("Predict Liquidity"):
    input_data = np.array([[volume_ma_7, price_volatility, liquidity_ratio]])
    prediction = model.predict(input_data)
    st.success(f"🔮 Predicted Liquidity: {prediction[0]:.4f}")
# pip install streamlit scikit-learn numpy joblib
