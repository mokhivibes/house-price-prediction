import streamlit as st
import joblib

st.title("House Price Predictor")
st.write("Ames, Iowa. Educational project, not financial advice.")

model = joblib.load("models/house_price_catboost.joblib")
st.write("Model loaded.")