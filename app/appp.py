# streamlit_app/app.py
import streamlit as st
import pandas as pd
import io
from pathlib import Path
from src.train_model import train_model
from src.forecast_future import generate_future_forecast

st.set_page_config(page_title="Sales Forecast", layout="wide")

st.title("📊 Sales Forecast Dashboard (Upload → Train → Forecast)")

uploaded = st.file_uploader("Upload CSV (raw sales)", type=["csv"])
if uploaded is not None:
    # save uploaded file directly to data/raw/
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    fpath = raw_dir / uploaded.name
    with open(fpath, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success(f"Saved uploaded file → {fpath}")

    if st.button("Train model on this file"):
        with st.spinner("Training..."):
            train_model(str(fpath))
        st.success("Training finished. Model saved.")

    st.markdown("---")
    days = st.slider("Forecast days", min_value=30, max_value=365, value=90)
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            forecast = generate_future_forecast(days)
        st.success("Forecast ready")
        st.dataframe(forecast)
        csv = forecast.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", data=csv, file_name="future_forecast.csv")
else:
    st.info("Upload a CSV to begin.")
