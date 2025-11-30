import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import base64
import io

# ============================
# CONFIG
# ============================
API_URL = "http://127.0.0.1:8000/forecast"   # FastAPI URL

st.set_page_config(page_title="AI Forecasting", layout="wide")

# ============================
# GLASSMORPHIC CSS
# ============================
glass_css = """
<style>
body {
    background: linear-gradient(135deg, #101020 0%, #0d0d0d 100%);
}

.section {
    background: rgba(255, 255, 255, 0.10);
    border-radius: 18px;
    padding: 25px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 20px rgba(0,255,255,0.2);
    margin-bottom: 25px;
}

.title {
    font-size: 36px;
    text-align: center;
    color: #00eaff;
    padding-bottom: 10px;
    text-shadow: 0 0 15px #00eaff;
}

.metric-card {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 0 20px rgba(0,150,255,0.3);
}

.upload-zone {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    border: 2px dashed rgba(0,255,255,0.3);
    padding: 40px;
    text-align: center;
    color: #00eaff;
}
</style>
"""

st.markdown(glass_css, unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown("<h1 class='title'>🚀 AI-Powered Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;color:#ddd;'>Upload dataset → Validate → Forecast → Download Report</p>", unsafe_allow_html=True)

# ============================
# FILE UPLOAD SECTION
# ============================
st.markdown("<div class='section'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload your Sales CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    st.markdown("<p style='color:#0ff;text-align:center;'>File uploaded ✔</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================
# MAIN ACTION
# ============================
if uploaded_file:

    st.markdown("<div class='section'>", unsafe_allow_html=True)

    st.subheader("⚙️ Configure Forecast")
    horizon = st.slider("Select forecast duration", 30, 365, 120)

    if st.button("✨ Generate Forecast", use_container_width=True):

        with st.spinner("Processing… Please wait ⏳"):

            files = {"file": uploaded_file.getvalue()}

            try:
                r = requests.post(f"{API_URL}?horizon={horizon}", files=files)

                if r.status_code != 200:
                    st.error(f"❌ API Error: {r.text}")
                    st.stop()

                response = r.json()

            except Exception as e:
                st.error(f"❌ Failed to connect to API.\n{e}")
                st.stop()

        forecast_data = pd.DataFrame(response["forecast"])

        st.success("🎉 Forecast generated successfully!")

        # ============================
        # KPI CARDS
        # ============================
        c1, c2, c3 = st.columns(3)

        c1.markdown(
            f"<div class='metric-card'><h3>Days Forecasted</h3><h2>{horizon}</h2></div>",
            unsafe_allow_html=True
        )
        c2.markdown(
            f"<div class='metric-card'><h3>Start Date</h3><h2>{forecast_data['date'].iloc[0]}</h2></div>",
            unsafe_allow_html=True
        )
        c3.markdown(
            f"<div class='metric-card'><h3>End Date</h3><h2>{forecast_data['date'].iloc[-1]}</h2></div>",
            unsafe_allow_html=True
        )

        # ============================
        # CHART
        # ============================
        st.markdown("<br>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_data["date"], y=forecast_data["forecast"],
            mode="lines", line=dict(color="#00eaff", width=3),
            name="Forecast"
        ))

        fig.update_layout(
            template="plotly_dark",
            title="📈 Future Sales Forecast",
            xaxis_title="Date", yaxis_title="Forecast Value"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ============================
        # DOWNLOAD (CSV)
        # ============================
        st.subheader("⬇️ Download Results")

        csv_data = forecast_data.to_csv(index=False).encode()

        st.download_button(
            "📄 Download Forecast CSV",
            csv_data,
            "forecast_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ============================
        # DOWNLOAD (PDF)
        # ============================
        try:
            import pdfkit
            html = forecast_data.to_html()
            pdf = pdfkit.from_string(html, False)

            st.download_button(
                "📘 Download Forecast PDF",
                pdf,
                "forecast_results.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except:
            st.warning("Install `pdfkit` & wkhtmltopdf` to enable PDF download.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload a CSV file to start forecasting.")

