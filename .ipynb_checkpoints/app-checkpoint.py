import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import timedelta

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = joblib.load("./models/final_xgboost_forecasting_model.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Sales Forecast Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;color:#00c6ff;'>📊 AI-Powered Sales Forecast Intelligence System</h1>", unsafe_allow_html=True)
st.write("Upload data → Analyze → Predict → Download Results")

# ===============================
# FILE UPLOAD
# ===============================
uploaded = st.file_uploader("📁 Upload CSV (Sales + Date)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # -------- Date Handling (Fixed) --------
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if not date_col:
        st.error("❌ No Date column found. Must contain 'date'.")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, format='mixed', errors='coerce')
    df = df.dropna(subset=[date_col])

    # -------- Sales Column Auto-Detect --------
    sales_col = None
    for col in df.columns:
        if ("sale" in col.lower()) or ("revenue" in col.lower()) or ("amount" in col.lower()):
            sales_col = col; break

    if not sales_col:
        num = df.select_dtypes(include=['float','int']).columns
        if len(num)==1:
            sales_col = num[0]; st.warning(f"⚠ Auto-selected numeric column: {sales_col}")
        else:
            st.error("Sales column not found — rename or tell me which one is sales."); st.stop()

    # ===============================
    # METRICS SECTION 📊
    # ===============================
    total_sales = df[sales_col].sum()
    avg_daily = df.groupby(date_col)[sales_col].sum().mean()
    last_30 = df.groupby(date_col)[sales_col].sum().tail(30).sum()
    prev_30 = df.groupby(date_col)[sales_col].sum().tail(60).head(30).sum()
    growth = ((last_30-prev_30)/prev_30*100) if prev_30>0 else 0

    c1,c2,c3 = st.columns(3)
    c1.metric("💰 Total Sales", f"₹{total_sales:,.2f}")
    c2.metric("📅 Avg Sales per Day", f"₹{avg_daily:,.2f}")
    c3.metric("📈 30-Day Growth", f"{growth:.2f}%")

    # ===============================
    # DATA VISUALS
    # ===============================
    daily = df.groupby(date_col)[sales_col].sum()

    fig1 = px.line(daily, title="📊 Historical Daily Sales")
    st.plotly_chart(fig1, use_container_width=True)

    # -------- Weekly/Monthly View --------
    m1,m2 = st.columns(2)
    monthly = daily.resample("M").sum()
    weekly  = daily.resample("W").sum()

    m1.plotly_chart(px.bar(monthly, title="🗓 Monthly Sales"), use_container_width=True)
    m2.plotly_chart(px.bar(weekly, title="📅 Weekly Revenue Trend"), use_container_width=True)

    # ===============================
        # FORECAST PANEL 🔥
    # ===============================
    st.subheader("🔮 Generate Future Forecast")

    # Create base daily series
    ts = daily.asfreq("D", fill_value=0)
    data = pd.DataFrame({"y": ts})
    data["date"] = data.index

    # ---------------------------
    # BASE FEATURES (TRAIN MATCH)
    # ---------------------------
    for lag in [1,2,7,14,28]:
        data[f"lag_{lag}"] = data["y"].shift(lag)

    data["roll_mean_7"]  = data["y"].rolling(7).mean()
    data["roll_mean_30"] = data["y"].rolling(30).mean()
    data["roll_std_7"]   = data["y"].rolling(7).std()

    # ---- Salary Cycle Impact ----
    data["day_of_month"]   = data["date"].dt.day
    data["is_salary_week"] = data["day_of_month"].between(1, 7).astype(int)
    data["is_month_end"]   = data["day_of_month"].between(25, 31).astype(int)
    data["days_from_salary"] = (data["day_of_month"] - 1).abs()
    data["salary_impact"] = np.where(data["days_from_salary"] <= 7,
                                 1 / (data["days_from_salary"] + 1), 0)

    data = data.dropna()
    feature_cols = [
    "lag_1", "lag_2", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_30", "roll_std_7",
    "dayofweek","month","year","is_weekend",
    "day_of_month","is_salary_week","is_month_end",
    "days_from_salary","salary_impact"
    ]


    # ---------------------------
    # USER FORECAST RANGE
    # ---------------------------
    days = st.slider("Forecast Days Ahead", 30, 365, 90)

    future = data.copy()

    for _ in range(days):

        last_row  = future.iloc[-1]  
        next_date = last_row["date"] + timedelta(days=1)

        row = {
            "date": next_date,

            # Time
            "dayofweek": next_date.weekday(),
            "month": next_date.month,
            "year": next_date.year,
            "is_weekend": int(next_date.weekday() >= 5),

            # Salary Effects
            "day_of_month": next_date.day,
            "is_salary_week": int(next_date.day <= 7),
            "is_month_end": int(next_date.day >= 25),
            "days_from_salary": abs(next_date.day - 1),
            "salary_impact": 1 / (abs(next_date.day - 1) + 1) if abs(next_date.day-1) <= 7 else 0,
        }

        # ---------------------------
        # ADD LAGS FROM FUTURE DATA
        # ---------------------------
        for lag in [1,2,7,14,28]:
            row[f"lag_{lag}"] = future["y"].iloc[-lag]

        # ---------------------------
        # ADD ROLLING WINDOWS
        # ---------------------------
        row["roll_mean_7"]  = future["y"].tail(7).mean()
        row["roll_mean_30"] = future["y"].tail(30).mean()
        row["roll_std_7"]   = future["y"].tail(7).std()

        # ---------------------------
        # FINAL MODEL PREDICTION
        # ---------------------------
        row["y"] = model.predict(pd.DataFrame([row])[feature_cols])[0]

        future = pd.concat([future, pd.DataFrame([row])], ignore_index=True)

    forecast = future.tail(days)[["date","y"]].rename(columns={"y":"forecast"})

    # ===============================
    # PLOT RESULT
    # ===============================
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"],    y=data["y"],        name="Historical"))
    fig.add_trace(go.Scatter(x=forecast["date"],y=forecast["forecast"], name="Forecast", line=dict(color="red", width=3)))
    fig.update_layout(title="📈 Future Forecast with Salary Cycle Intelligence", template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)
    st.download_button("⬇ Download Forecast CSV", forecast.to_csv(index=False), "forecast.csv")
