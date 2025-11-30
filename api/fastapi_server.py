# ==============================
# FASTAPI SALES FORECAST SERVER
# ==============================

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import joblib
from datetime import timedelta
import io

app = FastAPI(title="AI Sales Forecast API", version="1.0")

MODEL_PATH = "models/final_xgboost_forecasting_model.pkl"


# -----------------------------------------
# Helper: Detect date + target columns
# -----------------------------------------
def detect_columns(df):
    date_col = None
    target_col = None

    # Detect date column
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        raise HTTPException(status_code=400, detail="❌ No date column found.")

    # Detect target column (sales)
    for col in df.columns:
        if any(k in col.lower() for k in ["sale", "amount", "revenue"]):
            target_col = col
            break

    # If not found → auto numeric
    if target_col is None:
        num_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
        if len(num_cols) == 1:
            target_col = num_cols[0]
        else:
            raise HTTPException(
                status_code=400,
                detail="❌ No sales column detected. Please rename your target column."
            )

    return date_col, target_col


# -----------------------------------------
# API ROUTES
# -----------------------------------------
@app.get("/")
def root():
    return {"status": "API running", "routes": ["/forecast"]}


@app.post("/forecast")
async def forecast_api(horizon: int = 60, file: UploadFile = File(...)):
    """
    Upload CSV → Predict next N days → Return forecast JSON
    """

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

    except Exception:
        raise HTTPException(status_code=400, detail="❌ Unable to read CSV. Upload a valid file.")

    # Detect date + target columns
    try:
        date_col, target_col = detect_columns(df)
    except HTTPException as e:
        return JSONResponse(status_code=400, content={"error": e.detail})

    # Clean & sort
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Build daily series
    ts = df.groupby(date_col)[target_col].sum().asfreq("D", fill_value=0)

    data = pd.DataFrame({"y": ts})
    data["date"] = data.index

    # Feature engineering
    for lag in [1,2,7,14,28]:
        data[f"lag_{lag}"] = data["y"].shift(lag)

    data["roll_mean_7"]  = data["y"].rolling(7).mean()
    data["roll_mean_30"] = data["y"].rolling(30).mean()
    data["roll_std_7"]   = data["y"].rolling(7).std()

    data["dayofweek"]  = data.date.dt.dayofweek
    data["month"]      = data.date.dt.month
    data["year"]       = data.date.dt.year
    data["is_weekend"] = (data.dayofweek >= 5).astype(int)

    data = data.dropna()

    feature_cols = [
        c for c in data.columns
        if c not in ["y", "date"]
    ]

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except:
        raise HTTPException(status_code=500, detail="❌ Model file missing. Train model first.")

    # Multi-step forecasting
    future = data.copy().reset_index(drop=True)

    for _ in range(horizon):
        last = future.iloc[-1]
        next_date = last["date"] + timedelta(days=1)

        row = {
            "date": next_date,
            "dayofweek": next_date.weekday(),
            "month": next_date.month,
            "year": next_date.year,
            "is_weekend": int(next_date.weekday() >= 5),
        }

        for lag in [1,2,7,14,28]:
            row[f"lag_{lag}"] = future["y"].iloc[-lag]

        row["roll_mean_7"]  = future["y"].tail(7).mean()
        row["roll_mean_30"] = future["y"].tail(30).mean()
        row["roll_std_7"]   = future["y"].tail(7).std()

        X = pd.DataFrame([row])[feature_cols]
        row["y"] = float(model.predict(X)[0])

        future = pd.concat([future, pd.DataFrame([row])], ignore_index=True)

    final = future.tail(horizon)[["date", "y"]]
    final.columns = ["date", "forecast"]

    # Convert dates to strings for JSON
    final["date"] = final["date"].astype(str)

    return {"horizon": horizon, "forecast": final.to_dict(orient="records")}
