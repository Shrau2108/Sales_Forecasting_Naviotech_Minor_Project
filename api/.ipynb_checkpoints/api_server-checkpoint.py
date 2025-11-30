from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from datetime import timedelta

# =============================
# CONFIG
# =============================
MODEL_PATH = "final_xgboost_forecasting_model.pkl"
HIST_DATA_PATH = "historical_data.csv"   # must contain: date, y + feature cols

# Load model
model = joblib.load(MODEL_PATH)

# Load base data (same structure as during training)
base_df = pd.read_csv(HIST_DATA_PATH)
base_df["date"] = pd.to_datetime(base_df["date"])

# IMPORTANT: feature_cols must match training
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_30", "roll_std_7",
    "dayofweek", "month", "year", "is_weekend",
    "day_of_month", "is_salary_week", "is_month_end",
    "days_from_salary", "salary_impact"
]

app = FastAPI(
    title="Sales Forecasting API",
    description="Real-time forecast API powered by XGBoost time-series model",
    version="1.0.0"
)

# =============================
# Request / Response Schemas
# =============================

class ForecastRequest(BaseModel):
    horizon_days: int = 30   # how many days into the future


class ForecastPoint(BaseModel):
    date: str
    forecast: float


class ForecastResponse(BaseModel):
    horizon_days: int
    points: List[ForecastPoint]


# =============================
# Helper: forecast into future
# =============================

def generate_future_forecast(horizon_days: int) -> pd.DataFrame:
    """
    Uses the trained model + last rows of historical data
    to simulate forward N days.
    """
    future = base_df.copy().sort_values("date").reset_index(drop=True)

    for _ in range(horizon_days):
        last_row = future.iloc[-1]
        next_date = last_row["date"] + timedelta(days=1)

        row = {
            "date": next_date,

            # Calendar features
            "dayofweek": next_date.weekday(),
            "month": next_date.month,
            "year": next_date.year,
            "is_weekend": int(next_date.weekday() >= 5),

            # Salary cycle
            "day_of_month": next_date.day,
            "is_salary_week": int(next_date.day <= 7),
            "is_month_end": int(next_date.day >= 25),
            "days_from_salary": abs(next_date.day - 1),
            "salary_impact": 1/(abs(next_date.day - 1)+1) if abs(next_date.day - 1) <= 7 else 0,
        }

        # Lag features from last known y
        for lag in [1, 2, 7, 14, 28]:
            row[f"lag_{lag}"] = future["y"].iloc[-lag]

        # Rolling
        row["roll_mean_7"]  = future["y"].tail(7).mean()
        row["roll_mean_30"] = future["y"].tail(30).mean()
        row["roll_std_7"]   = future["y"].tail(7).std()

        X_next = pd.DataFrame([row])[FEATURE_COLS]
        y_next = model.predict(X_next)[0]
        row["y"] = y_next

        future = pd.concat([future, pd.DataFrame([row])], ignore_index=True)

    forecast = future.tail(horizon_days)[["date", "y"]].rename(columns={"y": "forecast"})
    return forecast


# =============================
# FastAPI Endpoints
# =============================

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Sales Forecasting API running 🚀"}


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(req: ForecastRequest):
    horizon = max(1, min(req.horizon_days, 365))  # cap between 1 and 365
    forecast_df = generate_future_forecast(horizon)

    points = [
        ForecastPoint(date=str(row["date"].date()), forecast=float(row["forecast"]))
        for _, row in forecast_df.iterrows()
    ]

    return ForecastResponse(
        horizon_days=horizon,
        points=points
    )
