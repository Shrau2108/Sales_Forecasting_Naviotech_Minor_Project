# src/forecast_future.py
import pandas as pd
import joblib
from datetime import timedelta
from pathlib import Path

def generate_future_forecast(days: int = 90):
    """
    Reads data/features/features_timeseries.csv and models/final_xgb_model.pkl,
    simulates multi-step forecast for `days` days, and saves data/features/future_forecast.csv
    """
    feature_path = Path("data/features/features_timeseries.csv")
    model_path   = Path("models/final_xgb_model.pkl")

    if not feature_path.exists():
        raise FileNotFoundError("Run training first — feature dataset missing.")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train model first.")

    data = pd.read_csv(feature_path, parse_dates=["date"])
    feature_cols = [c for c in data.columns if c not in ["y","date"]]

    model = joblib.load(model_path)

    future = data.copy().sort_values("date").reset_index(drop=True)

    for _ in range(days):
        last = future.iloc[-1]
        next_date = pd.to_datetime(last["date"]) + timedelta(days=1)

        row = {"date": next_date}

        # day features
        row["dayofweek"] = next_date.weekday()
        row["month"] = next_date.month
        row["year"] = next_date.year
        row["is_weekend"] = int(next_date.weekday() >= 5)
        row["day_of_month"] = next_date.day
        row["is_salary_week"] = 1 if next_date.day <= 7 else 0
        row["is_month_end"] = 1 if next_date.day >= 25 else 0
        row["days_from_salary"] = abs(next_date.day - 1)
        row["salary_impact"] = 1/(abs(next_date.day - 1)+1) if abs(next_date.day - 1) <= 7 else 0

        for lag in [1,2,7,14,28]:
            row[f"lag_{lag}"] = future["y"].iloc[-lag]

        row["roll_mean_7"]  = future["y"].tail(7).mean()
        row["roll_mean_30"] = future["y"].tail(30).mean()
        row["roll_std_7"]   = future["y"].tail(7).std()

        X_next = pd.DataFrame([row])[feature_cols]
        row["y"] = model.predict(X_next)[0]

        future = pd.concat([future, pd.DataFrame([row])], ignore_index=True)

    forecast = future.tail(days)[["date","y"]].rename(columns={"y":"forecast"})
    out = Path("data/features")
    out.mkdir(parents=True, exist_ok=True)
    forecast_path = out / "future_forecast.csv"
    forecast.to_csv(forecast_path, index=False)

    print(f"📈 Future forecast saved: {forecast_path}")
    return forecast
