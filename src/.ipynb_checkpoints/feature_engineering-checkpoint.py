# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path

def build_time_series_features(df: pd.DataFrame, date_col: str, target_col: str, freq: str = "D"):
    """
    Inputs: cleaned df with date_col and target_col.
    Outputs: dataframe with columns: date, y, lag_1..lag_28, roll_mean_7/30, roll_std_7,
             dayofweek, month, year, is_weekend, salary cycle features.
    Side-effect: saves data/features/features_timeseries.csv
    """
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col])
    ts = df2.groupby(date_col)[target_col].sum().sort_index()
    ts = ts.asfreq(freq, fill_value=0)  # missing dates as 0 sales

    data = pd.DataFrame({"y": ts})
    data["date"] = data.index

    for lag in [1,2,7,14,28]:
        data[f"lag_{lag}"] = data["y"].shift(lag)

    data["roll_mean_7"]  = data["y"].rolling(7).mean()
    data["roll_mean_30"] = data["y"].rolling(30).mean()
    data["roll_std_7"]   = data["y"].rolling(7).std()

    data["dayofweek"] = data["date"].dt.dayofweek
    data["month"]     = data["date"].dt.month
    data["year"]      = data["date"].dt.year
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    data["day_of_month"] = data["date"].dt.day
    data["is_salary_week"] = data["day_of_month"].between(1,7).astype(int)
    data["is_month_end"] = data["day_of_month"].between(25,31).astype(int)
    data["days_from_salary"] = (data["day_of_month"] - 1).abs()
    data["salary_impact"] = np.where(data["days_from_salary"] <= 7, 1/(data["days_from_salary"] + 1), 0)

    data = data.dropna().reset_index(drop=True)

    out = Path("data/features")
    out.mkdir(parents=True, exist_ok=True)
    feature_path = out / "features_timeseries.csv"
    data.to_csv(feature_path, index=False)

    return data
