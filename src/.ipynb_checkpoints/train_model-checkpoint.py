# src/train_model.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.data_prep import load_and_clean_csv
from src.feature_engineering import build_time_series_features

def train_model(csv_path: str, force_retrain: bool = True):
    """
    Trains an XGBoost model end-to-end.
    Input: csv_path e.g. "data/raw/train.csv"
    Side-effects:
      - Saves features to data/features/features_timeseries.csv
      - Saves model to models/final_xgb_model.pkl
      - Prints MAE/RMSE
    """
    df, date_col, target_col = load_and_clean_csv(csv_path)
    data = build_time_series_features(df, date_col, target_col)

    feature_cols = [c for c in data.columns if c not in ["y","date"]]
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test  = data.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["y"]
    X_test,  y_test  = test[feature_cols],  test["y"]

    # naive baseline: using lag_1
    mae_naive = mean_absolute_error(y_test, test["lag_1"])
    rmse_naive = mean_squared_error(y_test, test["lag_1"], squared=False)

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # save
    Path("models").mkdir(parents=True, exist_ok=True)
    model_path = Path("models") / "final_xgb_model.pkl"
    joblib.dump(model, model_path)

    print("✅ Training complete")
    print(f"Model saved to: {model_path}")
    print(f"MAE: {mae:.2f} (Naive: {mae_naive:.2f})")
    print(f"RMSE: {rmse:.2f} (Naive: {rmse_naive:.2f})")

    return model
