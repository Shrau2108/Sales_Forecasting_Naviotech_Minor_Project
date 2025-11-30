import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ====================================
# PATHS (DIRECT, NO CONFIG NEEDED)
# ====================================

RAW_DIR      = Path("data/raw")
FEATURE_DIR  = Path("data/features")
MODEL_DIR    = Path("models")

MODEL_PATH   = MODEL_DIR / "final_xgboost_forecasting_model.pkl"
HIST_PATH    = FEATURE_DIR / "historical_full_features.csv"
LOG_PATH     = MODEL_DIR / "training_log.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ====================================
# 1. Load latest RAW CSV
# ====================================

def get_latest_raw_csv():
    files = list(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("❌ No CSV found in data/raw/")
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return latest

raw_file = get_latest_raw_csv()
print(f"📥 Using latest dataset → {raw_file}")

df = pd.read_csv(raw_file)
df.columns = df.columns.str.strip()

# ====================================
# 2. Detect DATE column
# ====================================
date_col = None
for c in df.columns:
    if "date" in c.lower():
        date_col = c
        break

if date_col is None:
    raise ValueError("❌ No DATE column found.")

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

# ====================================
# 3. Detect TARGET column (Order_Comp)
# ====================================

if "Order_Comp" not in df.columns:
    raise ValueError("❌ Target column 'Order_Comp' not found.")

TARGET = "Order_Comp"

# daily aggregated series
ts = df.groupby(date_col)[TARGET].sum().asfreq("D", fill_value=0)

data = pd.DataFrame({"y": ts})
data["date"] = data.index

# ====================================
# 4. Feature Engineering
# ====================================

# Lags
for lag in [1,2,7,14,28]:
    data[f"lag_{lag}"] = data["y"].shift(lag)

# Rolling windows
data["roll_mean_7"]  = data["y"].rolling(7).mean()
data["roll_mean_30"] = data["y"].rolling(30).mean()
data["roll_std_7"]   = data["y"].rolling(7).std()

# Calendar features
data["dayofweek"]  = data.date.dt.dayofweek
data["month"]      = data.date.dt.month
data["year"]       = data.date.dt.year
data["is_weekend"] = (data.dayofweek >= 5).astype(int)

# Salary-cycle features
data["day_of_month"]   = data.date.dt.day
data["is_salary_week"] = data.day_of_month.between(1,7).astype(int)
data["is_month_end"]   = data.day_of_month.between(25,31).astype(int)
data["days_from_salary"] = (data.day_of_month - 1).abs()
data["salary_impact"] = np.where(
    data.days_from_salary <= 7,
    1/(data.days_from_salary+1),
    0
)

# Remove rows where features incomplete
data = data.dropna().reset_index(drop=True)

feature_cols = [c for c in data.columns if c not in ["y","date"]]

# ====================================
# 5. Time-based Split
# ====================================

split_date = data["date"].quantile(0.8)

train = data[data["date"] < split_date]
test  = data[data["date"] >= split_date]

X_train, y_train = train[feature_cols], train["y"]
X_test,  y_test  = test[feature_cols],  test["y"]

# ====================================
# 6. Train Advanced XGBoost
# ====================================

model = XGBRegressor(
    n_estimators=2500,
    learning_rate=0.035,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    tree_method="hist"
)

print("🚀 Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n=======================")
print("   🔁 RETRAINING DONE")
print("=======================\n")
print(f"MAE : {mae}")
print(f"RMSE: {rmse}\n")

# ====================================
# 7. Save updated model + features
# ====================================

joblib.dump(model, MODEL_PATH)
print(f"💾 Updated model saved → {MODEL_PATH}")

data.to_csv(HIST_PATH, index=False)
print(f"📚 Latest full feature dataset saved → {HIST_PATH}")

# ====================================
# 8. Log training metrics
# ====================================

log_row = pd.DataFrame([{
    "timestamp": datetime.now().isoformat(),
    "dataset": raw_file.name,
    "mae": mae,
    "rmse": rmse,
    "data_points": len(data),
    "train_rows": len(train),
    "test_rows": len(test)
}])

if LOG_PATH.exists():
    prev = pd.read_csv(LOG_PATH)
    pd.concat([prev, log_row], ignore_index=True).to_csv(LOG_PATH, index=False)
else:
    log_row.to_csv(LOG_PATH, index=False)

print(f"📊 Metrics logged → {LOG_PATH}")
