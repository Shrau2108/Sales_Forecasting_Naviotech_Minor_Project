# src/data_prep.py
import pandas as pd
from pathlib import Path

def load_and_clean_csv(filepath: str):
    """
    Load CSV with direct path (e.g. "data/raw/train.csv"), detect date & numeric target.
    Returns: df (cleaned), date_col_name, target_col_name
    Side-effect: saves cleaned CSV to data/processed/cleaned_<origname>.csv
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()

    # Detect date column
    date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    date_col = None
    if date_candidates:
        date_col = date_candidates[0]
    else:
        # fallback: choose first column with >70% datetime-parseable values
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    date_col = c
                    break
            except Exception:
                continue
    if date_col is None:
        raise ValueError("Could not auto-detect a date column. Rename a column to include 'date' or 'time'.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).reset_index(drop=True)

    # Detect numeric target
    numeric_cols = df.select_dtypes(include=["int","float"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found to use as target. Ensure sales/amount column is numeric.")
    # Prefer columns with 'sale'/'revenue'/'amount' in name
    target_col = None
    for t in ["sale","revenue","amount","order","gmv","transaction","total"]:
        for col in numeric_cols:
            if t in col.lower():
                target_col = col
                break
        if target_col:
            break
    if target_col is None:
        # fallback: choose numeric column with highest variance
        target_col = df[numeric_cols].var().sort_values(ascending=False).index[0]

    # sort chronologically
    df = df.sort_values(date_col).reset_index(drop=True)

    # drop exact duplicates on (date, target)
    df = df.drop_duplicates(subset=[date_col, target_col])

    # save cleaned
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    cleaned_name = out / f"cleaned_{p.name}"
    df.to_csv(cleaned_name, index=False)

    return df, date_col, target_col
