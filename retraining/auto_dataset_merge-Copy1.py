import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

def merge_all_raw():
    files = list(RAW_DIR.glob("*.csv"))
    if not files:
        print("No files to merge.")
        return

    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)

    merged.to_csv(RAW_DIR / "merged_master_data.csv", index=False)
    print("Merged dataset saved as merged_master_data.csv")

if __name__ == "__main__":
    merge_all_raw()
