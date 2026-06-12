import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=== Audit Processed CSV Files ===")
for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
        print(f"\nFile: {csv_path.name}")
        print(f"Shape: {df.shape}")
        print("Columns:", list(df.columns))
        # Find if there is a label column
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'sentiment' in c.lower() or 'emotion' in c.lower()]
        for col in label_cols:
            print(f"Distribution of '{col}':")
            print(df[col].value_counts(dropna=False))
    except Exception as e:
        print(f"Error reading {csv_path.name}: {e}")
