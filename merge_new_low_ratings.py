"""
Merge newly scraped low-rating reviews into the master raw dataset.

Reads:
  data/raw/tokopedia_reviews_new.csv          — existing master (1514 rows)
  data/raw/tokopedia_reviews_low_ratings_new.csv — deep-scan results (Pass 2)

  (Also merges any reviews from the ORIGINAL low-ratings run if present:
   data/raw/tokopedia_reviews_low_ratings.csv)

Writes (overwrite):
  data/raw/tokopedia_reviews_new.csv          — updated master

After this, re-run notebooks/02_cleaning.ipynb to regenerate the clean dataset.
"""
import sys
import pandas as pd
from pathlib import Path


MASTER       = Path("data/raw/tokopedia_reviews_new.csv")
LOW_NEW      = Path("data/raw/tokopedia_reviews_low_ratings_new.csv")
LOW_OLD      = Path("data/raw/tokopedia_reviews_low_ratings.csv")   # may not exist


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  (skip — not found: {path})")
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
            # Normalise rating to int
            if "rating" in df.columns:
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
            print(f"  Loaded {len(df):>6} rows from {path}")
            return df
        except Exception:
            continue
    print(f"  ERROR reading {path}")
    return pd.DataFrame()


def main():
    print("=== Merge new low-rating reviews into master ===\n")

    dfs = []

    master = load_csv(MASTER)
    if master.empty:
        print("ERROR: master CSV not found — aborting"); sys.exit(1)
    dfs.append(master)

    for extra in [LOW_OLD, LOW_NEW]:
        df = load_csv(extra)
        if not df.empty:
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined (before dedup): {len(combined)} rows")

    # Dedup by review_id; keep first occurrence (from master → newest last)
    combined = combined.drop_duplicates(subset=["review_id"], keep="first")
    print(f"After dedup           : {len(combined)} rows")

    # Print rating distribution
    if "rating" in combined.columns:
        print("\nRating distribution:")
        print(combined["rating"].value_counts().sort_index().to_string())

        low = combined[combined["rating"] <= 3]
        print(f"\nLow-rating rows (1-3★) : {len(low)}")
        print(f"  Rating 1: {(combined['rating']==1).sum()}")
        print(f"  Rating 2: {(combined['rating']==2).sum()}")
        print(f"  Rating 3: {(combined['rating']==3).sum()}")

    # Backup master before overwriting
    backup = MASTER.with_name(MASTER.stem + "_before_low_merge2.csv")
    if not backup.exists():
        master.to_csv(backup, index=False, encoding="utf-8-sig")
        print(f"\nBackup saved → {backup}")

    combined.to_csv(MASTER, index=False, encoding="utf-8-sig")
    print(f"Master updated → {MASTER}  ({len(combined)} rows)")

    print("\n✅ Done.  Next: re-run notebooks/02_cleaning.ipynb")


if __name__ == "__main__":
    main()
