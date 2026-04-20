import pandas as pd
import numpy as np

def refine_dataset_count(file_path):
    df = pd.read_csv(file_path)
    total_current = len(df)
    print(f"Current total rows: {total_current}")

    # Determine original count (User wants it to be the same as 'scraping result')
    # Previous count was 1561.
    TARGET_COUNT = 1561 
    DIFF = total_current - TARGET_COUNT # Should be around 80
    
    if DIFF <= 0:
        print("Dataset size is already at or below target. No need to drop rows.")
        return

    print(f"Need to drop {DIFF} 'Positif' rows to return to original count.")

    # Identify original 'Positif' rows to drop
    # We want to keep the NEWLY added rows (which are mostly Netral/Negatif)
    # and drop some OLD Positif rows.
    
    # Let's filter for Rating 5.0
    mask_rating_5 = (df['rating'] == 5) | (df['rating'] == 5.0)
    df_pos = df[mask_rating_5]
    df_other = df[~mask_rating_5]
    
    if len(df_pos) < DIFF:
        print(f"Error: Not enough Positif rows to drop ({len(df_pos)} < {DIFF})")
        return

    # Randomly sample DIFF rows to drop from the Positif set
    # Using a fixed seed for reproducibility if needed, or just random
    rows_to_drop = df_pos.sample(n=DIFF, random_state=42).index
    
    df_final = df.drop(rows_to_drop)
    
    print(f"Final total rows: {len(df_final)}")
    
    # Save back to CSV
    df_final.to_csv(file_path, index=False)
    print(f"Done. Dataset {file_path} refined.")

if __name__ == "__main__":
    DATA_PATH = "data/raw/tokopedia_reviews_new.csv"
    refine_dataset_count(DATA_PATH)
