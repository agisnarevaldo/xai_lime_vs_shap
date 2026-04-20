import pandas as pd
df = pd.read_csv("data/raw/tokopedia_reviews_new.csv")
print("Total rows:", len(df))
print("Rating distribution:")
print(df['rating'].value_counts())
