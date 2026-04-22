
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import re

# Setup paths
RAW_PATH = 'd:/xai_lime_vs_shap/data/raw/tokopedia_reviews_new.csv'
CLEAN_PATH = 'd:/xai_lime_vs_shap/data/processed/tokopedia_reviews_clean.csv'
BALANCED_PATH = 'd:/xai_lime_vs_shap/data/processed/tokopedia_reviews_balanced.csv'
FIGURES_DIR = 'd:/xai_lime_vs_shap/outputs/figures/'

os.makedirs(FIGURES_DIR, exist_ok=True)

# Helper for cleaning (similar to cleaner.py)
def quick_clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 1. TABLE 4.2 - Structural Cleaning Stats
df_raw = pd.read_csv(RAW_PATH)
c1 = len(df_raw)
df_dedupe = df_raw.drop_duplicates(subset=['review_id'])
c2 = len(df_dedupe)
df_no_empty = df_dedupe[df_dedupe['review_text'].notna() & (df_dedupe['review_text'].str.strip().str.len() > 0)]
c3 = len(df_no_empty)
df_valid_rating = df_no_empty[df_no_empty['rating'].astype(float).between(1, 5)]
c4 = len(df_valid_rating)

print(f"Table 4.2 Stats:")
print(f"Data awal: {c1}")
print(f"Deduplikasi: {c2}")
print(f"Hapus data kosong: {c3}")
print(f"Validasi rating: {c4}")

# 2. TABLE 4.3 - Labeling (Before Balancing)
def map_sentiment(r):
    r = int(float(r))
    if r >= 4: return "Positif"
    elif r == 3: return "Netral"
    else: return "Negatif"

df_valid_rating['sentiment_label'] = df_valid_rating['rating'].apply(map_sentiment)
dist = df_valid_rating['sentiment_label'].value_counts()
total = len(df_valid_rating)

print("\nTable 4.3 Stats (Before Balancing):")
for label in ['Positif', 'Netral', 'Negatif']:
    count = dist.get(label, 0)
    pct = (count / total) * 100
    print(f"{label}: {count} ({pct:.1f}%)")

# FIGURE 4.1 - Bar Chart Before Balancing
plt.figure(figsize=(8, 5))
sns.countplot(data=df_valid_rating, x='sentiment_label', order=['Positif', 'Netral', 'Negatif'], palette='viridis')
plt.title('Distribusi Label Sentimen (Sebelum Penyeimbangan)')
plt.ylabel('Jumlah Ulasan')
plt.xlabel('Sentimen')
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_1_distribution_before.png'))
plt.close()

# 3. TABLE 4.4 - Examples
sample_text = "Barangnya bagus banget!!!"
print("\nTable 4.4 Examples:")
print(f"Asli: {sample_text}")
print(f"Case Folding: {sample_text.lower()}")
print(f"Cleaning: {quick_clean(sample_text)}")
print(f"Final: {quick_clean(sample_text)}")

# FIGURE 4.2 & 4.3 - WordClouds
text_before = " ".join(df_valid_rating['review_text'].fillna("").astype(str))
wc_before = WordCloud(width=800, height=400, background_color='white').generate(text_before)
wc_before.to_file(os.path.join(FIGURES_DIR, 'fig4_2_wordcloud_before.png'))

df_valid_rating['clean_text'] = df_valid_rating['review_text'].apply(quick_clean)
text_after = " ".join(df_valid_rating['clean_text'])
wc_after = WordCloud(width=800, height=400, background_color='white').generate(text_after)
wc_after.to_file(os.path.join(FIGURES_DIR, 'fig4_3_wordcloud_after.png'))

# 4. TABLE 4.5 - After Balancing
# We read the balanced file if it exists, otherwise we simulate or use the real one
if os.path.exists(BALANCED_PATH):
    df_balanced = pd.read_csv(BALANCED_PATH)
else:
    # Fallback/Mock for calculation if precisely needed
    df_balanced = df_valid_rating.copy() # Placeholder if not run yet

dist_bal = df_balanced['sentiment_label'].value_counts()
print("\nTable 4.5 Stats (After Balancing):")
for label in ['Positif', 'Netral', 'Negatif']:
    before = dist.get(label, 0)
    after = dist_bal.get(label, 0)
    print(f"{label}: {before} -> {after}")

# FIGURE 4.4 - Bar Chart After Balancing
plt.figure(figsize=(8, 5))
sns.countplot(data=df_balanced, x='sentiment_label', order=['Positif', 'Netral', 'Negatif'], palette='magma')
plt.title('Distribusi Label Sentimen (Setelah Penyeimbangan)')
plt.ylabel('Jumlah Ulasan')
plt.xlabel('Sentimen')
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_4_distribution_after.png'))
plt.close()

print("\nTable 4.6 Summary:")
print(f"Data awal: {c1}")
print(f"Setelah cleaning: {len(df_valid_rating)}")
print(f"Setelah labeling: {len(df_valid_rating)}")
print(f"Setelah balancing: {len(df_balanced)}")
