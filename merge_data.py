"""
Merge old backup data with new scraped data
"""
import json
import pandas as pd
import os

# Paths
backup_json = 'data/raw/tokopedia_reviews_backup.json'
new_json = 'data/raw/tokopedia_reviews.json'
output_csv = 'data/raw/tokopedia_reviews.csv'
output_json = 'data/raw/tokopedia_reviews.json'

# Load backup
print("Loading backup data...")
with open(backup_json, 'r', encoding='utf-8') as f:
    backup_data = json.load(f)
backup_reviews = backup_data.get('reviews', [])
print(f"  Backup: {len(backup_reviews)} reviews")

# Load new
print("Loading new data...")
with open(new_json, 'r', encoding='utf-8') as f:
    new_data = json.load(f)
new_reviews = new_data.get('reviews', [])
print(f"  New: {len(new_reviews)} reviews")

# Combine and deduplicate
print("\nMerging and deduplicating...")
all_reviews = backup_reviews + new_reviews
seen = set()
unique_reviews = []

for r in all_reviews:
    rid = r.get('review_id', '')
    if rid and rid not in seen:
        seen.add(rid)
        unique_reviews.append(r)

print(f"  Combined: {len(all_reviews)} -> {len(unique_reviews)} unique")

# Products - combine
all_products = backup_data.get('products', []) + new_data.get('products', [])
seen_urls = set()
unique_products = []
for p in all_products:
    url = p.get('url', '')
    if url and url not in seen_urls:
        seen_urls.add(url)
        unique_products.append(p)

# Save JSON
merged_data = {
    'products': unique_products,
    'reviews': unique_reviews
}

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)
print(f"\n✅ Saved JSON: {output_json}")

# Save CSV
df = pd.DataFrame(unique_reviews)
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"✅ Saved CSV: {output_csv}")

# Stats
print(f"\n{'='*50}")
print("📊 FINAL MERGED STATISTICS")
print(f"{'='*50}")
print(f"Total products: {len(unique_products)}")
print(f"Total reviews: {len(unique_reviews)}")
print(f"Unique reviewers: {df['reviewer_name'].nunique()}")

print("\n📦 Reviews per Product (top 10):")
print(df.groupby('product_name').size().sort_values(ascending=False).head(10))
