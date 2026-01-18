"""
Merge all available review data and deduplicate
"""
import json
import pandas as pd

print("="*60)
print("MERGING ALL REVIEW DATA")
print("="*60)

# Load all data sources
sources = []

# 1. Backup data
try:
    with open('data/raw/tokopedia_reviews_backup.json', 'r', encoding='utf-8') as f:
        backup = json.load(f)
    sources.append(('Backup', backup.get('reviews', [])))
    print(f"1. Backup: {len(backup.get('reviews', []))} reviews")
except:
    print("1. Backup: not found")

# 2. New scraped data
try:
    with open('data/raw/tokopedia_reviews_new.json', 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    sources.append(('New', new_data.get('reviews', [])))
    print(f"2. New: {len(new_data.get('reviews', []))} reviews")
except:
    print("2. New: not found")

# 3. Original data (if different from backup)
try:
    with open('data/raw/tokopedia_reviews.json', 'r', encoding='utf-8') as f:
        original = json.load(f)
    sources.append(('Original', original.get('reviews', [])))
    print(f"3. Original: {len(original.get('reviews', []))} reviews")
except:
    print("3. Original: not found")

# Combine all
all_reviews = []
for name, reviews in sources:
    all_reviews.extend(reviews)

print(f"\nTotal raw: {len(all_reviews)} reviews")

# Deduplicate by review_id
seen = set()
unique_reviews = []
for r in all_reviews:
    rid = r.get('review_id', '')
    if rid and rid not in seen:
        seen.add(rid)
        unique_reviews.append(r)

print(f"After dedup: {len(unique_reviews)} unique reviews")

# Save merged data
merged = {
    'products': [],
    'reviews': unique_reviews
}

with open('data/raw/tokopedia_reviews_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(unique_reviews)
df.to_csv('data/raw/tokopedia_reviews_merged.csv', index=False, encoding='utf-8')

print(f"\nSaved to:")
print(f"  - data/raw/tokopedia_reviews_merged.json")
print(f"  - data/raw/tokopedia_reviews_merged.csv")

print(f"\n{'='*60}")
print(f"RESULT: {len(unique_reviews)} unique reviews")
print(f"{'='*60}")

if len(unique_reviews) >= 2000:
    print("TARGET REACHED!")
else:
    print(f"Need {2000 - len(unique_reviews)} more reviews to reach 2000")
