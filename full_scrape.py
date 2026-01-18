"""
Full scraping script for all iPhone 17 URLs
Expected: ~2000 reviews in ~30-45 minutes
"""
import sys
import json
import os

sys.path.insert(0, '.')
from src.scraping.runner import run_scraper

# Load all URLs
with open('link.txt', 'r', encoding='utf-8') as f:
    all_urls = [line.strip() for line in f if line.strip().startswith('http')]

print(f"Total URLs: {len(all_urls)}")

# Prioritize high-review stores
priority_stores = ['ismile-indonesia', 'milano-cell', 'senz88', 'digitechmall', 
                   'ibox-official', 'digimap-official-store', 'collinsofficial', 'distriponsel']
priority_urls = [u for u in all_urls if any(s in u for s in priority_stores)]
regular_urls = [u for u in all_urls if u not in priority_urls]

print(f"Priority: {len(priority_urls)} URLs")
print(f"Regular: {len(regular_urls)} URLs")

# STEP 1: Priority scraping
print("\n" + "="*60)
print("STEP 1: PRIORITY SCRAPING")
print("="*60)
priority_result = run_scraper(priority_urls, max_reviews=800)
print(f"\nPriority result: {len(priority_result['reviews'])} reviews")

# STEP 2: Regular scraping  
print("\n" + "="*60)
print("STEP 2: REGULAR SCRAPING")
print("="*60)
regular_result = run_scraper(regular_urls, max_reviews=200)
print(f"\nRegular result: {len(regular_result['reviews'])} reviews")

# Combine
all_reviews = priority_result.get('reviews', []) + regular_result.get('reviews', [])
all_products = priority_result.get('products', []) + regular_result.get('products', [])

# Deduplicate
seen = set()
unique_reviews = []
for r in all_reviews:
    rid = r.get('review_id', '')
    if rid and rid not in seen:
        seen.add(rid)
        unique_reviews.append(r)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Total products: {len(all_products)}")
print(f"Total reviews (raw): {len(all_reviews)}")
print(f"Total reviews (unique): {len(unique_reviews)}")

# Save
output_data = {
    'products': all_products,
    'reviews': unique_reviews
}

with open('data/raw/tokopedia_reviews_new.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

import pandas as pd
df = pd.DataFrame(unique_reviews)
df.to_csv('data/raw/tokopedia_reviews_new.csv', index=False, encoding='utf-8')

print(f"\nSaved to:")
print(f"  - data/raw/tokopedia_reviews_new.json")
print(f"  - data/raw/tokopedia_reviews_new.csv")
