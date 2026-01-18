import json
from collections import Counter

# Load data
with open('data/raw/tokopedia_reviews.json', encoding='utf-8') as f:
    d = json.load(f)

print("=" * 60)
print("ANALYSIS: Expected vs Scraped Reviews")
print("=" * 60)

# Products with expected review counts
print("\nProducts with reviews (expected from page):")
total_expected = 0
for p in d['products']:
    if p.get('review_count'):
        name = (p.get('name') or 'Unknown')[:45]
        count = p['review_count']
        total_expected += count
        print(f"  {name}: {count}")

print(f"\n📊 Total expected: {total_expected}")
print(f"📊 Total scraped: {len(d['reviews'])}")
print(f"📊 Extraction rate: {len(d['reviews'])/total_expected*100:.1f}%" if total_expected > 0 else "")

# Actual reviews per product
print("\n" + "-" * 60)
print("Actual reviews scraped per product:")
reviews_by_product = Counter([r.get('product_name', 'Unknown')[:45] for r in d['reviews']])
for name, count in reviews_by_product.most_common():
    print(f"  {name}: {count}")

print("\n" + "-" * 60)
print("Products that FAILED to scrape (name is null):")
failed = [p for p in d['products'] if not p.get('name')]
print(f"  Failed: {len(failed)} products")
for p in failed[:5]:
    shop = p['url'].split('/')[3]
    print(f"    - {shop}")
