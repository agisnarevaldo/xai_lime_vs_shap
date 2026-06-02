import csv
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"

categories = Counter()

with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cat = row.get('Category')
        if cat:
            categories[cat] += 1

print("Category distribution in raw data:")
for cat, count in categories.most_common():
    print(f"  {cat}: {count}")
