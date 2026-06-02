import csv
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"

reviews = []
unique_reviews = set()

with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get('Customer Review', '').strip()
        reviews.append(text)
        unique_reviews.add(text)

print("Total raw rows:", len(reviews))
print("Total unique reviews (by text):", len(unique_reviews))
print("Total duplicate rows:", len(reviews) - len(unique_reviews))
