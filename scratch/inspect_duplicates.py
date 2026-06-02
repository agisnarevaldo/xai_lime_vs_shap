import csv
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"

reviews_map = defaultdict(list)

with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        text = row.get('Customer Review', '').strip()
        emotion = row.get('Emotion', '')
        rating = row.get('Customer Rating', '')
        reviews_map[text].append((idx, emotion, rating))

duplicates_count = 0
conflicting_labels_count = 0

print("=== Duplicate Reviews Analysis ===")
for text, occurrences in reviews_map.items():
    if len(occurrences) > 1:
        duplicates_count += 1
        emotions = set(occ[1] for occ in occurrences)
        if len(emotions) > 1:
            conflicting_labels_count += 1
            if conflicting_labels_count <= 10:
                print(f"Text: {repr(text)}")
                print(f"Occurrences: {occurrences}")
                print("-" * 50)

print(f"Total unique texts with duplicates: {duplicates_count}")
print(f"Duplicates with conflicting emotion labels: {conflicting_labels_count}")
