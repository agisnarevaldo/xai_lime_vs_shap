import csv
import sys
import random
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"

rows = []
with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

random.seed(42)

# 1. Analyze Duplicates and Conflicting Labels
text_to_rows = defaultdict(list)
for r in rows:
    text_to_rows[r['Customer Review'].strip()].append(r)

conflicting_dupes = []
consistent_dupes = []
for text, group in text_to_rows.items():
    if len(group) > 1:
        emotions = set(r['Emotion'] for r in group)
        if len(emotions) > 1:
            conflicting_dupes.append((text, group))
        else:
            consistent_dupes.append((text, group))

print(f"=== 1. DUPLICATE AND CONFLICTING LABELS ===")
print(f"Total reviews in dataset: {len(rows)}")
print(f"Unique reviews: {len(text_to_rows)}")
print(f"Total duplicates: {len(rows) - len(text_to_rows)}")
print(f"Duplicates with identical labels: {len(consistent_dupes)}")
print(f"Duplicates with conflicting labels: {len(conflicting_dupes)}")

print("\nSample Conflicting Duplicates:")
for i, (text, group) in enumerate(conflicting_dupes[:5]):
    print(f"  {i+1}. Text: {repr(text)}")
    for r in group:
        print(f"     Rating: {r['Customer Rating']} | Emotion: {r['Emotion']} | Category: {r['Category']}")
    print("-" * 50)

# 2. Keyword Association Audit
# Let's check how specific emotional keywords are distributed across labels
keywords = {
    'kecewa': 'disappointed (often Sadness/Anger)',
    'aman': 'safe (often Happy/Love, but check if related to Fear)',
    'suka': 'like (often Love/Happy)',
    'sayang': 'unfortunately/love',
    'takut': 'afraid (Fear)',
    'khawatir': 'worried (Fear)',
    'marah': 'angry (Anger)',
    'kesal': 'irritated (Anger)',
    'puas': 'satisfied (Happy/Love)',
    'kurang': 'less/lacking (Sadness/Anger)'
}

print("\n=== 2. KEYWORD DISTRIBUTION AUDIT ===")
for kw, desc in keywords.items():
    kw_counts = Counter()
    total_matches = 0
    examples = []
    for r in rows:
        text = r['Customer Review'].lower()
        if kw in text:
            kw_counts[r['Emotion']] += 1
            total_matches += 1
            if len(examples) < 2:
                examples.append((r['Customer Review'], r['Emotion'], r['Customer Rating']))
                
    print(f"Keyword: '{kw}' ({desc}) -> Total matches: {total_matches}")
    for emo, count in kw_counts.most_common():
        pct = (count / total_matches) * 100
        print(f"  - {emo}: {count} ({pct:.1f}%)")
    print("  Sample texts:")
    for ex in examples:
        print(f"    * [{ex[1]}, Rating {ex[2]}] {repr(ex[0])}")
    print("-" * 50)

# 3. Rating vs Emotion Sanity Check
# Let's inspect Rating 5: Happy vs Love
print("\n=== 3. RATING 5: HAPPY VS LOVE COMPARISON ===")
rating_5_happy = [r for r in rows if r['Customer Rating'] == '5' and r['Emotion'] == 'Happy']
rating_5_love = [r for r in rows if r['Customer Rating'] == '5' and r['Emotion'] == 'Love']
print(f"Total 5-Star Happy: {len(rating_5_happy)}")
print(f"Total 5-Star Love : {len(rating_5_love)}")

print("\nSample 5-Star Happy:")
for r in random.sample(rating_5_happy, min(5, len(rating_5_happy))):
    print(f"  - {repr(r['Customer Review'])}")

print("\nSample 5-Star Love:")
for r in random.sample(rating_5_love, min(5, len(rating_5_love))):
    print(f"  - {repr(r['Customer Review'])}")

# 4. Rating 1: Anger vs Fear vs Sadness Comparison
print("\n=== 4. RATING 1: NEGATIVE EMOTIONS COMPARISON ===")
rating_1_anger = [r for r in rows if r['Customer Rating'] == '1' and r['Emotion'] == 'Anger']
rating_1_fear = [r for r in rows if r['Customer Rating'] == '1' and r['Emotion'] == 'Fear']
rating_1_sadness = [r for r in rows if r['Customer Rating'] == '1' and r['Emotion'] == 'Sadness']
print(f"Total 1-Star Anger  : {len(rating_1_anger)}")
print(f"Total 1-Star Fear   : {len(rating_1_fear)}")
print(f"Total 1-Star Sadness: {len(rating_1_sadness)}")

print("\nSample 1-Star Anger:")
for r in random.sample(rating_1_anger, min(4, len(rating_1_anger))):
    print(f"  - {repr(r['Customer Review'])}")

print("\nSample 1-Star Fear:")
for r in random.sample(rating_1_fear, min(4, len(rating_1_fear))):
    print(f"  - {repr(r['Customer Review'])}")

print("\nSample 1-Star Sadness:")
for r in random.sample(rating_1_sadness, min(4, len(rating_1_sadness))):
    print(f"  - {repr(r['Customer Review'])}")
