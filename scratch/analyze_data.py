import csv
from collections import Counter
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@[\w_]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', ' ', text)
    text = re.sub(r'([!?.,]){2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("=" * 60)
print("TEXT LENGTH ANALYSIS")
print("=" * 60)
with open('data/processed/prdect_train.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

lengths = [len(row['review_clean'].split()) for row in rows]
lengths.sort()
n = len(lengths)
print(f"Train Text length (words) stats:")
print(f"  Min: {min(lengths)}")
print(f"  Max: {max(lengths)}")
print(f"  Avg: {sum(lengths)/n:.1f}")
print(f"  Median: {lengths[n//2]}")
print(f"  P90: {lengths[int(n*0.9)]}")
print(f"  P95: {lengths[int(n*0.95)]}")
print(f"  P99: {lengths[int(n*0.99)]}")
print(f"  >128 words: {sum(1 for l in lengths if l > 128)} ({100*sum(1 for l in lengths if l>128)/n:.1f}%)")
print(f"  >64 words: {sum(1 for l in lengths if l > 64)} ({100*sum(1 for l in lengths if l>64)/n:.1f}%)")
print(f"  >32 words: {sum(1 for l in lengths if l > 32)} ({100*sum(1 for l in lengths if l>32)/n:.1f}%)")
print(f"  <=10 words: {sum(1 for l in lengths if l <= 10)} ({100*sum(1 for l in lengths if l<=10)/n:.1f}%)")

print()
print("=" * 60)
print("SAMPLE SHORT TEXTS")
print("=" * 60)
short = [(r['review_clean'], r['emotion_label'], len(r['review_clean'].split())) for r in rows if len(r['review_clean'].split()) <= 3]
print(f"Samples with <= 3 words: {len(short)}")
for t, e, l in short[:15]:
    print(f"  [{e}] \"{t}\" ({l} words)")

print()
print("=" * 60)
print("DUPLICATE TEXT ANALYSIS")
print("=" * 60)
with open('data/processed/prdect_clean.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

text_labels = [(r['review_clean'], r['emotion_label']) for r in all_rows]
text_count = Counter(t for t, e in text_labels)
dup_texts = {t: c for t, c in text_count.items() if c > 1}
print(f"Total unique texts: {len(text_count)}")
print(f"Texts with duplicates: {len(dup_texts)}")
print(f"Total duplicate instances: {sum(c for c in dup_texts.values()) - len(dup_texts)}")

# Check same text, different labels (noisy labels)
from collections import defaultdict
text_to_labels = defaultdict(set)
for t, e in text_labels:
    text_to_labels[t].add(e)

noisy_texts = {t: lbls for t, lbls in text_to_labels.items() if len(lbls) > 1}
print(f"\nTexts with conflicting labels: {len(noisy_texts)}")
for t, lbls in list(noisy_texts.items())[:5]:
    print(f"  \"{t[:80]}\" -> {lbls}")

print()
print("=" * 60)
print("TRAIN-TEST LEAKAGE CHECK")
print("=" * 60)
train_texts = set(r['review_clean'] for r in rows)
with open('data/processed/prdect_test.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    test_rows = list(reader)
test_texts = set(r['review_clean'] for r in test_rows)
overlap = train_texts & test_texts
print(f"Train texts: {len(train_texts)}")
print(f"Test texts: {len(test_texts)}")
print(f"Overlapping texts (leakage): {len(overlap)}")
if overlap:
    for t in list(overlap)[:5]:
        print(f"  \"{t[:80]}\"")

print()
print("=" * 60)
print("EMOJI / SPECIAL CHARACTER CHECK")
print("=" * 60)
with open('data/processed/prdect_train.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

has_emoji = [r for r in rows if re.search(r'[^\x00-\x7F]', r['review_clean'])]
print(f"Rows with non-ASCII in review_clean: {len(has_emoji)}")
for r in has_emoji[:5]:
    print(f"  [{r['emotion_label']}] \"{r['review_clean'][:100]}\"")

print()
print("=" * 60)
print("CLASS IMBALANCE RATIO")
print("=" * 60)
train_dist = Counter(r['emotion_label'] for r in rows)
max_count = max(train_dist.values())
min_count = min(train_dist.values())
print(f"Imbalance ratio (max/min): {max_count/min_count:.2f}x")
for cls, cnt in sorted(train_dist.items(), key=lambda x: -x[1]):
    pct = 100 * cnt / sum(train_dist.values())
    print(f"  {cls:10s}: {cnt:5d} ({pct:.1f}%)")
