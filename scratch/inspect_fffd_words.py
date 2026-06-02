import csv
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"

fffd_words = Counter()

with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get('Customer Review', '')
        if '\ufffd' in text:
            # find all words containing \ufffd
            words = text.split()
            for w in words:
                if '\ufffd' in w:
                    fffd_words[w] += 1

print("Total unique words containing \ufffd:", len(fffd_words))
print("\nTop 30 words containing \ufffd:")
for w, count in fffd_words.most_common(30):
    print(f"  {w}: {count}")
