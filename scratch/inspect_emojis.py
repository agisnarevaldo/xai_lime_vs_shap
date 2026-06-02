import csv
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"

non_ascii_counter = Counter()

with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get('Customer Review', '')
        for char in text:
            if ord(char) > 127:
                non_ascii_counter[char] += 1

print("Total non-ASCII characters found:", sum(non_ascii_counter.values()))
print("\nTop 30 non-ASCII characters (likely emojis or special punctuation):")
for char, count in non_ascii_counter.most_common(30):
    print(f"Char: {char} (Unicode: U+{ord(char):04X}) | Count: {count}")
