import csv
import sys
import re
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"

emoticons = [
    r':\)', r':\(', r':D', r';\)', r':\'\(', r':-D', r':-P', r':P', r'xd', r'xD', r':v', r';_;', r'T_T', r'\(:\)', r'\(:\(', r':o', r':O'
]

emoticon_patterns = {emo: re.compile(emo) for emo in emoticons}
emoticon_counts = Counter()

with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get('Customer Review', '')
        for emo, pattern in emoticon_patterns.items():
            matches = pattern.findall(text)
            if matches:
                emoticon_counts[emo] += len(matches)

print("Emoticon occurrences in raw reviews:")
for emo, count in emoticon_counts.most_common():
    print(f"  {emo}: {count}")
