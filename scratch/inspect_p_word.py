import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"
with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get('Customer Review', '')
        if 'p\ufffdbyran' in text or 'p\ufffdb' in text or 'p\ufffd' in text:
            if 'p\ufffd' in text:
                print(f"Text: {text}")
