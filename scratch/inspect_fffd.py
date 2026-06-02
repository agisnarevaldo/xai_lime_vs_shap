import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"
count = 0
with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get('Customer Review', '')
        if '\ufffd' in text:
            print(f"Rating: {row['Customer Rating']} | Emotion: {row['Emotion']} | Text: {text}")
            count += 1
            if count >= 10:
                break
