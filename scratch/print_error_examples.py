import csv
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

error_file = "outputs/finetuning_indobert/reports/error_analysis.csv"

errors = []
with open(error_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        errors.append(row)

by_pair = defaultdict(list)
for e in errors:
    by_pair[(e['true_emotion'], e['pred_emotion'])].append(e)

def print_pair_examples(true_emo, pred_emo, count=5):
    print(f"\n=== Examples of True: {true_emo} | Pred: {pred_emo} (showing {count}) ===")
    pair_errors = by_pair[(true_emo, pred_emo)]
    # sort by confidence
    pair_errors.sort(key=lambda x: float(x['confidence']), reverse=True)
    for i, row in enumerate(pair_errors[:count]):
        print(f"  {i+1}. Text: {row['review_text']}")
        print(f"     Confidence: {float(row['confidence']):.4f}")
        print("-" * 40)

print_pair_examples('love', 'happy')
print_pair_examples('happy', 'love')
print_pair_examples('sadness', 'fear')
print_pair_examples('fear', 'sadness')
