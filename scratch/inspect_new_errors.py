import csv
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

error_file = "outputs/finetuning_indobert/reports/error_analysis.csv"

errors = []
with open(error_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        errors.append(row)

print("Total errors:", len(errors))

patterns = Counter()
for e in errors:
    patterns[(e['true_emotion'], e['pred_emotion'])] += 1

print("\nError patterns (True vs Pred):")
for (t, p), count in patterns.most_common(15):
    print(f"  True: {t:<8} | Pred: {p:<8} | Count: {count}")

print("\nSample high confidence errors:")
errors_sorted = sorted(errors, key=lambda x: float(x['confidence']), reverse=True)
for i, row in enumerate(errors_sorted[:10]):
    print(f"Error {i+1}:")
    print(f"  Text: {row['review_text']}")
    print(f"  True: {row['true_emotion']} | Pred: {row['pred_emotion']} (confidence: {float(row['confidence']):.4f})")
    print("-" * 50)
