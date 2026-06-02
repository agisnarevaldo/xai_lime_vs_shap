import csv
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

errors = []
with open("outputs/finetuning_indobert/reports/error_analysis.csv", mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        errors.append(row)

print("Total errors:", len(errors))

# count patterns
patterns = Counter()
for e in errors:
    patterns[(e['true_emotion'], e['pred_emotion'])] += 1

print("\nError distribution (True Label vs Pred Label):")
for (true_emo, pred_emo), count in patterns.most_common(15):
    print(f"True: {true_emo} | Pred: {pred_emo} | Count: {count}")

print("\nTop 10 high confidence errors:")
errors_sorted = sorted(errors, key=lambda x: float(x['confidence']), reverse=True)
for row in errors_sorted[:10]:
    print(f"Text: {row['review_text']}")
    print(f"True: {row['true_emotion']} | Pred: {row['pred_emotion']} (confidence: {float(row['confidence']):.4f})")
    print("-" * 50)
