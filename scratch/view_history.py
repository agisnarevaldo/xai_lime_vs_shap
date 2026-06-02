import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

history_path = "outputs/finetuning_indobert/reports/training_history.json"
try:
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    print("Training History keys:", history.keys())
except Exception as e:
    print("Failed to load json history:", e)
    # try csv
    import csv
    with open("outputs/finetuning_indobert/reports/training_history.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            print(row)
            if i >= 15:
                break
