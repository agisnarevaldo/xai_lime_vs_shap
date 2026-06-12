import json

with open("02_finetuning_indobert_emotion2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    source = "".join(cell.get("source", []))
    if "BertForSequenceClassification.from_pretrained" in source:
        print(f"=== Cell {idx} ===")
        print(source)
        print("-" * 50)
