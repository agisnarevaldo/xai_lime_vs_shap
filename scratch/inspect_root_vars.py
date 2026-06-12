import json

import json

with open("02_finetuning_indobert_emotion2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(1, 6):
    cell = nb['cells'][idx]
    print(f"=== Cell {idx} ({cell['cell_type']}) ===")
    print("".join(cell.get("source", [])))
    print("-" * 50)

