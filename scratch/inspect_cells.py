import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def inspect_cells(filepath, indices):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for idx in indices:
        if idx < len(nb.get('cells', [])):
            cell = nb['cells'][idx]
            print(f"==================================================")
            print(f"CELL {idx} ({cell.get('cell_type')}):")
            print(f"==================================================")
            source = "".join(cell.get('source', []))
            clean_source = source.encode('ascii', errors='replace').decode('ascii')
            print(clean_source)
            print()

inspect_cells("02_finetuning_indobert_emotion2.ipynb", [2, 4, 8, 12])
