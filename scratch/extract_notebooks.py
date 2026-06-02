import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_cells(filepath, indices):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for idx in indices:
        if idx >= len(nb.get('cells', [])):
            print(f"Index {idx} out of range")
            continue
        cell = nb.get('cells', [])[idx]
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        source_str = "".join(source)
        print(f"=== Cell {idx} ({cell_type}) ===")
        print(source_str)
        print("-" * 60)

get_cells("02_finetuning_indobert_emotion2.ipynb", [16, 17])
