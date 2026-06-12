import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding
sys.stdout.reconfigure(encoding='utf-8')

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

nb_path = Path("02_preprocessing_prdect.ipynb")

if nb_path.exists():
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    print(f"==================================================")
    print(f"Notebook: {nb_path}")
    print(f"==================================================")
    
    for idx, cell in enumerate(nb.get("cells", [])):
        if idx == 8:
            source = "".join(cell.get("source", []))
            print(f"\n--- Cell {idx} ({cell.get('cell_type')}) ---")
            print(source)
            print("-" * 50)
else:
    print(f"Notebook not found at: {nb_path}")


