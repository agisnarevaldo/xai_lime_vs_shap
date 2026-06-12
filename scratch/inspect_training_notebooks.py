import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def find_keywords_in_notebook(filepath, keywords):
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    print(f"\n==================================================")
    print(f"Notebook: {filepath}")
    print(f"==================================================")
    
    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        
        found = [kw for kw in keywords if kw in source]
        if found:
            print(f"Cell {idx} ({cell_type}) matches {found}:")
            print(source[:800])
            outputs = cell.get("outputs", [])
            if outputs:
                print(">> Outputs:")
                for out in outputs:
                    out_type = out.get("output_type")
                    if out_type == "stream":
                        text = "".join(out.get("text", []))
                        print(text[:300] + ("..." if len(text) > 300 else ""))
                    elif out_type == "execute_result":
                        data = out.get("data", {})
                        text = "".join(data.get("text/plain", []))
                        print(text[:300] + ("..." if len(text) > 300 else ""))
            print("-" * 50)

keywords = ["read_csv", "save_pretrained", "OUTPUT_DIR", "DATA_PATH", "train.csv", "test.csv", "prdect", "tokopedia"]

print("--- Auditing Preprocessing & Training Notebooks in Root ---")
find_keywords_in_notebook("02_preprocessing_prdect.ipynb", keywords)
find_keywords_in_notebook("02_finetuning_indobert_emotion2.ipynb", keywords)

print("\n--- Auditing Preprocessing & Training Notebooks in notebooks/ ---")
find_keywords_in_notebook("notebooks/02_preprocessing_prdect.ipynb", keywords)
find_keywords_in_notebook("notebooks/02_finetuning_indobert_emotion2.ipynb", keywords)
