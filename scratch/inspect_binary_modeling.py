import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def inspect_notebook_cells(filepath):
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n==================================================")
    print(f"Notebook: {filepath}")
    print(f"Total cells: {len(nb['cells'])}")
    print(f"==================================================")
    
    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        
        # We target cells loading CSVs, saving models, or showing final test metrics
        keywords = ["read_csv", "save_pretrained", "accuracy", "Confusion Matrix", "Classification Report", "tokopedia_reviews_binary"]
        found = [kw for kw in keywords if kw in source]
        
        outputs = cell.get("outputs", [])
        out_text = ""
        for out in outputs:
            if out.get("output_type") == "stream":
                out_text += "".join(out.get("text", []))
            elif out.get("output_type") == "execute_result":
                out_text += "".join(out.get("data", {}).get("text/plain", []))
                
        out_found = [kw for kw in keywords if kw in out_text]
        
        if found or out_found:
            print(f"\n--- Cell {idx} ({cell_type}) ---")
            print(source[:400])
            if out_text:
                print(">> Outputs:")
                print(out_text[:600] + ("..." if len(out_text) > 600 else ""))
            print("-" * 50)

inspect_notebook_cells("notebooks/03_modeling_binary.ipynb")
inspect_notebook_cells("notebooks/04a_modeling_binary.ipynb")
inspect_notebook_cells("notebooks/03_modeling_binary-collab.ipynb")
