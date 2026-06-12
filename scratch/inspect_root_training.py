import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def inspect_notebook_cells(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"==================================================")
    print(f"Notebook: {filepath}")
    print(f"Total cells: {len(nb['cells'])}")
    print(f"==================================================")
    
    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        
        # Check if cell has model training, paths, data loading, or evaluation outputs
        keywords = ["read_csv", "save_pretrained", "OUTPUT_DIR", "prdect", "1057", "98.2", "Accuracy"]
        found = [kw for kw in keywords if kw in source]
        
        # We also want to look at specific training results printed in the outputs
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
            print(source[:500])
            if out_text:
                print(">> Outputs:")
                print(out_text[:1000] + ("..." if len(out_text) > 1000 else ""))
            print("-" * 50)

inspect_notebook_cells("02_finetuning_indobert_emotion2.ipynb")
