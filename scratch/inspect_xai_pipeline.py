import json
import sys

# Ensure UTF-8 output encoding
sys.stdout.reconfigure(encoding='utf-8')

import json
import sys

# Ensure UTF-8 output encoding
sys.stdout.reconfigure(encoding='utf-8')

def inspect_notebook(filepath, start_cell, end_cell):
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    print(f"==================================================")
    print(f"Notebook: {filepath}")
    print(f"Displaying cells from {start_cell} to {end_cell}")
    print(f"==================================================")
    
    for idx in range(start_cell, min(end_cell + 1, len(nb['cells']))):
        cell = nb['cells'][idx]
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        print(f"\n--- Cell {idx} ({cell_type}) ---")
        print(source)
        
        # Print outputs if any and if it's a code cell
        outputs = cell.get("outputs", [])
        if outputs:
            print(">> Outputs:")
            for out in outputs:
                out_type = out.get("output_type")
                if out_type == "stream":
                    text = "".join(out.get("text", []))
                    print(text[:1000] + ("..." if len(text) > 1000 else ""))
                elif out_type == "execute_result":
                    data = out.get("data", {})
                    text = "".join(data.get("text/plain", []))
                    print(text[:1000] + ("..." if len(text) > 1000 else ""))
                elif out_type == "error":
                    ename = out.get("ename")
                    evalue = out.get("evalue")
                    traceback = "\n".join(out.get("traceback", []))
                    print(f"ERROR: {ename} - {evalue}\n{traceback[:1000]}")
        print("-" * 50)

inspect_notebook("notebooks/04c_explainability_binary_pipeline.ipynb", 3, 10)


