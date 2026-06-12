import json
import os
from pathlib import Path

def search_notebooks(directory, queries):
    p = Path(directory)
    notebooks = list(p.glob("**/*.ipynb"))
    
    print(f"Searching {len(notebooks)} notebooks for queries: {queries}")
    
    for nb_path in notebooks:
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            
            for cell_idx, cell in enumerate(nb.get("cells", [])):
                cell_type = cell.get("cell_type")
                source = "".join(cell.get("source", []))
                
                # Check source
                for q in queries:
                    if q in source:
                        print(f"[SOURCE MATCH] File: {nb_path.relative_to(p)}, Cell: {cell_idx} ({cell_type}), Query: '{q}'")
                        print(source[:300])
                        print("-" * 50)
                
                # Check outputs
                outputs = cell.get("outputs", [])
                for out in outputs:
                    out_text = ""
                    out_type = out.get("output_type")
                    if out_type == "stream":
                        out_text = "".join(out.get("text", []))
                    elif out_type == "execute_result":
                        data = out.get("data", {})
                        out_text = "".join(data.get("text/plain", []))
                    elif out_type == "error":
                        out_text = out.get("ename", "") + " " + out.get("evalue", "") + " " + "".join(out.get("traceback", []))
                        
                    for q in queries:
                        if q in out_text:
                            print(f"[OUTPUT MATCH] File: {nb_path.relative_to(p)}, Cell: {cell_idx} ({cell_type}), Query: '{q}'")
                            print("Output preview:")
                            # print surrounding text of match
                            idx = out_text.find(q)
                            start = max(0, idx - 150)
                            end = min(len(out_text), idx + 150)
                            print(out_text[start:end])
                            print("-" * 50)
        except Exception as e:
            print(f"Error reading {nb_path}: {e}")

queries = ["1057", "98.2", "98.20", "19"]
search_notebooks(".", queries)
