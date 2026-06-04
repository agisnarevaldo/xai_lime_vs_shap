import json
import os

def parse_outputs():
    nb_path = "notebooks/04c_explainability_binary_pipeline.ipynb"
    if not os.path.exists(nb_path):
        print(f"Notebook not found at: {nb_path}")
        return
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        if cell_type == "code":
            # Print cell source header
            source = "".join(cell.get("source", []))
            first_line = source.split("\n")[0] if source else ""
            print(f"\n=== Cell {i}: {first_line[:50]} ===")
            
            # Print outputs
            outputs = cell.get("outputs", [])
            for out in outputs:
                out_type = out.get("output_type")
                if out_type == "stream":
                    print("".join(out.get("text", [])))
                elif out_type == "execute_result" or out_type == "display_data":
                    data = out.get("data", {})
                    if "text/plain" in data:
                        print("".join(data["text/plain"]))
                elif out_type == "error":
                    print("ERROR: ", out.get("ename"), out.get("evalue"))

if __name__ == "__main__":
    parse_outputs()
