import json
import os

def audit_notebook(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=== Audit Notebook: {path} ===")
    cells = nb.get("cells", [])
    print(f"Total cells: {len(cells)}")
    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        # Look for configuration, loading data, path variables
        if "pd.read_csv" in source or "BASE_PATH" in source or "DATA_PATH" in source or "MODEL_DIR" in source or "model.save_pretrained" in source:
            print(f"Cell {idx} ({cell_type}):")
            print(source[:500])
            print("-" * 50)

print("Auditing root notebooks:")
audit_notebook("02_preprocessing_prdect.ipynb")
audit_notebook("02_finetuning_indobert_emotion2.ipynb")
audit_notebook("notebooks/04c_explainability_binary_pipeline.ipynb")
