import json

with open("notebooks/04c_explainability_binary_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb['cells'][2]
print("=== Cell 2 (imports) ===")
print("".join(cell.get("source", [])))
