import json

def inspect_cell_3():
    with open("02_preprocessing_prdect.ipynb", 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cell = nb['cells'][3]
    print("=== CELL 3 SOURCE ===")
    print("".join(cell['source']))

inspect_cell_3()
