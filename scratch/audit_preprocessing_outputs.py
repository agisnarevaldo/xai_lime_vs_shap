import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def inspect_preprocessing_outputs(filepath):
    print(f"\n=== CELL OUTPUTS IN PREPROCESSING: {filepath} ===")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        if cell_type == 'code':
            source = "".join(cell.get('source', []))
            outputs = cell.get('outputs', [])
            
            out_text = ""
            for out in outputs:
                if 'text' in out:
                    out_text += "".join(out['text'])
                elif 'data' in out and 'text/plain' in out['data']:
                    out_text += "".join(out['data']['text/plain'])
            
            if out_text:
                print(f"Cell {idx} (Code) source: {source.splitlines()[0] if source.splitlines() else '<empty>'}")
                print("Output:")
                print(out_text.strip())
                print("-" * 50)

inspect_preprocessing_outputs("02_preprocessing_prdect.ipynb")
