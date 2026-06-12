import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def inspect_notebook_cell_outputs(filepath):
    print(f"\n=== INSPECTING CELL OUTPUTS IN: {filepath} ===")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        if cell_type == 'code':
            source = "".join(cell.get('source', []))
            outputs = cell.get('outputs', [])
            
            # Print if output has specific keywords like "Total" or "Train size" or "Tabel" or "Accuracy"
            out_text = ""
            for out in outputs:
                if 'text' in out:
                    out_text += "".join(out['text'])
                elif 'data' in out and 'text/plain' in out['data']:
                    out_text += "".join(out['data']['text/plain'])
            
            interesting_keywords = ["Train size", "Test size", "Jumlah Baris", "DISTRIBUSI SENTIMEN", "Ensemble Test", "Accuracy", "negative", "positive"]
            has_keyword = any(kw in out_text or kw in source for kw in interesting_keywords)
            
            if has_keyword and out_text:
                print(f"Cell {idx} (Code) source first line: {source.splitlines()[0] if source.splitlines() else '<empty>'}")
                print("Output:")
                print(out_text.strip())
                print("-" * 50)

inspect_notebook_cell_outputs("02_preprocessing_prdect.ipynb")
inspect_notebook_cell_outputs("02_finetuning_indobert_emotion2.ipynb")
