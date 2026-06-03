import json
import sys
import os
import traceback

sys.stdout.reconfigure(encoding='utf-8')

def run_notebook(filepath):
    print(f"Running notebook: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Setup global namespace for execution
    global_ns = {
        '__name__': '__main__',
        'IN_COLAB': False,  # Explicitly force local mode
        'display': print    # Define display as print for non-Jupyter environment
    }
    
    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            # skip colab cell magic like !pip install
            clean_lines = []
            for line in source.split('\n'):
                if line.strip().startswith('!') or line.strip().startswith('%'):
                    indent = len(line) - len(line.lstrip())
                    clean_lines.append(" " * indent + "pass # " + line.strip())
                else:
                    clean_lines.append(line)
            clean_source = "\n".join(clean_lines)
            
            print(f"\n--- Executing Cell {idx} ---")
            try:
                # Compile and execute the cell code
                code_obj = compile(clean_source, f"cell_{idx}", "exec")
                exec(code_obj, global_ns)
            except Exception as e:
                print(f"[ERROR] Exception in Cell {idx}:")
                traceback.print_exc()
                sys.exit(1)
                
    print("\nNotebook execution completed successfully!")

if __name__ == "__main__":
    run_notebook("02_preprocessing_prdect.ipynb")
