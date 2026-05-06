"""
Script: fix_notebook04.py
Tujuan: Memperbaiki NameError 'predict_proba' dengan menyisipkan kembali definisinya 
        dan memastikan urutan cell benar di notebook 04.
"""

import nbformat
from pathlib import Path

BASE = Path(__file__).parent.parent
NB04 = BASE / "notebooks" / "04_explainability_prdect.ipynb"

def make_md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)

def make_code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)

PREDICT_PROBA_MD = """## 3. Define Prediction Wrapper
LIME dan SHAP membutuhkan fungsi wrapper yang menerima list teks dan mengembalikan probabilitas untuk setiap kelas."""

PREDICT_PROBA_CODE = """\
## 3. Define Prediction Wrapper
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    # 1. Tokenization & Feature Extraction (IndoBERT)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Ambil [CLS] token (index 0)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    # 2. Prediction (MLP Classifier)
    embeddings_t = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = classifier(embeddings_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs
"""

def fix_notebook04():
    print(f"Reading {NB04}...")
    nb = nbformat.read(NB04, as_version=4)
    
    # Cek apakah predict_proba sudah ada
    has_predict_proba = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'def predict_proba' in cell.source:
            has_predict_proba = True
            break
            
    if has_predict_proba:
        print("predict_proba already exists. Checking position...")
    else:
        print("predict_proba missing. Inserting Section 3...")
        
        # Cari posisi Section 2 (Define MLP)
        sec2_idx = -1
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and '## 2. Define MLP' in cell.source:
                sec2_idx = i
                break
        
        if sec2_idx != -1:
            # Sisipkan setelah Section 2
            nb.cells.insert(sec2_idx + 1, make_md(PREDICT_PROBA_MD))
            nb.cells.insert(sec2_idx + 2, make_code(PREDICT_PROBA_CODE))
            print(f"Inserted predict_proba after Section 2 (index {sec2_idx})")
        else:
            # Fallback: sisipkan sebelum Section 4
            sec4_idx = -1
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'markdown' and '## 4. Load Test Data' in cell.source:
                    sec4_idx = i
                    break
            if sec4_idx != -1:
                nb.cells.insert(sec4_idx, make_md(PREDICT_PROBA_MD))
                nb.cells.insert(sec4_idx + 1, make_code(PREDICT_PROBA_CODE))
                print(f"Inserted predict_proba before Section 4 (index {sec4_idx})")
            else:
                print("Could not find suitable insertion point!")
                return

    # 3. Pastikan SAMPLE_SETS di-flatten ke 'samples' di akhir Section 4
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'SAMPLE_SETS[cls] = pick_samples' in cell.source:
            if "samples = pd.DataFrame" not in cell.source:
                cell.source += "\n\n# Flatten SAMPLE_SETS for backward compatibility with LIME/SHAP cells\n"
                cell.source += "import pandas as pd\n"
                cell.source += "all_selected = []\n"
                cell.source += "for cls in SAMPLE_SETS:\n"
                cell.source += "    for stype in SAMPLE_SETS[cls]:\n"
                cell.source += "        all_selected.append(SAMPLE_SETS[cls][stype])\n"
                cell.source += "samples = pd.DataFrame(all_selected)\n"
                cell.source += "print(f'\\nTotal samples selected: {len(samples)}')"
                print("Added flattening logic to Section 4.")

    # Simpan perubahan
    nbformat.write(nb, NB04)
    print(f"Fixed {NB04} successfully.")

if __name__ == "__main__":
    fix_notebook04()
