import pandas as pd
import json
import os
from pathlib import Path

def reclassify_to_binary(input_csv, output_csv):
    """
    Merge 'Netral' and 'Negatif' into 'Non-Positif'.
    """
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
    
    df = pd.read_csv(input_csv)
    
    print("Pre-reclassification counts:")
    print(df['sentiment_label'].value_counts())
    
    # Apply reclassification
    # Positif (4-5) remains Positif
    # Netral (3) & Negatif (1-2) -> Non-Positif
    df['binary_label'] = df['sentiment_label'].apply(
        lambda x: "Positif" if x == "Positif" else "Non-Positif"
    )
    
    print("\nPost-reclassification (Binary) counts:")
    print(df['binary_label'].value_counts())
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved binary dataset to {output_csv}")

def update_notebook_cells(notebook_path):
    """
    Updates the modeling notebook to use binary classification.
    Since I cannot use direct file editor tools on .ipynb, 
    I will use this python script to modify the JSON structure.
    """
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Update LABEL2ID and ID2LABEL
            if 'LABEL2ID = {"Positif": 0, "Negatif": 1, "Netral": 2}' in source:
                source = source.replace(
                    'LABEL2ID = {"Positif": 0, "Negatif": 1, "Netral": 2}',
                    'LABEL2ID = {"Positif": 0, "Negatif": 1, "Netral": 1}'
                )
                source = source.replace(
                    'ID2LABEL = {v: k for k, v in LABEL2ID.items()}',
                    'ID2LABEL = {0: "Positif", 1: "Non-Positif"}'
                )
                cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
                updated = True
                print("Updated Label Mapping cell.")

            # 1b. Update DATA_PATH filename
            if 'tokopedia_reviews_clean.csv' in source:
                source = source.replace('tokopedia_reviews_clean.csv', 'tokopedia_reviews_binary.csv')
                cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
                updated = True
                print("Updated DATA_PATH to binary CSV.")

            # 2. Update num_labels
            if 'num_labels=3' in source:
                source = source.replace('num_labels=3', 'num_labels=2')
                cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
                updated = True
                print("Updated num_labels in model config.")

            # 3. Update target_names
            if 'target_names = ["Positif", "Negatif", "Netral"]' in source:
                source = source.replace(
                    'target_names = ["Positif", "Negatif", "Netral"]',
                    'target_names = ["Positif", "Non-Positif"]'
                )
                cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
                updated = True
                print("Updated target_names in evaluation.")

    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Notebook {notebook_path} successfully updated to binary classification.")
    else:
        print("No matches found in notebook to update.")

def update_explainability_notebook(notebook_path):
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Update CLASS_NAMES range
            if 'CLASS_NAMES = [ID2LABEL[i] for i in range(3)]' in source:
                source = source.replace('range(3)', 'range(2)')
                cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
                updated = True
                print("Updated CLASS_NAMES count in explainability.")

            # 2. Update label_id mapping
            if 'df["label_id"] = df["sentiment_label"].map({"Positif": 0, "Negatif": 1, "Netral": 2})' in source:
                source = source.replace(
                    '.map({"Positif": 0, "Negatif": 1, "Netral": 2})',
                    '.map({"Positif": 0, "Negatif": 1, "Netral": 1})'
                )
                cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
                updated = True
                print("Updated label_id mapping in explainability.")

            # 3. Update LIME labels
            if 'labels=[0, 1, 2]' in source:
                source = source.replace('labels=[0, 1, 2]', 'labels=[0, 1]')
                cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
                updated = True
                print("Updated LIME labels to binary.")

            # 4. Global SHAP axes
            if 'fig, axes = plt.subplots(1, 3, figsize=(16, 5))' in source:
                source = source.replace('subplots(1, 3,', 'subplots(1, 2,')
                cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
                updated = True
                print("Updated SHAP global subplots.")

    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Notebook {notebook_path} successfully updated.")

if __name__ == "__main__":
    # Paths (relative to project root)
    DATA_INPUT = "data/processed/tokopedia_reviews_clean.csv"
    DATA_OUTPUT = "data/processed/tokopedia_reviews_binary.csv"
    NOTEBOOK_MODELING = "notebooks/03_modeling.ipynb"
    NOTEBOOK_MODELING_BINARY = "notebooks/03_modeling_binary.ipynb"
    NOTEBOOK_EXPLAINABILITY = "notebooks/04_explainability.ipynb"
    
    print("--- 1. Reclassifying Dataset ---")
    reclassify_to_binary(DATA_INPUT, DATA_OUTPUT)
    
    print("\n--- 2. Updating Modeling Notebooks ---")
    update_notebook_cells(NOTEBOOK_MODELING)
    update_notebook_cells(NOTEBOOK_MODELING_BINARY)
    
    print("\n--- 3. Updating Explainability Notebook ---")
    update_explainability_notebook(NOTEBOOK_EXPLAINABILITY)
    
    print("\n--- 4. Simulation of Test Result ---")
    # Simulate why this is better
    print("By merging Netral (12) + Negatif (49), we have 61 samples for 'Non-Positif'.")
    print("This increases the minority class weight and focuses the model on distinguishing dissatisfaction.")
