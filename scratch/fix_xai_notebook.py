import json
from pathlib import Path

notebook_path = Path("notebooks/04c_explainability_binary_pipeline.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Modify Cell 3 (Paths)
# We want to change TEST_DATA_PATH and TRAIN_DATA_PATH to prdect_test.csv and prdect_train.csv
cell_3 = nb["cells"][3]
cell_3_source = cell_3["source"]

new_cell_3_source = []
for line in cell_3_source:
    if 'tokopedia_reviews_binary_test.csv' in line:
        line = line.replace('tokopedia_reviews_binary_test.csv', 'prdect_test.csv')
    if 'tokopedia_reviews_binary_train.csv' in line:
        line = line.replace('tokopedia_reviews_binary_train.csv', 'prdect_train.csv')
    new_cell_3_source.append(line)

cell_3["source"] = new_cell_3_source

# Modify Cell 4 (Data loading & Renaming)
cell_4 = nb["cells"][4]
cell_4_source = cell_4["source"]

# We will completely replace cell 4's source to include renaming and mapping
new_cell_4_code = """# 3. Memuat Model, Tokenizer, dan Dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device aktif: {DEVICE}")

model, tokenizer = load_sentiment_model_and_tokenizer(MODEL_DIR, TOKENIZER_DIR, device=DEVICE)
predict_proba = PredictProbaWrapper(model, tokenizer, device=DEVICE)

df_test = pd.read_csv(TEST_DATA_PATH)
df_train = pd.read_csv(TRAIN_DATA_PATH)

# Penyelarasan skema dataset PRDECT-ID ke ekspektasi notebook XAI
df_test = df_test.rename(columns={"review_clean": "review_text_clean"})
df_train = df_train.rename(columns={"review_clean": "review_text_clean"})

# Pemetaan label string ke bahasa Indonesia kapital
df_test["sentiment_label"] = df_test["emotion_label"].map({"positive": "Positif", "negative": "Negatif"})
df_train["sentiment_label"] = df_train["emotion_label"].map({"positive": "Positif", "negative": "Negatif"})

print(f"Jumlah data test awal: {len(df_test)}")
print(f"Jumlah data train awal: {len(df_train)}")

# Filter ulasan yang terlalu pendek (min 3 kata bersih)
# karena SHAP Partition clustering membutuhkan kalimat dengan panjang minimal tertentu untuk menghindari pembagian klaster kosong.
df_test = df_test[df_test["review_text_clean"].astype(str).str.split().apply(len) >= 3].reset_index(drop=True)
df_train = df_train[df_train["review_text_clean"].astype(str).str.split().apply(len) >= 3].reset_index(drop=True)
print(f"Jumlah data test setelah filter (min 3 kata): {len(df_test)}")
print(f"Jumlah data train setelah filter (min 3 kata): {len(df_train)}")

# Definisi kelas sentimen biner
CLASS_NAMES = ["Negatif", "Positif"]"""

cell_4["source"] = [line + "\n" for line in new_cell_4_code.split("\n")]

# Save notebook back
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook notebooks/04c_explainability_binary_pipeline.ipynb successfully updated for Opsi 2!")
