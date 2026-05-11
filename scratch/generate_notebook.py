import nbformat as nbf
import os

# Create directory if not exists
os.makedirs('h:/My Drive/xai_lime_vs_shap/notebooks', exist_ok=True)

nb = nbf.v4.new_notebook()

# ---------------------------------------------------------------------------
# 1. INTRO
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""# Tahap 1 — Dataset dan Preprocessing (PRDECT-ID)
Tujuan dari notebook ini adalah untuk menyiapkan dataset berkualitas tinggi dari PRDECT-ID untuk digunakan dalam eksperimen **emotion classification**. Seluruh proses didokumentasikan dengan visualisasi dan tabel komparasi sebagai bahan laporan skripsi."""))

# ---------------------------------------------------------------------------
# 2. CONFIG & IMPORTS
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Konfigurasi & Import Library
Mendefinisikan parameter global dan library yang diperlukan."""))

nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Global Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128

# Path Configuration (Colab & Local Friendly)
import os
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Running in Google Colab. Mounting Drive...")
    drive.mount('/content/drive')
    # Update this to match your Google Drive project path
    BASE_PATH = "/content/drive/MyDrive/xai_lime_vs_shap"
else:
    print("Running locally.")
    BASE_PATH = ".." # Assuming running from notebooks/ folder

RAW_DATA_PATH = f"{BASE_PATH}/data/raw/PRDECT-ID Dataset.csv"
PROCESSED_DIR = f"{BASE_PATH}/data/processed"

print(f"Eksperimen dimulai pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")"""))

# ---------------------------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Load Dataset
Membaca dataset asli PRDECT-ID."""))

nb.cells.append(nbf.v4.new_code_cell("""df_raw = pd.read_csv(RAW_DATA_PATH)
print(f"Dataset berhasil dimuat: {RAW_DATA_PATH}")
df_raw.head()"""))

# ---------------------------------------------------------------------------
# 4. METADATA
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Dataset Metadata Summary
Memberikan gambaran umum mengenai statistik data awal."""))

nb.cells.append(nbf.v4.new_code_cell("""print("=== Dataset Metadata ===")
print(f"Jumlah Baris: {df_raw.shape[0]}")
print(f"Jumlah Kolom: {df_raw.shape[1]}")
print(f"Daftar Kolom: {df_raw.columns.tolist()}")
print(f"Jumlah Kelas (Emotion): {df_raw['Emotion'].nunique()}")
print("-" * 30)
print("Missing Value Summary:")
print(df_raw.isna().sum())
print("-" * 30)
print(f"Jumlah Data Duplikat: {df_raw.duplicated().sum()}")"""))

# ---------------------------------------------------------------------------
# 5. COLUMN SELECTION
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Seleksi Kolom
Hanya mengambil kolom `Customer Review` dan `Emotion`.

**Rationale**: Kolom lain seperti Category, Price, dan Rating tidak relevan terhadap tugas klasifikasi emosi dan berpotensi menjadi noise bagi model."""))

nb.cells.append(nbf.v4.new_code_cell("""df = df_raw[['Customer Review', 'Emotion']].copy()
print(f"Shape setelah seleksi kolom: {df.shape}")
df.head()"""))

# ---------------------------------------------------------------------------
# 6. DUPLICATE REMOVAL
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 5. Pembersihan Data Duplikat & Missing Values"""))

nb.cells.append(nbf.v4.new_code_cell("""# Hapus Duplikat
initial_count = len(df)
df = df.drop_duplicates().reset_index(drop=True)
duplicates_removed = initial_count - len(df)
print(f"Jumlah duplikat dihapus: {duplicates_removed}")

# Hapus Missing Values
df = df.dropna().reset_index(drop=True)
print(f"Jumlah baris setelah pembersihan: {len(df)}")"""))

# ---------------------------------------------------------------------------
# 7. CLEANING DESIGN DECISION
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 6. Preprocessing Design Decision
Pada eksperimen ini, **Stemming** dan **Stopword Removal** sengaja **TIDAK** dilakukan.

**Alasan**: Model berbasis Transformer seperti **IndoBERT** dilatih untuk memahami konteks dalam urutan kalimat yang utuh. Menghapus stopword atau mengubah kata ke bentuk dasarnya dapat menghilangkan informasi semantik dan sintaksis yang krusial bagi pemahaman konteks emosi oleh model."""))

# ---------------------------------------------------------------------------
# 8. TEXT CLEANING
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 7. Cleaning Text
Proses pembersihan teks meliputi: lowercase, penghapusan tanda baca, karakter khusus, dan normalisasi spasi."""))

nb.cells.append(nbf.v4.new_code_cell("""def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation & special characters
    # Menggunakan regex untuk menghapus tanda baca dan karakter selain alfanumerik dan spasi
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Simpan teks asli untuk perbandingan
df['Review_Original'] = df['Customer Review']
df['Customer Review'] = df['Customer Review'].apply(clean_text)

# Tabel Komparasi Before vs After untuk Skripsi
print("Tabel Komparasi Pembersihan Teks (5 Sampel):")
df[['Review_Original', 'Customer Review']].head(5)"""))

# ---------------------------------------------------------------------------
# 9. ENCODE LABEL
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 8. Encode Label
Mengubah label teks menjadi representasi numerik."""))

nb.cells.append(nbf.v4.new_code_cell("""le = LabelEncoder()
df['Label'] = le.fit_transform(df['Emotion'])

# Buat Tabel Mapping
mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
mapping_df = pd.DataFrame(list(mapping.items()), columns=['Label', 'Encoding'])

print("Label Mapping:")
display(mapping_df)

# Visualisasi Distribusi Kelas Final
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Emotion', palette='viridis', order=df['Emotion'].value_counts().index)
plt.title("Distribusi Kelas Emosi (Final)")
plt.xlabel("Emosi")
plt.ylabel("Jumlah")
plt.xticks(rotation=45)
plt.show()"""))

# ---------------------------------------------------------------------------
# 10. STRATIFIED SPLIT
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 9. Stratified Train-Test Split
Membagi data menjadi 80% Train dan 20% Test secara stratified.

**Rationale**: Stratified split digunakan karena dataset bersifat **imbalanced**. Teknik ini memastikan proporsi setiap kelas emosi tetap sama baik di subset data latih maupun data uji."""))

nb.cells.append(nbf.v4.new_code_cell("""df_train, df_test = train_test_split(
    df, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=df['Label']
)

print(f"Jumlah Data Train: {len(df_train)}")
print(f"Jumlah Data Test : {len(df_test)}")

# Visualisasi Validasi Stratifikasi
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.countplot(x=df_train['Emotion'], ax=ax[0], palette='magma', order=df['Emotion'].value_counts().index)
ax[0].set_title("Distribusi Label - Train Set")
ax[0].tick_params(axis='x', rotation=45)

sns.countplot(x=df_test['Emotion'], ax=ax[1], palette='magma', order=df['Emotion'].value_counts().index)
ax[1].set_title("Distribusi Label - Test Set")
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()"""))

# ---------------------------------------------------------------------------
# 11. SUMMARY & SAVE
# ---------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""## 10. Experiment Summary & Save Outputs"""))

nb.cells.append(nbf.v4.new_code_cell("""# Buat directory processed jika belum ada
import os
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Save Files
df.to_csv(f"{PROCESSED_DIR}/prdect_clean.csv", index=False)
df_train.to_csv(f"{PROCESSED_DIR}/prdect_train.csv", index=False)
df_test.to_csv(f"{PROCESSED_DIR}/prdect_test.csv", index=False)

# Save Label Mapping
with open(f"{PROCESSED_DIR}/prdect_label_mapping.json", "w") as f:
    json.dump(mapping, f)

print("--- Eksperimen Selesai ---")
print(f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Data     : {len(df)}")
print(f"Data Train     : {len(df_train)}")
print(f"Data Test      : {len(df_test)}")
print(f"Random Seed    : {RANDOM_STATE}")
print(f"Output Saved to: {PROCESSED_DIR}")"""))

# ---------------------------------------------------------------------------
# SAVE NOTEBOOK
# ---------------------------------------------------------------------------
with open('h:/My Drive/xai_lime_vs_shap/notebooks/01_preprocessing_prdect_emotion.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook '01_preprocessing_prdect_emotion.ipynb' has been created successfully.")
