import json

def modify_preprocessing_v5():
    filepath = "02_preprocessing_prdect.ipynb"
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Cell 2: Imports & Config
    cell_2 = nb['cells'][2]
    cell_2_source = [
        "## 1. Imports & Config\n",
        "import os\n",
        "import re\n",
        "import warnings\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from pathlib import Path\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.model_selection import StratifiedKFold\n",
        "import unicodedata\n",
        "\n",
        "warnings.filterwarnings('ignore')\n",
        "SEED = 42\n",
        "np.random.seed(SEED)\n",
        "\n",
        "def find_project_root() -> Path:\n",
        "    markers = [\"data\", \"notebooks\", \"src\"]\n",
        "    candidate = Path.cwd()\n",
        "    for _ in range(3):\n",
        "        if all((candidate / m).exists() for m in markers):\n",
        "            return candidate\n",
        "        candidate = candidate.parent\n",
        "    return Path.cwd()\n",
        "\n",
        "PROJECT_ROOT = find_project_root()\n",
        "RAW_DIR     = PROJECT_ROOT / \"data\" / \"raw\"\n",
        "PROC_DIR    = PROJECT_ROOT / \"data\" / \"processed\"\n",
        "RAW_DIR.mkdir(parents=True, exist_ok=True)\n",
        "PROC_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# PIVOT: Menggunakan kolom 'Sentiment' untuk klasifikasi biner (Positive vs Negative)\n",
        "TEXT_COL  = \"Customer Review\"\n",
        "LABEL_COL = \"Sentiment\"\n",
        "\n",
        "print(f\"Project root : {PROJECT_ROOT}\")\n",
        "print(f\"Raw dir      : {RAW_DIR}\")\n",
        "print(f\"Processed    : {PROC_DIR}\")\n",
        "print(f\"TEXT_COL     : {TEXT_COL}\")\n",
        "print(f\"LABEL_COL    : {LABEL_COL}\")"
    ]
    cell_2['source'] = cell_2_source

    # Cell 9: Apply Cleaning, Label Mapping, Deduplication & Academic Tables
    cell_9 = nb['cells'][9]
    cell_9_source = [
        "## 8. Apply Cleaning, Label Mapping & Deduplication\n",
        "SENTIMENT_MAP = {\n",
        "    'positive': 1,\n",
        "    'negative': 0\n",
        "}\n",
        "ID2SENTIMENT = {v: k for k, v in SENTIMENT_MAP.items()}\n",
        "\n",
        "# Raw row count for Table 2\n",
        "len_raw = len(df)\n",
        "\n",
        "# Normalize label (lowercase & strip)\n",
        "df['emotion_label'] = df[LABEL_COL].str.lower().str.strip()\n",
        "\n",
        "# Apply mapping\n",
        "df['label'] = df['emotion_label'].map(SENTIMENT_MAP)\n",
        "df = df.dropna(subset=['label'])\n",
        "\n",
        "# --- 1. Audit Conflicting Labels (Sentiment Level) ---\n",
        "duplicate_mask = df.duplicated(subset=[TEXT_COL], keep=False)\n",
        "duplicates_df = df[duplicate_mask]\n",
        "\n",
        "conflicting_texts = []\n",
        "grouped_dupes = duplicates_df.groupby(TEXT_COL)\n",
        "for text, group in grouped_dupes:\n",
        "    if group['label'].nunique() > 1:\n",
        "        conflicting_texts.append(text)\n",
        "\n",
        "conflicting_df = duplicates_df[duplicates_df[TEXT_COL].isin(conflicting_texts)].copy()\n",
        "\n",
        "# Save conflicting labels report for manual review (should find 0 conflicts for Sentiment)\n",
        "reports_dir = Path(\"outputs/finetuning_indobert/reports\")\n",
        "reports_dir.mkdir(parents=True, exist_ok=True)\n",
        "conflicting_out = reports_dir / \"conflicting_labels.csv\"\n",
        "conflicting_df[[TEXT_COL, 'Sentiment', 'Customer Rating', 'Category']].to_csv(conflicting_out, index=False)\n",
        "print(f\"Saved conflicting labels report ({len(conflicting_df)} rows) to: {conflicting_out}\")\n",
        "\n",
        "# --- 2. Deduplicate (keep the first occurrence) ---\n",
        "df = df.drop_duplicates(subset=[TEXT_COL], keep='first').copy()\n",
        "len_dedup = len(df)\n",
        "\n",
        "# --- 3. Apply Text Cleaning (PURE TEXT ONLY for XAI/LIME/SHAP mapping) ---\n",
        "df['review_clean'] = df[TEXT_COL].apply(clean_text)\n",
        "\n",
        "# Hapus baris kosong atau terlalu pendek\n",
        "df = df.dropna(subset=['review_clean'])\n",
        "df = df[df['review_clean'].str.strip().str.len() > 3]\n",
        "len_clean = len(df)\n",
        "df['label'] = df['label'].astype(int)\n",
        "\n",
        "print(\"\\n\" + \"=\"*60 + \"\\nTABEL AKADEMIK UNTUK LAPORAN SKRIPSI\\n\" + \"=\"*60)\n",
        "\n",
        "# --- TABEL 1: Sebelum vs Sesudah Pembersihan ---\n",
        "print(\"\\n### TABEL 1: CONTOH PEMBERSIHAN TEKS (BEFORE vs AFTER)\")\n",
        "sample_cleaning = df[[TEXT_COL, 'review_clean']].head(5)\n",
        "print(sample_cleaning.to_markdown(index=False))\n",
        "print(\"-\" * 60)\n",
        "\n",
        "# --- TABEL 2: Statistik Row Dataset (Sebelum vs Sesudah) ---\n",
        "print(\"\\n### TABEL 2: STATISTIK ROW DATASET (SEBELUM vs SESUDAH)\")\n",
        "stats = {\n",
        "    \"Tahapan Preprocessing\": [\n",
        "        \"1. Dataset Mentah (Original Raw)\", \n",
        "        \"2. Setelah Deduplikasi (Deduplicated)\", \n",
        "        \"3. Setelah Filter Teks Pendek (Cleaned)\"\n",
        "    ],\n",
        "    \"Jumlah Baris\": [len_raw, len_dedup, len_clean],\n",
        "    \"Baris Dibuang\": [0, len_raw - len_dedup, len_dedup - len_clean]\n",
        "}\n",
        "print(pd.DataFrame(stats).to_markdown(index=False))\n",
        "print(\"-\" * 60)\n",
        "\n",
        "# --- TABEL 3: Distribusi Kelas Sentimen Final ---\n",
        "print(\"\\n### TABEL 3: DISTRIBUSI SENTIMEN FINAL\")\n",
        "dist_df = df['emotion_label'].value_counts().reset_index()\n",
        "dist_df.columns = ['Kelas Sentimen', 'Jumlah Ulasan']\n",
        "dist_df['Persentase (%)'] = (dist_df['Jumlah Ulasan'] / len(df)) * 100\n",
        "print(dist_df.to_markdown(index=False))\n",
        "print(\"=\"*60)"
    ]
    cell_9['source'] = cell_9_source

    # Cell 10: Stratified Split & Stratified K-Fold
    cell_10 = nb['cells'][10]
    cell_10_source = [
        "## 9. Stratified Split (80% Train / 20% Test) & Stratified K-Fold\n",
        "# Split into Train and Test\n",
        "df_train, df_test = train_test_split(\n",
        "    df, test_size=0.2, random_state=SEED, stratify=df['label']\n",
        ")\n",
        "\n",
        "# Add Stratified 5-Fold split to the training set for cross-validation\n",
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n",
        "df_train['fold'] = -1\n",
        "for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['label'])):\n",
        "    df_train.loc[df_train.index[val_idx], 'fold'] = fold_idx\n",
        "\n",
        "print(f\"Train size: {len(df_train)}\")\n",
        "print(f\"Test size : {len(df_test)}\")\n",
        "\n",
        "print(\"\\nFold distribution in Train:\")\n",
        "print(df_train.groupby(['fold', 'emotion_label']).size().unstack(fill_value=0))"
    ]
    cell_10['source'] = cell_10_source

    # Cell 11: Export Data
    cell_11 = nb['cells'][11]
    cell_11_source = [
        "## 10. Export Data\n",
        "import json\n",
        "\n",
        "# Export cleaned full dataset\n",
        "clean_out = PROC_DIR / \"prdect_clean.csv\"\n",
        "df[['review_clean', 'emotion_label', 'label']].to_csv(clean_out, index=False)\n",
        "print(f\"Saved: {clean_out}\")\n",
        "\n",
        "# Export train/test split (including fold in train)\n",
        "train_out = PROC_DIR / \"prdect_train.csv\"\n",
        "test_out  = PROC_DIR / \"prdect_test.csv\"\n",
        "df_train[['review_clean', 'emotion_label', 'label', 'fold']].to_csv(train_out, index=False)\n",
        "df_test[['review_clean', 'emotion_label', 'label']].to_csv(test_out, index=False)\n",
        "print(f\"Saved: {train_out}\")\n",
        "print(f\"Saved: {test_out}\")\n",
        "\n",
        "# Export label map\n",
        "label_map_out = PROC_DIR / \"prdect_label_map.json\"\n",
        "with open(label_map_out, 'w') as f:\n",
        "    json.dump({\"label2id\": SENTIMENT_MAP, \"id2label\": ID2SENTIMENT}, f, indent=2)\n",
        "print(f\"Saved: {label_map_out}\")\n",
        "\n",
        "print(\"\\n=== Summary ===\")\n",
        "print(f\"Full dataset : {len(df)} rows\")\n",
        "print(f\"Train set    : {len(df_train)} rows\")\n",
        "print(f\"Test set     : {len(df_test)} rows\")\n",
        "print(f\"Label mapping: {SENTIMENT_MAP}\")"
    ]
    cell_11['source'] = cell_11_source

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully modified preprocessing notebook for Academic Sentiment Classification!")

modify_preprocessing_v5()
