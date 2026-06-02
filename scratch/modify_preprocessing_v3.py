import json

def modify_preprocessing_v3():
    filepath = "02_preprocessing_prdect.ipynb"
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Cell 9: Apply Cleaning, Label Mapping, Deduplication, and DROP_CONFLICTS
    cell_9 = nb['cells'][9]
    cell_9_source = [
        "## 8. Apply Cleaning, Label Mapping & Deduplication\n",
        "EMOTION_MAP = {\n",
        "    'happy'  : 0,\n",
        "    'love'   : 1,\n",
        "    'anger'  : 2,\n",
        "    'fear'   : 3,\n",
        "    'sadness': 4,\n",
        "}\n",
        "ID2EMOTION = {v: k for k, v in EMOTION_MAP.items()}\n",
        "\n",
        "# Normalize label\n",
        "df['emotion_label'] = df[LABEL_COL].str.lower().str.strip()\n",
        "\n",
        "# Apply mapping\n",
        "df['label'] = df['emotion_label'].map(EMOTION_MAP)\n",
        "df = df.dropna(subset=['label'])\n",
        "\n",
        "# --- 1. Audit Conflicting Labels ---\n",
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
        "# Save conflicting labels report for manual review\n",
        "reports_dir = Path(\"outputs/finetuning_indobert/reports\")\n",
        "reports_dir.mkdir(parents=True, exist_ok=True)\n",
        "conflicting_out = reports_dir / \"conflicting_labels.csv\"\n",
        "conflicting_df[[TEXT_COL, 'Emotion', 'Customer Rating', 'Category']].to_csv(conflicting_out, index=False)\n",
        "print(f\"Saved conflicting labels report ({len(conflicting_df)} rows) to: {conflicting_out}\")\n",
        "\n",
        "# --- 2. Deduplicate & Handle Conflicts ---\n",
        "DROP_CONFLICTS = True  # Set to True to completely remove reviews with conflicting labels to eliminate noise\n",
        "before_drop = len(df)\n",
        "if DROP_CONFLICTS:\n",
        "    # Drop all occurrences of reviews that have conflicting labels\n",
        "    df = df[~df[TEXT_COL].isin(conflicting_texts)].copy()\n",
        "    print(f\"Dropped conflicting reviews: {before_drop} -> {len(df)} rows (removed {before_drop - len(df)} rows)\")\n",
        "\n",
        "before_dedup = len(df)\n",
        "df = df.drop_duplicates(subset=[TEXT_COL], keep='first').copy()\n",
        "print(f\"Deduplicated: {before_dedup} -> {len(df)} rows ({before_dedup - len(df)} duplicate rows removed)\")\n",
        "\n",
        "# Apply text cleaning\n",
        "df['review_clean'] = df[TEXT_COL].apply(clean_text)\n",
        "\n",
        "# Hapus baris kosong\n",
        "before_empty = len(df)\n",
        "df = df.dropna(subset=['review_clean'])\n",
        "df = df[df['review_clean'].str.strip().str.len() > 3]\n",
        "\n",
        "# --- 3. Prepend Metadata (Customer Rating & Category) to clean text ---\n",
        "# This embeds the rating and product category directly into the text for BERT self-attention\n",
        "df['review_clean'] = \"Rating: \" + df['Customer Rating'].astype(str) + \". Kategori: \" + df['Category'].astype(str) + \". Teks: \" + df['review_clean']\n",
        "\n",
        "after_empty = len(df)\n",
        "df['label'] = df['label'].astype(int)\n",
        "\n",
        "print(f\"Data setelah filter teks kosong & penambahan metadata: {before_empty} -> {after_empty} ({before_empty - after_empty} dihapus)\")\n",
        "print(f\"\\nDistribusi final:\")\n",
        "print(df['emotion_label'].value_counts())"
    ]
    cell_9['source'] = cell_9_source

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully updated preprocessing notebook with DROP_CONFLICTS!")

modify_preprocessing_v3()
