import json

def fix_training_keyerror():
    filepath = "02_finetuning_indobert_emotion2.ipynb"
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Cell 4: Load Processed Dataset & Schema Normalization
    cell_4 = nb['cells'][4]
    cell_4_source = [
        "df_train = pd.read_csv(TRAIN_DATA_PATH)\n",
        "df_test = pd.read_csv(TEST_DATA_PATH)\n",
        "\n",
        "with open(MAPPING_PATH, 'r') as f:\n",
        "    label_mapping = json.load(f)\n",
        "\n",
        "print(f\"Train samples: {len(df_train)}\")\n",
        "print(f\"Test samples : {len(df_test)}\")\n",
        "print(\"-\" * 30)\n",
        "print(\"Label Mapping:\")\n",
        "print(label_mapping)\n",
        "\n",
        "# Normalize schema from preprocessing output\n",
        "text_column = 'review_clean' if 'review_clean' in df_train.columns else 'Customer Review'\n",
        "label_name_column = 'emotion_label' if 'emotion_label' in df_train.columns else 'Emotion'\n",
        "label_id_column = 'label' if 'label' in df_train.columns else 'Label'\n",
        "\n",
        "# Robust check for cross-validation 'fold' column\n",
        "if 'fold' in df_train.columns:\n",
        "    fold_column = 'fold'\n",
        "elif 'Fold' in df_train.columns:\n",
        "    fold_column = 'Fold'\n",
        "else:\n",
        "    print(\"WARNING: 'fold' column not found in training dataset. Generating folds dynamically...\")\n",
        "    from sklearn.model_selection import StratifiedKFold\n",
        "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n",
        "    df_train['fold'] = -1\n",
        "    label_col_for_fold = label_id_column if label_id_column in df_train.columns else label_name_column\n",
        "    if df_train[label_col_for_fold].dtype == object:\n",
        "        # map to integers if label column contains string names\n",
        "        temp_labels = df_train[label_col_for_fold].astype(str).str.lower().str.strip().map(EMOTION_MAP)\n",
        "    else:\n",
        "        temp_labels = df_train[label_col_for_fold].astype(int)\n",
        "    \n",
        "    # Populate folds\n",
        "    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train, temp_labels)):\n",
        "        df_train.loc[val_idx, 'fold'] = fold_idx\n",
        "    fold_column = 'fold'\n",
        "\n",
        "train_df = df_train[[text_column, label_name_column, label_id_column, fold_column]].copy()\n",
        "test_df = df_test[[text_column, label_name_column, label_id_column]].copy()\n",
        "\n",
        "train_df.columns = ['text', 'emotion_label', 'label', 'fold']\n",
        "test_df.columns = ['text', 'emotion_label', 'label']\n",
        "\n",
        "train_df['emotion_label'] = train_df['emotion_label'].astype(str).str.lower().str.strip()\n",
        "test_df['emotion_label'] = test_df['emotion_label'].astype(str).str.lower().str.strip()\n",
        "train_df['label'] = train_df['label'].astype(int)\n",
        "test_df['label'] = test_df['label'].astype(int)\n",
        "train_df['fold'] = train_df['fold'].astype(int)\n",
        "\n",
        "train_df = train_df.reset_index(drop=True)\n",
        "test_df = test_df.reset_index(drop=True)\n",
        "\n",
        "print(f\"Train set    : {len(train_df)}\")\n",
        "print(f\"Test set     : {len(test_df)}\")\n",
        "print(\"Fold distribution in Train:\")\n",
        "print(train_df.groupby(['fold', 'emotion_label']).size().unstack(fill_value=0))"
    ]
    cell_4['source'] = cell_4_source

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully patched training notebook KeyError!")

fix_training_keyerror()
