import pandas as pd
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

train_path = "data/processed/prdect_train.csv"
test_path = "data/processed/prdect_test.csv"
history_path = "outputs/finetuning_indobert/reports/training_history.json"
metrics_path = "outputs/finetuning_indobert/reports/test_metrics.json"

print("=== 1. DATASETS ON DISK ===")
if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    print(f"Train path: {train_path}")
    print(f"  Rows count      : {len(train_df)}")
    print(f"  Columns         : {list(train_df.columns)}")
    if 'fold' in train_df.columns:
        print("  Fold distribution:")
        print(train_df['fold'].value_counts().sort_index())
        print("  Sentiment class distribution in train:")
        print(train_df['emotion_label'].value_counts())
else:
    print(f"Train path {train_path} not found!")

if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
    print(f"Test path: {test_path}")
    print(f"  Rows count      : {len(test_df)}")
    print(f"  Columns         : {list(test_df.columns)}")
    print("  Sentiment class distribution in test:")
    print(test_df['emotion_label'].value_counts())
else:
    print(f"Test path {test_path} not found!")

print("\n=== 2. TRAINING HISTORY AND METRICS ===")
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    print(f"Training History loaded from: {history_path}")
    print(f"  Number of folds : {len(history)}")
    for i, fold_hist in enumerate(history):
        epochs_run = len(fold_hist.get('epoch', []))
        last_train_loss = fold_hist.get('train_loss', [])[-1] if fold_hist.get('train_loss') else None
        last_val_loss = fold_hist.get('val_loss', [])[-1] if fold_hist.get('val_loss') else None
        print(f"    Fold {i+1}: {epochs_run} epochs run, final train_loss={last_train_loss}, final val_loss={last_val_loss}")
else:
    print(f"History path {history_path} not found!")

if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    print(f"Test Metrics loaded from: {metrics_path}")
    print(json.dumps(metrics, indent=2))
else:
    print(f"Metrics path {metrics_path} not found!")

print("\n=== 3. COMPARISON ANALYSIS ===")
total_data_disk = len(train_df) + len(test_df)
print(f"Total rows on processed disk: {total_data_disk} (Train {len(train_df)} + Test {len(test_df)})")
