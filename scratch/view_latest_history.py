import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

history_path = "outputs/finetuning_indobert/reports/training_history.json"
with open(history_path, 'r', encoding='utf-8') as f:
    history = json.load(f)

for i, fold_hist in enumerate(history):
    print(f"\nFold {i + 1}:")
    print(f"  Epochs trained: {len(fold_hist['epoch'])}")
    print(f"  Train loss    : {[f'{x:.4f}' for x in fold_hist['train_loss']]}")
    print(f"  Val loss      : {[f'{x:.4f}' for x in fold_hist['val_loss']]}")
    print(f"  Val acc       : {[f'{x:.4f}' for x in fold_hist['val_acc']]}")
