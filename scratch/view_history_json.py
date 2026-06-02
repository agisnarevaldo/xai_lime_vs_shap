import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

history_path = "outputs/finetuning_indobert/reports/training_history.json"
with open(history_path, 'r', encoding='utf-8') as f:
    history = json.load(f)

print("Type of history:", type(history))
print("Length of history:", len(history))
for i, fold_hist in enumerate(history):
    print(f"\nFold {i + 1} history keys:", fold_hist.keys())
    print(f"Epochs trained: {len(fold_hist['epoch'])}")
    print(f"Train loss progression: {[f'{x:.4f}' for x in fold_hist['train_loss']]}")
    print(f"Val loss progression  : {[f'{x:.4f}' for x in fold_hist['val_loss']]}")
    print(f"Val acc progression   : {[f'{x:.4f}' for x in fold_hist['val_acc']]}")
