import json

# Analyze training dynamics
with open('outputs/finetuning_indobert/reports/training_history.json', 'r') as f:
    hist = json.load(f)

epochs = hist['epoch']
train_acc = hist['train_acc']
val_acc = hist['val_acc']
train_loss = hist['train_loss']
val_loss = hist['val_loss']
val_f1 = hist['val_macro_f1']

print("=" * 60)
print("TRAINING DYNAMICS ANALYSIS")
print("=" * 60)
print(f"\n{'Epoch':>5} | {'Train Acc':>9} | {'Val Acc':>7} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val F1':>6} | {'Gap (overfitting)':>18}")
print("-" * 85)
for i in range(len(epochs)):
    gap = train_acc[i] - val_acc[i]
    loss_gap = val_loss[i] - train_loss[i]
    overfitting = "⚠ OVERFIT" if gap > 0.15 else ""
    print(f"{epochs[i]:>5} | {train_acc[i]:>9.4f} | {val_acc[i]:>7.4f} | {train_loss[i]:>10.4f} | {val_loss[i]:>8.4f} | {val_f1[i]:>6.4f} | {gap:>+8.4f} {overfitting}")

print("\n" + "=" * 60)
print("OVERFITTING ANALYSIS")
print("=" * 60)

# Find the peak val performance
best_val_f1_ep = max(range(len(val_f1)), key=lambda i: val_f1[i])
print(f"Best val Macro F1: {val_f1[best_val_f1_ep]:.4f} at epoch {epochs[best_val_f1_ep]}")
print(f"At best epoch:")
print(f"  Train acc: {train_acc[best_val_f1_ep]:.4f}")
print(f"  Val acc:   {val_acc[best_val_f1_ep]:.4f}")
print(f"  Train loss: {train_loss[best_val_f1_ep]:.4f}")
print(f"  Val loss:   {val_loss[best_val_f1_ep]:.4f}")
acc_gap = train_acc[best_val_f1_ep] - val_acc[best_val_f1_ep]
loss_gap = val_loss[best_val_f1_ep] - train_loss[best_val_f1_ep]
print(f"  Acc gap (train-val): {acc_gap:+.4f}")
print(f"  Loss gap (val-train): {loss_gap:+.4f}")

# Check if val loss plateaued after epoch 2
print(f"\nVal F1 plateau analysis:")
print(f"  Epoch 2 val F1: {val_f1[1]:.4f}")
print(f"  Best val F1:    {max(val_f1):.4f}")
print(f"  Improvement from ep2 to best: {max(val_f1) - val_f1[1]:+.4f}")
print(f"\nConclusion: Val F1 essentially plateaued after epoch 2, ranging {min(val_f1[1:]):.4f}-{max(val_f1):.4f}")
print(f"(a range of only {max(val_f1) - min(val_f1[1:]):.4f})")
print(f"\nOverfitting ratio (final epoch): train_acc/val_acc = {train_acc[-1]/val_acc[-1]:.2f}")
print(f"Val loss increased by {(val_loss[-1]/val_loss[1]-1)*100:.1f}% from epoch 2 to epoch 9")

print("\n" + "=" * 60)
print("LABEL CONFUSION MATRIX (based on error_analysis.csv)")
print("=" * 60)
import csv
from collections import defaultdict

errors = defaultdict(lambda: defaultdict(int))
total = defaultdict(int)
correct = defaultdict(int)

with open('outputs/finetuning_indobert/reports/error_analysis.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = row['true_emotion']
        p = row['pred_emotion']
        errors[t][p] += 1

# Test distribution
with open('data/processed/prdect_test.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total[row['emotion_label']] += 1

for cls in ['happy', 'love', 'anger', 'fear', 'sadness']:
    errs = sum(errors[cls].values())
    correct_pct = 100 * (total[cls] - errs) / total[cls]
    print(f"\n{cls.upper()} (n={total[cls]}, {correct_pct:.1f}% correct):")
    if errors[cls]:
        for pred, cnt in sorted(errors[cls].items(), key=lambda x: -x[1]):
            print(f"  -> predicted as {pred:10s}: {cnt:3d} ({100*cnt/total[cls]:.1f}%)")

print("\n" + "=" * 60)
print("FINE/FEAR/SADNESS CONFUSION TRIANGLE")
print("=" * 60)
print("Key insight: fear/sadness/anger are highly confused:")
print(f"  fear -> sadness: {errors['fear']['sadness']:3d}")
print(f"  fear -> anger:   {errors['fear']['anger']:3d}")
print(f"  sadness -> fear: {errors['sadness']['fear']:3d}")
print(f"  sadness -> anger:{errors['sadness']['anger']:3d}")
print(f"  anger -> fear:   {errors['anger']['fear']:3d}")
print(f"  anger -> sadness:{errors['anger']['sadness']:3d}")
print(f"\n  happy <-> love confusion:")
print(f"  happy -> love:   {errors['happy']['love']:3d}")
print(f"  love -> happy:   {errors['love']['happy']:3d}")
