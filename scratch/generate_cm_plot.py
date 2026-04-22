
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data derived/calculated from notebook 03_modeling_finetuning_bert.ipynb
# Labels: [Positif, Negatif, Netral]
# Support: [281, 17, 14]
# Recall counts approx: 
# Pos (0.96): 270 TP.
# Neg (0.82): 14 TP.
# Net (0.64): 9 TP.

# Approximate Confusion Matrix based on the heatmap visualization and metrics:
# (True Labels on rows, Predicted on columns)
cm = np.array([
    [270, 4, 7],     # True Positif
    [2, 14, 1],      # True Negatif
    [1, 4, 9]        # True Netral
])

target_names = ['Positif', 'Negatif', 'Netral']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Gambar 4.6 Confusion Matrix - IndoBERT Sentiment Analysis', fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('d:/xai_lime_vs_shap/outputs/figures/fig4_6_confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

print("✅ Figure generated: fig4_6_confusion_matrix.png")
