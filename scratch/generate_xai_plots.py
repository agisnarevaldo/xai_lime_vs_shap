
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lime
from lime.lime_text import LimeTextExplainer
import shap
import os

# Config
MODEL_DIR = "d:/xai_lime_vs_shap/outputs/indobert_finetuned"
CLASS_NAMES = ["Positif", "Negatif", "Netral"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIGURE_DIR = "d:/xai_lime_vs_shap/outputs/figures"

print(f"Loading model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

def predict_proba(texts):
    if isinstance(texts, str): texts = [texts]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.cpu().numpy()

# 1. LIME Explanation (Netral Sample)
print("Generating LIME plot...")
explainer_lime = LimeTextExplainer(class_names=CLASS_NAMES)
netral_text = "Biasa aja sih, kualitas standar cuma adminnya ramah."
exp = explainer_lime.explain_instance(netral_text, predict_proba, num_features=10, labels=[2])

# Manual bar plot for LIME (since exp.as_pyplot_figure() might behave differently in headless)
plt.figure(figsize=(10, 6))
vals = exp.as_list(label=2)
features, weights = zip(*vals)
features = [str(f) for f in features]
colors = ['green' if w > 0 else 'red' for w in weights]
plt.barh(features, weights, color=colors)
plt.title(f"LIME Explanation for Class 'Netral'\nText: {netral_text}", fontweight='bold')
plt.xlabel("Local Weight (Impact)")
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig4_7_lime_explanation.png'), dpi=300)
plt.close()

# 2. SHAP Text Plot (Positif Sample)
print("Generating SHAP Text plot...")
# SHAP Partition explainer for text
explainer_shap = shap.Explainer(predict_proba, tokenizer, output_names=CLASS_NAMES)
positif_text = "Produk original, packing aman, seller ramah."
shap_values = explainer_shap([positif_text])

# SHAP bar plot for this sample (Class Positif - index 0)
plt.figure(figsize=(10, 5))
shap.plots.bar(shap_values[0, :, 0], show=False)
plt.title("SHAP Feature Importance (Local - Positif)", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig4_8_shap_local_bar.png'), dpi=300)
plt.close()

# 3. SHAP Beeswarm (Global Subset)
print("Generating SHAP Beeswarm plot...")
df = pd.read_csv("d:/xai_lime_vs_shap/data/processed/tokopedia_reviews_clean.csv")
# Take a small balanced subset for global explanation
subset_df = df.groupby('label').apply(lambda x: x.sample(min(len(x), 20))).reset_index(drop=True)
subset_texts = subset_df['text'].tolist()

shap_values_batch = explainer_shap(subset_texts)

# Beeswarm per class
# Positif (Class 0)
plt.figure(figsize=(12, 6))
plt.title("SHAP Global Impact (Class: Positif)", fontweight='bold')
shap.plots.beeswarm(shap_values_batch[:, :, 0], show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig4_9_shap_beeswarm_positif.png'), dpi=300)
plt.close()

# Netral/Negatif (Class 2 - Netral)
plt.figure(figsize=(12, 6))
plt.title("SHAP Global Impact (Class: Netral)", fontweight='bold')
shap.plots.beeswarm(shap_values_batch[:, :, 2], show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig4_10_shap_beeswarm_netral.png'), dpi=300)
plt.close()

print("All XAI plots generated successfully.")
