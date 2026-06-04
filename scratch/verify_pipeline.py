import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import shap
from lime.lime_text import LimeTextExplainer

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.xai.explainer import PredictProbaWrapper, load_sentiment_model_and_tokenizer
from src.xai.metrics import (
    extract_lime_features, extract_shap_features,
    calculate_jaccard_similarity, calculate_spearman_correlation,
    remove_tokens, keep_only_tokens,
    calculate_comprehensiveness, calculate_sufficiency
)
from src.xai.visualizer import plot_local_comparison, plot_metric_distributions, plot_aopc_curves

def verify_pipeline():
    print("=== START PIPELINE VERIFICATION ===")
    
    # Define paths
    MODEL_DIR = "outputs/finetuning_indobert/model/final_indobert_sentiment"
    TOKENIZER_DIR = "outputs/finetuning_indobert/tokenizer/indobert_tokenizer"
    TEST_DATA_PATH = "data/processed/tokopedia_reviews_binary_test.csv"
    
    # 1. Load Model and Tokenizer (CPU)
    print("1. Loading model & tokenizer...")
    device = torch.device("cpu")
    model, tokenizer = load_sentiment_model_and_tokenizer(MODEL_DIR, TOKENIZER_DIR, device=device)
    predict_proba = PredictProbaWrapper(model, tokenizer, device=device)
    
    # 2. Load dataset
    print("2. Loading dataset...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    sample_text = str(df_test.iloc[0]["review_text_clean"])
    true_label = df_test.iloc[0]["sentiment_label"]
    
    print(f"Sample review: '{sample_text}'")
    print(f"True label: {true_label}")
    
    # 3. Predict probability
    probs = predict_proba([sample_text])[0]
    pred_idx = np.argmax(probs)
    pred_label = "Positif" if pred_idx == 1 else "Negatif"
    print(f"Predicted label: {pred_label} with probability {probs[pred_idx]:.4f}")
    
    # 4. LIME explanation
    print("4. Testing LIME explanation...")
    class_names = ["Negatif", "Positif"]
    explainer_lime = LimeTextExplainer(class_names=class_names)
    exp_lime = explainer_lime.explain_instance(
        sample_text,
        predict_proba,
        num_features=5,
        num_samples=100,  # low samples for fast verification
        labels=[pred_idx]
    )
    lime_feats = extract_lime_features(exp_lime, label_idx=pred_idx, top_k=5)
    print(f"LIME Features: {lime_feats}")
    
    # 5. SHAP explanation
    print("5. Testing SHAP explanation...")
    bg_texts = df_test["review_text_clean"].sample(5, random_state=42).astype(str).tolist()
    masker = shap.maskers.Text(tokenizer=" ")
    explainer_shap = shap.Explainer(predict_proba, masker, output_names=class_names)
    shap_val = explainer_shap([sample_text])
    shap_feats = extract_shap_features(shap_val[0, :, pred_idx], top_k=5)
    print(f"SHAP Features: {shap_feats}")
    
    # 6. Compare metrics
    jaccard = calculate_jaccard_similarity(lime_feats, shap_feats)
    spearman = calculate_spearman_correlation(lime_feats, shap_feats)
    print(f"Jaccard Similarity: {jaccard:.4f}")
    print(f"Spearman Correlation: {spearman:.4f}")
    
    # 7. Test comprehensiveness and sufficiency
    lime_words = [w for w, _ in lime_feats]
    removed_text = remove_tokens(sample_text, lime_words[:2])
    perturbed_prob = predict_proba([removed_text])[0][pred_idx]
    comp = calculate_comprehensiveness(probs[pred_idx], perturbed_prob)
    
    kept_text = keep_only_tokens(sample_text, lime_words[:2])
    kept_prob = predict_proba([kept_text])[0][pred_idx] if len(kept_text.strip()) > 0 else 0.5
    suff = calculate_sufficiency(probs[pred_idx], kept_prob)
    
    print(f"Comprehensiveness (k=2): {comp:.4f}")
    print(f"Sufficiency (k=2): {suff:.4f}")
    
    # 8. Test Visualizations
    print("8. Testing visualization exports...")
    os.makedirs("outputs/figures/xai_verify", exist_ok=True)
    plot_local_comparison(
        lime_feats,
        shap_feats,
        sample_id=999,
        true_label=true_label,
        pred_label=pred_label,
        save_path="outputs/figures/xai_verify/test_local_comparison.png"
    )
    
    plot_metric_distributions(
        jaccard_scores=[jaccard, jaccard * 0.9, jaccard * 1.1],
        runtimes_lime=[0.5, 0.4, 0.6],
        runtimes_shap=[5.2, 4.8, 5.5],
        save_path="outputs/figures/xai_verify/test_metrics_distributions.png"
    )
    
    plot_aopc_curves(
        k_values=[1, 2, 3],
        aopc_lime=[0.1, 0.3, 0.5],
        aopc_shap=[0.15, 0.35, 0.55],
        save_path="outputs/figures/xai_verify/test_aopc_curves.png"
    )
    
    print("=== PIPELINE VERIFICATION SUCCESSFUL! ===")

if __name__ == "__main__":
    verify_pipeline()
