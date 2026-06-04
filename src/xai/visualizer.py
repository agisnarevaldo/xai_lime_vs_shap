import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for professional research graphics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def plot_local_comparison(lime_feats, shap_feats, sample_id, true_label, pred_label, save_path=None):
    """
    Plots a side-by-side horizontal bar chart comparing LIME and SHAP attributions
    for a single text sample.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    
    # 1. LIME Plot
    lime_words = [item[0] for item in lime_feats][::-1]
    lime_weights = [item[1] for item in lime_feats][::-1]
    lime_colors = ['#3182bd' if w > 0 else '#de2d26' for w in lime_weights]
    
    axes[0].barh(lime_words, lime_weights, color=lime_colors, edgecolor='grey', height=0.6)
    axes[0].axvline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_title(f"LIME Explanations (Target: {pred_label})")
    axes[0].set_xlabel("Feature Weight")
    
    # 2. SHAP Plot
    shap_words = [item[0] for item in shap_feats][::-1]
    shap_weights = [item[1] for item in shap_feats][::-1]
    shap_colors = ['#3182bd' if w > 0 else '#de2d26' for w in shap_weights]
    
    axes[1].barh(shap_words, shap_weights, color=shap_colors, edgecolor='grey', height=0.6)
    axes[1].axvline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_title(f"SHAP Explanations (Target: {pred_label})")
    axes[1].set_xlabel("Shapley Value")
    
    # Global Title
    fig.suptitle(
        f"XAI Local Comparison - Sample #{sample_id}\n(True Label: {true_label} | Pred Label: {pred_label})", 
        y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved local comparison plot to: {save_path}")
        
    plt.close()

def plot_metric_distributions(jaccard_scores, runtimes_lime, runtimes_shap, save_path=None):
    """
    Plots the distributions of Jaccard similarities and compares execution times.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Jaccard similarity distribution
    sns.histplot(jaccard_scores, bins=10, kde=True, ax=axes[0], color='#4f81bd')
    axes[0].axvline(np.mean(jaccard_scores), color='red', linestyle='--', linewidth=1.5, 
                     label=f'Mean: {np.mean(jaccard_scores):.4f}')
    axes[0].set_title("Distribution of Jaccard Similarity between LIME and SHAP")
    axes[0].set_xlabel("Jaccard Similarity Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    
    # 2. Runtime comparison
    runtimes = [runtimes_lime, runtimes_shap]
    sns.boxplot(data=runtimes, ax=axes[1], palette=['#9bbb59', '#c0504d'])
    axes[1].set_xticklabels(['LIME', 'SHAP'])
    axes[1].set_title("Execution Time Comparison per Sentence")
    axes[1].set_ylabel("Time (seconds)")
    
    # Annotate mean runtimes
    mean_lime = np.mean(runtimes_lime)
    mean_shap = np.mean(runtimes_shap)
    axes[1].text(0, mean_lime, f"Mean: {mean_lime:.2f}s", ha='center', va='bottom', fontweight='bold')
    axes[1].text(1, mean_shap, f"Mean: {mean_shap:.2f}s", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics distribution plot to: {save_path}")
        
    plt.close()

def plot_faithfulness_curves(k_values, comp_lime, comp_shap, suff_lime, suff_shap, save_path=None):
    """
    Plots the Comprehensiveness and Sufficiency curves side-by-side (1x2 subplot).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Comprehensiveness Plot
    axes[0].plot(k_values, comp_lime, marker='o', linewidth=2, color='#9bbb59', label='LIME')
    axes[0].plot(k_values, comp_shap, marker='s', linewidth=2, color='#c0504d', label='SHAP')
    axes[0].set_title("Comprehensiveness (AOPC) - Deletion (Higher is Better)")
    axes[0].set_xlabel("Number of Deleted Tokens (k)")
    axes[0].set_ylabel("Drop in True Class Probability")
    axes[0].set_xticks(k_values)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. Sufficiency Plot
    axes[1].plot(k_values, suff_lime, marker='o', linewidth=2, color='#9bbb59', label='LIME')
    axes[1].plot(k_values, suff_shap, marker='s', linewidth=2, color='#c0504d', label='SHAP')
    axes[1].set_title("Sufficiency - Preservation (Lower is Better)")
    axes[1].set_xlabel("Number of Kept Tokens (k)")
    axes[1].set_ylabel("Drop in True Class Probability")
    axes[1].set_xticks(k_values)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    fig.suptitle("XAI Faithfulness Evaluation - Comprehensiveness vs Sufficiency", y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved faithfulness curves to: {save_path}")
        
    plt.close()

def plot_global_shap_importance(top_pos_words, top_neg_words, save_path=None):
    """
    Plots the top 10 positive and top 10 negative words based on mean absolute SHAP values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Positive words plot (pushes towards Positif class)
    words_pos = [item[0] for item in top_pos_words][::-1]
    weights_pos = [item[1] for item in top_pos_words][::-1]
    axes[0].barh(words_pos, weights_pos, color='#3182bd', edgecolor='grey', height=0.6)
    axes[0].set_title("Top 10 Kata Pendorong Sentimen Positif")
    axes[0].set_xlabel("Mean Absolute SHAP Value")
    
    # 2. Negative words plot (pushes towards Negatif class)
    words_neg = [item[0] for item in top_neg_words][::-1]
    weights_neg = [item[1] for item in top_neg_words][::-1]
    axes[1].barh(words_neg, weights_neg, color='#de2d26', edgecolor='grey', height=0.6)
    axes[1].set_title("Top 10 Kata Pendorong Sentimen Negatif")
    axes[1].set_xlabel("Mean Absolute SHAP Value")
    
    fig.suptitle("Global SHAP Feature Importance - Model Sentimen IndoBERT", y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved global SHAP plot to: {save_path}")
        
    plt.close()
