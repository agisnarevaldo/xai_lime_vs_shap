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
        # Create directory if it doesn't exist
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

def plot_aopc_curves(k_values, aopc_lime, aopc_shap, save_path=None):
    """
    Plots the Area Over Perturbation Curve (Comprehensiveness) comparison.
    A higher AOPC indicates a more faithful explainer (faster prediction drop).
    """
    plt.figure(figsize=(8, 5))
    
    plt.plot(k_values, aopc_lime, marker='o', linewidth=2, color='#9bbb59', label='LIME (Comprehensiveness)')
    plt.plot(k_values, aopc_shap, marker='s', linewidth=2, color='#c0504d', label='SHAP (Comprehensiveness)')
    
    plt.title("Comprehensiveness (AOPC) Curves - Deletion Method")
    plt.xlabel("Number of Deleted Tokens (k)")
    plt.ylabel("Drop in Prediction Probability")
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved AOPC curves plot to: {save_path}")
        
    plt.close()
