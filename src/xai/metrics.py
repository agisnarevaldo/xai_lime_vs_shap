import time
import numpy as np
from scipy.stats import spearmanr

def clean_token(token):
    """Helper to clean and normalize a text token, ignoring standalone punctuation and grammar stopwords."""
    if not isinstance(token, str):
        return ""
    tok = token.strip().lower()
    # 1. Filter out standalone punctuation (tokens without any alphanumeric characters)
    if tok and not any(c.isalnum() for c in tok):
        return ""
    # 2. Filter out functional Indonesian grammar stopwords (excludes critical negation/contrastive words)
    XAI_STOPWORDS = {
        "dan", "di", "yang", "sudah", "dengan", "ada", "ke", "dari", "ini", "itu",
        "saya", "kamu", "dia", "mereka", "kita", "kami", "untuk", "akan", "telah",
        "oleh", "pada", "adalah", "bahwa", "saja", "juga", "atau", "sebagai", "dalam",
        "saat", "karena", "tersebut", "tentang", "serta", "yaitu", "seperti", "telah",
        "bagi", "secara", "hanya", "ia", "sih", "kah", "pun", "deh", "kok", "lah", "tuh"
    }
    if tok in XAI_STOPWORDS:
        return ""
    return tok



def extract_lime_features(lime_exp, label_idx, top_k=5):
    """
    Extracts top-k features from a LIME explanation for a specific label index.
    Returns list of tuples: (cleaned_word, weight) sorted by absolute weight descending.
    """
    feat_list = lime_exp.as_list(label=label_idx)
    cleaned = []
    for word, weight in feat_list:
        w_clean = clean_token(word)
        if w_clean:
            cleaned.append((w_clean, weight))
    # Sort by absolute weight descending
    cleaned = sorted(cleaned, key=lambda x: abs(x[1]), reverse=True)
    return cleaned[:top_k]

def extract_shap_features(shap_val_for_class, top_k=5):
    """
    Extracts top-k features from a SHAP explanation slice (for a target class).
    Averages/aggregates duplicate tokens by summing weights.
    Returns list of tuples: (cleaned_word, weight) sorted by absolute weight descending.
    """
    tokens = shap_val_for_class.data
    weights = shap_val_for_class.values
    
    agg = {}
    for token, weight in zip(tokens, weights):
        t_clean = clean_token(token)
        if t_clean:
            agg[t_clean] = agg.get(t_clean, 0.0) + weight
            
    cleaned_agg = list(agg.items())
    cleaned_agg = sorted(cleaned_agg, key=lambda x: abs(x[1]), reverse=True)
    return cleaned_agg[:top_k]

def calculate_jaccard_similarity(lime_feats, shap_feats):
    """
    Calculates Jaccard Similarity between LIME and SHAP top features.
    lime_feats: list of tuples (word, weight)
    shap_feats: list of tuples (word, weight)
    """
    set_lime = {item[0] for item in lime_feats}
    set_shap = {item[0] for item in shap_feats}
    
    union = set_lime | set_shap
    if not union:
        return 0.0
    return len(set_lime & set_shap) / len(union)

def calculate_spearman_correlation(lime_feats, shap_feats):
    """
    Calculates Spearman Rank Correlation between LIME and SHAP weights
    for features that are present in both explanations.
    """
    # Create dicts for easy lookup
    dict_lime = dict(lime_feats)
    dict_shap = dict(shap_feats)
    
    # Find intersection of words
    common_words = list(dict_lime.keys() & dict_shap.keys())
    if len(common_words) < 2:
        return np.nan # Not enough points to calculate rank correlation
        
    lime_ranks = [dict_lime[w] for w in common_words]
    shap_ranks = [dict_shap[w] for w in common_words]
    
    correlation, _ = spearmanr(lime_ranks, shap_ranks)
    return correlation

def remove_tokens(text, tokens_to_remove):
    """Removes specified tokens from a space-separated string."""
    words = text.split()
    remove_set = {clean_token(t) for t in tokens_to_remove}
    remaining = [w for w in words if clean_token(w) not in remove_set]
    return " ".join(remaining)

def keep_only_tokens(text, tokens_to_keep):
    """Keeps only specified tokens in a space-separated string, removing others."""
    words = text.split()
    keep_set = {clean_token(t) for t in tokens_to_keep}
    kept = [w for w in words if clean_token(w) in keep_set]
    return " ".join(kept)

def calculate_comprehensiveness(original_prob, perturbed_prob):
    """
    Comprehensiveness = P(original) - P(original without top-k features)
    Higher is better (faithful explainer captures critical tokens).
    """
    return original_prob - perturbed_prob


def calculate_sufficiency(original_prob, kept_prob):
    """
    Sufficiency = P(original) - P(top-k features only)
    Lower (closer to 0) is better (kept features are sufficient).
    """
    return original_prob - kept_prob
