import shutil
from pathlib import Path

brain_dir = Path("C:/Users/Gis/.gemini/antigravity/brain/437b0af7-85ec-4a26-98ad-07142b379702")
workspace_dir = Path("g:/My Drive/xai_lime_vs_shap")

# List of figures to copy
figs_to_copy = [
    ("outputs/figures/xai/global_shap_feature_importance.png", "global_shap_feature_importance.png"),
    ("outputs/figures/fig4_8_shap_beeswarm.png", "fig4_8_shap_beeswarm.png"),
    ("outputs/figures/xai/faithfulness_evaluation_curves.png", "faithfulness_evaluation_curves.png"),
    ("outputs/figures/xai/metrics_distributions.png", "metrics_distributions.png")
]

print("=== Copying Figures to Brain Artifacts Directory ===")
for src_rel, dest_name in figs_to_copy:
    src_path = workspace_dir / src_rel
    dest_path = brain_dir / dest_name
    
    if src_path.exists():
        try:
            shutil.copy(src_path, dest_path)
            print(f"Success: {src_rel} -> {dest_path}")
        except Exception as e:
            print(f"Error copying {src_rel}: {e}")
    else:
        print(f"Warning: Source file not found: {src_path}")
