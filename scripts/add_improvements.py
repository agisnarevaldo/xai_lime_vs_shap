"""
Script: add_improvements.py
Tujuan: Menambahkan 3 perbaikan ke notebook penelitian XAI:
  1. Error Analysis ke notebook 03
  2. 3 sampel per kelas (LIME + SHAP) ke notebook 04
  3. Faithfulness Metrics ke notebook 04
"""

import nbformat
from pathlib import Path

BASE = Path(__file__).parent.parent
NB03 = BASE / "notebooks" / "03_modeling_prdect_multiclass.ipynb"
NB04 = BASE / "notebooks" / "04_explainability_prdect.ipynb"

# =============================================================================
# HELPER
# =============================================================================

def make_md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)

def make_code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)

# =============================================================================
# PERBAIKAN 1 — Error Analysis (Notebook 03)
# =============================================================================

ERROR_ANALYSIS_MD = """## 8. Error Analysis — Analisis Misklasifikasi

Analisis ini bertujuan untuk:
1. Mengidentifikasi pola kesalahan prediksi model (pasangan kelas yang sering tertukar)
2. Menganalisis secara kualitatif contoh review yang salah diklasifikasi
3. Memahami kelemahan semantik model berdasarkan distribusi error

Analisis misklasifikasi merupakan bagian penting dari evaluasi model NLP
untuk memahami ambiguitas antar kelas emosi pada dataset berbahasa Indonesia."""

ERROR_ANALYSIS_CODE1 = """\
## 8a. Ambil Prediksi Lengkap dengan Probabilitas
import pandas as pd
import numpy as np

classifier.eval()
with torch.no_grad():
    X_test_t   = torch.tensor(X_test_emb, dtype=torch.float32).to(DEVICE)
    logits_all = classifier(X_test_t)
    probs_all  = torch.softmax(logits_all, dim=-1).cpu().numpy()
    preds_all  = np.argmax(probs_all, axis=1)

# Buat DataFrame lengkap
df_test_ea = pd.read_csv(PROC_DIR / 'prdect_test.csv')
df_error = df_test_ea.copy()
df_error['true_label']   = y_test
df_error['pred_label']   = preds_all
df_error['true_emotion'] = [CLASS_NAMES[i] for i in y_test]
df_error['pred_emotion'] = [CLASS_NAMES[i] for i in preds_all]
df_error['confidence']   = probs_all.max(axis=1)
df_error['is_correct']   = (df_error['true_label'] == df_error['pred_label'])

# Filter misklasifikasi
df_mis = df_error[~df_error['is_correct']].copy()
n_mis  = len(df_mis)
n_tot  = len(df_error)

print("=== Error Analysis: Analisis Misklasifikasi ===")
print(f"Total sampel test    : {n_tot}")
print(f"Benar diklasifikasi  : {n_tot - n_mis} ({(n_tot-n_mis)/n_tot*100:.1f}%)")
print(f"Salah diklasifikasi  : {n_mis} ({n_mis/n_tot*100:.1f}%)")

# Frekuensi pasangan misklasifikasi
print("\\n--- Pola Misklasifikasi (True → Predicted) ---")
confusion_pairs = (
    df_mis
    .groupby(['true_emotion', 'pred_emotion'])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
)
print(confusion_pairs.to_string(index=False))

# Top 10 error dengan confidence tertinggi (paling meyakinkan tapi salah)
print("\\n--- Top 10 Misklasifikasi (Confidence Tertinggi) ---")
top_errors = df_mis.sort_values('confidence', ascending=False).head(10)
for _, row in top_errors.iterrows():
    txt = str(row['review_clean'])
    preview = txt[:100] + "..." if len(txt) > 100 else txt
    print(f"  True: {row['true_emotion']:8s} | Pred: {row['pred_emotion']:8s} | Conf: {row['confidence']:.3f}")
    print(f"  Review: {preview}")
    print()
"""

ERROR_ANALYSIS_CODE2 = """\
## 8b. Visualisasi Pola Misklasifikasi (Heatmap)
import matplotlib.pyplot as plt
import seaborn as sns

pivot_err = (
    confusion_pairs
    .pivot(index='true_emotion', columns='pred_emotion', values='count')
    .fillna(0).astype(int)
)

plt.figure(figsize=(8, 5))
sns.heatmap(
    pivot_err, annot=True, fmt='d', cmap='Reds',
    linewidths=0.5, cbar_kws={'label': 'Jumlah Misklasifikasi'}
)
plt.title(
    'Pola Misklasifikasi — IndoBERT + MLP\\n'
    '(Baris: Label Benar, Kolom: Label Prediksi)',
    fontsize=11
)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=30); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(MODEL_SAVE_DIR / 'error_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Error heatmap disimpan.")

# Simpan error analysis ke CSV
error_path = MODEL_SAVE_DIR / 'error_analysis.csv'
df_mis[['review_clean', 'true_emotion', 'pred_emotion', 'confidence']].to_csv(
    error_path, index=False
)
print(f"Error analysis CSV disimpan: {error_path}")
"""

# =============================================================================
# PERBAIKAN 2 — 3 Sampel Per Kelas (Notebook 04)
# =============================================================================

MULTI_SAMPLE_MD = """## 4. Load Test Data & Pilih 3 Sampel Per Emosi

Untuk analisis XAI yang lebih representatif, dipilih **3 sampel per kelas emosi**:
- **Sampel A** (High Confidence Correct): Prediksi benar dengan confidence ≥ 0.80
- **Sampel B** (Medium Confidence Correct): Prediksi benar dengan confidence < 0.80
- **Sampel C** (Misclassified): Model salah prediksi

Strategi ini memungkinkan analisis perbandingan penjelasan LIME dan SHAP
pada kondisi prediksi yang berbeda, termasuk kasus di mana model gagal."""

MULTI_SAMPLE_CODE = """\
## 4. Load Test Data & Pilih 3 Sampel Per Emosi
import pandas as pd
import numpy as np

df_test = pd.read_csv(PROC_DIR / 'prdect_test.csv')
print(f"Test set: {len(df_test)} rows")
print("\\nDistribusi:")
print(df_test['emotion_label'].value_counts())

# ---- Prediksi seluruh test set ----
all_texts = df_test['review_clean'].tolist()
all_probs = predict_proba(all_texts)
all_preds = np.argmax(all_probs, axis=1)
all_conf  = all_probs.max(axis=1)

df_test = df_test.copy()
df_test['true_id']    = [EMOTION_MAP[e] for e in df_test['emotion_label']]
df_test['pred_id']    = all_preds
df_test['confidence'] = all_conf
df_test['correct']    = (df_test['true_id'] == df_test['pred_id'])

SEED = 42

def pick_samples(df, emotion_name, seed=42):
    \"\"\"Pilih 3 sampel representatif per kelas emosi.\"\"\"
    sub = df[df['emotion_label'] == emotion_name].copy()
    correct  = sub[sub['correct'] == True].sort_values('confidence', ascending=False)
    wrong    = sub[sub['correct'] == False].sort_values('confidence', ascending=False)

    samples = {}
    # A: High confidence correct (conf >= 0.80)
    hi = correct[correct['confidence'] >= 0.80]
    samples['A_correct_high'] = hi.iloc[0] if len(hi) > 0 else correct.iloc[0]

    # B: Medium confidence correct (conf < 0.80)
    mid = correct[correct['confidence'] < 0.80]
    samples['B_correct_mid'] = mid.iloc[0] if len(mid) > 0 else (
        correct.iloc[1] if len(correct) > 1 else correct.iloc[0]
    )

    # C: Misclassified
    samples['C_misclassified'] = wrong.iloc[0] if len(wrong) > 0 else correct.iloc[-1]

    return samples

# Pilih sampel untuk semua kelas
SAMPLE_SETS = {}
for cls in CLASS_NAMES:
    SAMPLE_SETS[cls] = pick_samples(df_test, cls)
    print(f"\\n[{cls.upper()}]")
    for stype, row in SAMPLE_SETS[cls].items():
        pred_name = CLASS_NAMES[int(row['pred_id'])]
        print(f"  {stype}: true={row['emotion_label']}, pred={pred_name}, conf={row['confidence']:.3f}")
        print(f"    Text: {str(row['review_clean'])[:80]}...")
"""

# =============================================================================
# PERBAIKAN 3 — Faithfulness Metrics (Notebook 04)
# =============================================================================

FAITHFULNESS_MD = """## 10. Faithfulness Metrics — Perbandingan Kuantitatif LIME vs SHAP

Untuk memperkuat perbandingan LIME dan SHAP secara kuantitatif (sesuai pendekatan
penelitian eksperimen komputasional), digunakan tiga metrik:

| Metrik | Deskripsi |
|:---|:---|
| **Top-K Agreement Rate** | Jaccard similarity antara top-K fitur LIME dan SHAP |
| **Spearman Rank Correlation** | Korelasi peringkat fitur antara LIME dan SHAP |
| **Comprehensiveness Score** | Penurunan confidence saat top-K fitur dihapus (masking) |

Metrik-metrik ini memungkinkan evaluasi objektif terhadap kualitas penjelasan
yang dihasilkan oleh masing-masing teknik XAI."""

FAITHFULNESS_CODE = """\
## 10. Faithfulness Metrics — Perbandingan Kuantitatif LIME vs SHAP
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

TOP_K = 5  # Jumlah fitur teratas yang dibandingkan

results = []

for cls_name, samples in SAMPLE_SETS.items():
    for stype, row in samples.items():
        text = str(row['review_clean'])
        true_cls = row['emotion_label']

        # ---- LIME: ambil top-K fitur ----
        try:
            lime_exp = lime_explainer.explain_instance(
                text, predict_proba,
                num_features=TOP_K * 2,
                num_samples=200,
                labels=[EMOTION_MAP[true_cls]]
            )
            lime_feats = lime_exp.as_list(label=EMOTION_MAP[true_cls])
            lime_words  = [f[0] for f in lime_feats[:TOP_K]]
            lime_scores = np.array([abs(f[1]) for f in lime_feats[:TOP_K]])
        except Exception:
            lime_words, lime_scores = [], np.array([])

        # ---- SHAP: ambil top-K fitur ----
        try:
            tokens = text.split()
            shap_exp_single = shap_explainer([text])
            shap_vals = shap_exp_single.values[0, :, EMOTION_MAP[true_cls]]
            shap_words_all  = shap_exp_single.data[0]
            sorted_idx  = np.argsort(np.abs(shap_vals))[::-1]
            shap_words  = [shap_words_all[i] for i in sorted_idx[:TOP_K]]
            shap_scores = np.abs(shap_vals)[sorted_idx[:TOP_K]]
        except Exception:
            shap_words, shap_scores = [], np.array([])

        if len(lime_words) == 0 or len(shap_words) == 0:
            continue

        # ---- Metrik 1: Top-K Agreement Rate (Jaccard) ----
        lime_set = set(lime_words)
        shap_set = set(shap_words)
        union    = lime_set | shap_set
        inter    = lime_set & shap_set
        jaccard  = len(inter) / len(union) if union else 0.0

        # ---- Metrik 2: Spearman Rank Correlation ----
        # Buat vektor skor untuk kata yang ada di keduanya
        common = list(inter)
        if len(common) >= 2:
            lime_rank_scores = np.array([
                lime_scores[lime_words.index(w)] if w in lime_words else 0
                for w in common
            ])
            shap_rank_scores = np.array([
                shap_scores[shap_words.index(w)] if w in shap_words else 0
                for w in common
            ])
            rho, pval = spearmanr(lime_rank_scores, shap_rank_scores)
        else:
            rho, pval = float('nan'), float('nan')

        # ---- Metrik 3: Comprehensiveness (masking top-K LIME) ----
        orig_conf = predict_proba([text])[0][EMOTION_MAP[true_cls]]
        masked_text_lime = ' '.join([
            '[MASK]' if w in lime_set else w for w in text.split()
        ])
        masked_conf_lime = predict_proba([masked_text_lime])[0][EMOTION_MAP[true_cls]]
        comp_lime = float(orig_conf - masked_conf_lime)

        # Comprehensiveness (masking top-K SHAP)
        masked_text_shap = ' '.join([
            '[MASK]' if w in shap_set else w for w in text.split()
        ])
        masked_conf_shap = predict_proba([masked_text_shap])[0][EMOTION_MAP[true_cls]]
        comp_shap = float(orig_conf - masked_conf_shap)

        results.append({
            'class'           : cls_name,
            'sample_type'     : stype,
            'jaccard_topk'    : round(jaccard, 3),
            'spearman_rho'    : round(rho, 3) if not np.isnan(rho) else None,
            'comp_lime'       : round(comp_lime, 3),
            'comp_shap'       : round(comp_shap, 3),
            'lime_top_words'  : ', '.join(lime_words),
            'shap_top_words'  : ', '.join(shap_words),
        })
        print(f"[{cls_name}/{stype}] Jaccard={jaccard:.3f} | Spearman={rho:.3f} | Comp(LIME)={comp_lime:.3f} | Comp(SHAP)={comp_shap:.3f}")

df_metrics = pd.DataFrame(results)
print("\\n=== Tabel Faithfulness Metrics ===")
print(df_metrics[['class','sample_type','jaccard_topk','spearman_rho','comp_lime','comp_shap']].to_string(index=False))
"""

FAITHFULNESS_SUMMARY_CODE = """\
## 10b. Ringkasan Agregat Faithfulness Metrics
import matplotlib.pyplot as plt
import numpy as np

# Rata-rata per metrik
avg_jaccard   = df_metrics['jaccard_topk'].mean()
avg_spearman  = df_metrics['spearman_rho'].dropna().mean()
avg_comp_lime = df_metrics['comp_lime'].mean()
avg_comp_shap = df_metrics['comp_shap'].mean()

print("\\n=== Ringkasan Agregat Faithfulness Metrics ===")
print(f"  Top-{TOP_K} Agreement Rate (Jaccard) avg : {avg_jaccard:.3f}")
print(f"  Spearman Rank Correlation avg          : {avg_spearman:.3f}")
print(f"  Comprehensiveness LIME avg             : {avg_comp_lime:.3f}")
print(f"  Comprehensiveness SHAP avg             : {avg_comp_shap:.3f}")

# Tabel ringkasan untuk laporan
summary = {
    'Metrik': [
        f'Top-{TOP_K} Agreement Rate (Jaccard)',
        'Spearman Rank Correlation',
        'Comprehensiveness LIME',
        'Comprehensiveness SHAP',
    ],
    'Nilai Rata-rata': [
        f"{avg_jaccard:.3f}",
        f"{avg_spearman:.3f}",
        f"{avg_comp_lime:.3f}",
        f"{avg_comp_shap:.3f}",
    ],
    'Interpretasi': [
        'Tinggi = LIME & SHAP sepakat pada fitur penting',
        'Tinggi = urutan kepentingan fitur mirip',
        'Tinggi = fitur LIME benar-benar penting bagi model',
        'Tinggi = fitur SHAP benar-benar penting bagi model',
    ]
}
df_summary = pd.DataFrame(summary)
print("\\n")
print(df_summary.to_string(index=False))

# Simpan ke CSV
metrics_path = VIZ_DIR.parent / 'faithfulness_metrics.csv'
df_metrics.to_csv(metrics_path, index=False)
print(f"\\nFaithfulness metrics disimpan: {metrics_path}")

# Visualisasi perbandingan Comprehensiveness
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart per kelas — Comprehensiveness LIME vs SHAP
cls_avg = df_metrics.groupby('class')[['comp_lime', 'comp_shap']].mean().reset_index()
x = np.arange(len(cls_avg))
w = 0.35
axes[0].bar(x - w/2, cls_avg['comp_lime'], w, label='LIME', color='#2196F3', alpha=0.8)
axes[0].bar(x + w/2, cls_avg['comp_shap'], w, label='SHAP', color='#E91E63', alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(cls_avg['class'], rotation=20)
axes[0].set_title('Comprehensiveness per Kelas Emosi')
axes[0].set_ylabel('Comprehensiveness Score'); axes[0].legend()
axes[0].axhline(0, color='gray', linestyle='--', lw=0.8)

# Box plot Jaccard Agreement
df_metrics.boxplot(column='jaccard_topk', by='class', ax=axes[1])
axes[1].set_title(f'Top-{TOP_K} Agreement Rate (Jaccard) per Kelas')
axes[1].set_xlabel('Kelas Emosi'); axes[1].set_ylabel('Jaccard Similarity')
plt.suptitle('')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'faithfulness_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Faithfulness metrics plot disimpan.")
"""

# =============================================================================
# MAIN — Terapkan ke Notebook
# =============================================================================

def patch_notebook03():
    nb = nbformat.read(NB03, as_version=4)

    # Cari posisi setelah Section 7 Evaluation (cell "Confusion matrix saved")
    insert_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'Confusion matrix saved' in cell.source:
            insert_idx = i + 1
            break

    if insert_idx is None:
        # fallback: insert sebelum cell terakhir (Save Assets)
        insert_idx = len(nb.cells) - 1
        print(f"[NB03] Fallback: sisipkan di posisi {insert_idx}")
    else:
        print(f"[NB03] Sisipkan Error Analysis di posisi {insert_idx}")

    new_cells = [
        make_md(ERROR_ANALYSIS_MD),
        make_code(ERROR_ANALYSIS_CODE1),
        make_code(ERROR_ANALYSIS_CODE2),
    ]
    for j, c in enumerate(new_cells):
        nb.cells.insert(insert_idx + j, c)

    nbformat.write(nb, NB03)
    print(f"[NB03] Selesai — {len(nb.cells)} cells total.")


def patch_notebook04():
    nb = nbformat.read(NB04, as_version=4)

    # ---- Perbaikan 2: Ganti Section 4 (Load Test Data) ----
    sec4_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'groupby' in cell.source and 'emotion_label' in cell.source:
            sec4_idx = i
            break

    if sec4_idx is not None:
        # Ganti markdown + code Section 4
        nb.cells[sec4_idx - 1] = make_md(MULTI_SAMPLE_MD)
        nb.cells[sec4_idx]     = make_code(MULTI_SAMPLE_CODE)
        print(f"[NB04] Section 4 diganti di posisi {sec4_idx}")
    else:
        print("[NB04] WARNING: Section 4 tidak ditemukan, lewati perbaikan 2")

    # ---- Perbaikan 3: Tambahkan Faithfulness Metrics sebelum cell terakhir ----
    insert_idx = len(nb.cells) - 1  # sebelum summary terakhir
    new_cells = [
        make_md(FAITHFULNESS_MD),
        make_code(FAITHFULNESS_CODE),
        make_code(FAITHFULNESS_SUMMARY_CODE),
    ]
    for j, c in enumerate(new_cells):
        nb.cells.insert(insert_idx + j, c)

    nbformat.write(nb, NB04)
    print(f"[NB04] Selesai — {len(nb.cells)} cells total.")


if __name__ == "__main__":
    print("=" * 60)
    print("Mulai patching notebook XAI...")
    print("=" * 60)
    patch_notebook03()
    print()
    patch_notebook04()
    print()
    print("=" * 60)
    print("Semua perbaikan berhasil diterapkan!")
    print(f"  Notebook 03: {NB03}")
    print(f"  Notebook 04: {NB04}")
    print("=" * 60)
    print()
    print("Langkah selanjutnya:")
    print("  1. Upload notebook ke Google Colab")
    print("  2. Run semua cell dari atas")
    print("  3. Pastikan GPU aktif (Runtime > Change runtime type > T4 GPU)")
