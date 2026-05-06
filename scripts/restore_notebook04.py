import nbformat as nbf
import json
import os

def restore_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- SECTION 0: Header ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# 04 — XAI: LIME vs SHAP pada PRDECT-ID (5-Class Emotion)\n\n"
        "Notebook ini menjelaskan prediksi model hybrid IndoBERT + MLP menggunakan:\n"
        "- **LIME**: Local explanations (per sampel)\n"
        "- **SHAP**: Local & global explanations\n"
        "- **Faithfulness Metrics**: Evaluasi kuantitatif terhadap penjelasan XAI\n\n"
        "**Label:** Happy (0), Love (1), Anger (2), Fear (3), Sadness (4)"
    ))
    
    # --- SECTION 1: Colab Setup ---
    nb.cells.append(nbf.v4.new_code_cell(
        "## 0. Google Colab Setup\n"
        "import sys, os\n\n"
        "IN_COLAB = 'google.colab' in sys.modules\n\n"
        "if IN_COLAB:\n"
        "    print('Google Colab detected — mounting Drive & installing packages ...')\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n\n"
        "    DRIVE_PROJECT = '/content/drive/MyDrive/xai_lime_vs_shap'\n\n"
        "    if os.path.isdir(DRIVE_PROJECT):\n"
        "        os.chdir(DRIVE_PROJECT)\n"
        "        print(f'CWD set to: {DRIVE_PROJECT}')\n"
        "    else:\n"
        "        print(f\"WARNING: '{DRIVE_PROJECT}' not found. Please check your Drive path.\")\n\n"
        "    !pip install -q transformers lime shap datasets accelerate\n"
        "else:\n"
        "    print('Local environment — no Colab setup needed.')"
    ))
    
    # --- SECTION 2: Imports ---
    nb.cells.append(nbf.v4.new_code_cell(
        "## 1. Imports & Path Setup\n"
        "import json, warnings, torch, gc\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from transformers import AutoTokenizer, AutoModel\n"
        "from lime.lime_text import LimeTextExplainer\n"
        "import shap\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from scipy.stats import spearmanr\n\n"
        "warnings.filterwarnings('ignore')\n"
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "SEED = 42\n\n"
        "PROJECT_ROOT  = Path.cwd()\n"
        "MODEL_DIR     = PROJECT_ROOT / 'outputs' / 'prdect_indobert'\n"
        "PROC_DIR      = PROJECT_ROOT / 'data' / 'processed'\n"
        "VIZ_DIR       = MODEL_DIR / 'visualizations'\n"
        "VIZ_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "print(f'Project root : {PROJECT_ROOT}')\n"
        "print(f'Device      : {DEVICE}')"
    ))
    
    # --- SECTION 3: Model ---
    nb.cells.append(nbf.v4.new_code_cell(
        "## 2. Define MLP & Load Weights\n"
        "class MLP(torch.nn.Module):\n"
        "    def __init__(self, input_dim=768, hidden_dim=256, output_dim=5, dropout=0.3):\n"
        "        super().__init__()\n"
        "        self.net = torch.nn.Sequential(\n"
        "            torch.nn.Linear(input_dim, hidden_dim),\n"
        "            torch.nn.ReLU(),\n"
        "            torch.nn.Dropout(dropout),\n"
        "            torch.nn.Linear(hidden_dim, 128),\n"
        "            torch.nn.ReLU(),\n"
        "            torch.nn.Dropout(dropout),\n"
        "            torch.nn.Linear(128, output_dim)\n"
        "        )\n\n"
        "    def forward(self, x):\n"
        "        return self.net(x)\n\n"
        "with open(MODEL_DIR / 'label_map.json') as f:\n"
        "    label_map = json.load(f)\n"
        "ID2EMOTION  = {int(k): v for k, v in label_map['id2label'].items()}\n"
        "CLASS_NAMES = [ID2EMOTION[i] for i in range(len(ID2EMOTION))]\n"
        "N_CLASSES   = len(CLASS_NAMES)\n\n"
        "BERT_MODEL_NAME = 'indobenchmark/indobert-base-p1'\n"
        "tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)\n"
        "bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)\n"
        "bert_model.eval()\n\n"
        "classifier = MLP(output_dim=N_CLASSES).to(DEVICE)\n"
        "classifier.load_state_dict(torch.load(MODEL_DIR / 'mlp_classifier.pt', map_location=DEVICE))\n"
        "classifier.eval()\n\n"
        "print('Models loaded successfully.')"
    ))
    
    # --- SECTION 4: predict_proba (ROBUST VERSION) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "## 3. Define Prediction Wrapper (Robust for SHAP & LIME)\n"
        "def predict_proba(texts):\n"
        "    # 1. Konversi ke list jika input adalah numpy array (SHAP sering mengirimkan ini)\n"
        "    if isinstance(texts, np.ndarray):\n"
        "        texts = texts.tolist()\n"
        "    \n"
        "    # 2. Konversi ke list jika input adalah string tunggal\n"
        "    if isinstance(texts, str):\n"
        "        texts = [texts]\n"
        "        \n"
        "    # 3. Pastikan semua elemen di dalam list adalah string (SHAP kadang mengirimkan objek)\n"
        "    texts = [str(t) for t in texts]\n\n"
        "    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)\n"
        "    \n"
        "    with torch.no_grad():\n"
        "        outputs = bert_model(**inputs)\n"
        "        embeddings = outputs.last_hidden_state[:, 0, :]\n"
        "        logits = classifier(embeddings)\n"
        "        probs = torch.softmax(logits, dim=-1).cpu().numpy()\n"
        "    \n"
        "    return probs"
    ))
    
    # --- SECTION 5: Sampling ---
    nb.cells.append(nbf.v4.new_code_cell(
        "## 4. Load Test Data & Representative Sampling\n"
        "df_test = pd.read_csv(PROC_DIR / 'prdect_test.csv')\n\n"
        "# Filter ulasan kosong agar tidak menyebabkan error pada SHAP\n"
        "df_test = df_test[df_test['review_clean'].str.strip().str.len() > 0].copy()\n\n"
        "def pick_samples(df, label_name):\n"
        "    sub = df[df['emotion_label'] == label_name].copy()\n"
        "    texts = sub['review_clean'].astype(str).tolist()\n"
        "    probs = predict_proba(texts)\n"
        "    sub['pred_id'] = np.argmax(probs, axis=1)\n"
        "    sub['confidence'] = np.max(probs, axis=1)\n"
        "    \n"
        "    correct = sub[sub['pred_id'] == sub['label']].sort_values('confidence', ascending=False)\n"
        "    wrong   = sub[sub['pred_id'] != sub['label']].sort_values('confidence', ascending=False)\n"
        "    \n"
        "    res = {}\n"
        "    if len(correct) > 0:\n"
        "        res['A_correct_high'] = correct.iloc[0]\n"
        "        if len(correct) > 1:\n"
        "            res['B_correct_mid'] = correct.iloc[len(correct)//2]\n"
        "    if len(wrong) > 0:\n"
        "        res['C_misclassified'] = wrong.iloc[0]\n"
        "    return res\n\n"
        "SAMPLE_SETS = {cls: pick_samples(df_test, cls) for cls in CLASS_NAMES}\n"
        "all_selected = [SAMPLE_SETS[cls][stype] for cls in SAMPLE_SETS for stype in SAMPLE_SETS[cls]]\n"
        "samples = pd.DataFrame(all_selected)\n"
        "print(f'Total samples selected: {len(samples)}')"
    ))
    
    # --- SECTION 6: LIME ---
    nb.cells.append(nbf.v4.new_code_cell(
        "## 5. LIME Explanations\n"
        "explainer_lime = LimeTextExplainer(class_names=CLASS_NAMES)\n\n"
        "for idx, row in samples.iterrows():\n"
        "    text = str(row['review_clean'])\n"
        "    # labels=... memastikan LIME menghitung semua probabilitas\n"
        "    exp = explainer_lime.explain_instance(text, predict_proba, num_features=10, labels=list(range(N_CLASSES)))\n"
        "    \n"
        "    probs = predict_proba([text])[0]\n"
        "    pred_idx = int(np.argmax(probs))\n"
        "    \n"
        "    fig = exp.as_pyplot_figure(label=pred_idx)\n"
        "    plt.title(f'LIME: True={row[\"emotion_label\"]}, Pred={CLASS_NAMES[pred_idx]} ({probs[pred_idx]:.2f})')\n"
        "    plt.show()"
    ))
    
    # --- SECTION 7: SHAP ---
    nb.cells.append(nbf.v4.new_code_cell(
        "## 6. SHAP Explanations\n"
        "# Gunakan tokenizer asli BERT sebagai masker (jauh lebih stabil daripada regex)\n"
        "masker = shap.maskers.Text(tokenizer)\n"
        "explainer_shap = shap.Explainer(predict_proba, masker, output_names=CLASS_NAMES)\n\n"
        "# Gunakan batch_size untuk efisiensi memori\n"
        "shap_values = explainer_shap(samples['review_clean'].tolist(), batch_size=16)\n\n"
        "for i in range(len(samples)):\n"
        "    probs = predict_proba([samples.iloc[i]['review_clean']])[0]\n"
        "    pred_idx = int(np.argmax(probs))\n"
        "    print(f\"\\nSample {i}: True={samples.iloc[i]['emotion_label']}, Pred={CLASS_NAMES[pred_idx]}\")\n"
        "    # Gunakan text plot jika waterfall gagal pada token BERT\n"
        "    shap.plots.waterfall(shap_values[i, :, pred_idx])"
    ))
    
    # --- SECTION 8: Faithfulness Metrics (FULL IMPLEMENTATION) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "## 10. Quantitative Faithfulness Metrics\n"
        "from scipy.stats import spearmanr\n\n"
        "def get_top_k_features(explanation, k=5, mode='lime'):\n"
        "    if mode == 'lime':\n"
        "        # explanation adalah objek LIME\n"
        "        # Ambil label prediksi teratas\n"
        "        label = list(explanation.local_exp.keys())[0]\n"
        "        # Ambil top k fitur positif\n"
        "        features = [x[0] for x in explanation.local_exp[label] if x[1] > 0][:k]\n"
        "        return set(features)\n"
        "    else:\n"
        "        # explanation adalah array shap_values untuk satu sampel\n"
        "        # Ambil top k indeks fitur dengan nilai tertinggi\n"
        "        top_indices = np.argsort(explanation)[-k:]\n"
        "        return set(top_indices)\n\n"
        "metrics_results = []\n"
        "print(\"Calculating Faithfulness Metrics for 15 samples...\")\n\n"
        "for i in range(len(samples)):\n"
        "    text = str(samples.iloc[i]['review_clean'])\n"
        "    probs = predict_proba([text])[0]\n"
        "    pred_idx = int(np.argmax(probs))\n"
        "    \n"
        "    # 1. LIME Top Features\n"
        "    exp_lime = explainer_lime.explain_instance(text, predict_proba, num_features=20, labels=[pred_idx])\n"
        "    lime_feats = dict(exp_lime.as_list(label=pred_idx))\n"
        "    \n"
        "    # 2. SHAP Top Features (mengambil dari hasil yang sudah dihitung sebelumnya)\n"
        "    # shap_values[i, :, pred_idx]\n"
        "    shap_vals_sample = shap_values[i, :, pred_idx].values\n"
        "    feature_names = shap_values[i, :, pred_idx].feature_names\n"
        "    shap_feats = dict(zip(feature_names, shap_vals_sample))\n"
        "    \n"
        "    # Ambil Top 5 kata kunci (yang positif saja)\n"
        "    top_lime = sorted([k for k,v in lime_feats.items() if v > 0], key=lambda x: lime_feats[x], reverse=True)[:5]\n"
        "    top_shap = sorted([k for k,v in shap_feats.items() if v > 0], key=lambda x: shap_feats[x], reverse=True)[:5]\n"
        "    \n"
        "    # 3. Jaccard Similarity\n"
        "    set_lime, set_shap = set(top_lime), set(top_shap)\n"
        "    jaccard = len(set_lime & set_shap) / len(set_lime | set_shap) if len(set_lime | set_shap) > 0 else 0\n"
        "    \n"
        "    metrics_results.append({\n"
        "        'Sample_ID': i,\n"
        "        'True_Label': samples.iloc[i]['emotion_label'],\n"
        "        'Pred_Label': CLASS_NAMES[pred_idx],\n"
        "        'Jaccard_Similarity': jaccard,\n"
        "        'Top_LIME': \", \".join(top_lime),\n"
        "        'Top_SHAP': \", \".join(top_shap)\n"
        "    })\n\n"
        "df_metrics = pd.DataFrame(metrics_results)\n"
        "df_metrics.to_csv(MODEL_DIR / 'faithfulness_metrics.csv', index=False)\n"
        "print(f\"\\nAverage Jaccard Similarity: {df_metrics['Jaccard_Similarity'].mean():.4f}\")\n"
        "display(df_metrics[['Sample_ID', 'True_Label', 'Pred_Label', 'Jaccard_Similarity']])"
    ))

    # --- Save ---
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/04_explainability_prdect.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print('Notebook 04 restored successfully WITH COMPREHENSIVE FIX.')

if __name__ == '__main__':
    restore_notebook()
