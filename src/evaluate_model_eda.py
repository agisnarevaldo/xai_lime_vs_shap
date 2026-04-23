import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Config ---
MODEL_PATH = "outputs/indobert_finetuned"
DATA_PATH = "data/processed/tokopedia_reviews_balanced_eda.csv"
FIG_DIR = "outputs/figures"
MAX_LEN = 128
BATCH_SIZE = 16

os.makedirs(FIG_DIR, exist_ok=True)

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer(
            review, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, 
            return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, real_values

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_PATH)
    LABEL_MAP = {"Negatif": 0, "Netral": 1, "Positif": 2}
    ID2LABEL = {0: "Negatif", 1: "Netral", 2: "Positif"}
    df['label'] = df['sentiment_label'].map(LABEL_MAP)
    
    _, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model = model.to(device)
    
    test_data_loader = DataLoader(
        ReviewDataset(df_test.review_text_clean.to_numpy(), df_test.label.to_numpy(), tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    y_pred, y_test = get_predictions(model, test_data_loader, device)
    
    # Classification Report
    print("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=list(LABEL_MAP.keys()))
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(LABEL_MAP.keys()), 
                yticklabels=list(LABEL_MAP.keys()))
    plt.title('Gambar 4.6 Confusion Matrix - IndoBERT Sentiment Analysis')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(FIG_DIR, "fig4_6_confusion_matrix.png"))
    print(f"Evaluation complete. Plots saved to {FIG_DIR}")

if __name__ == "__main__":
    main()
