import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- Config ---
MODEL_NAME = "indobenchmark/indobert-base-p1"
DATA_PATH = "data/processed/tokopedia_reviews_balanced_eda.csv"
OUTPUT_DIR = "outputs/indobert_finetuned"
FIG_DIR = "outputs/figures"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# --- Dataset Class ---
class ReviewDataset(Dataset):
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
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# --- Training Loop ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for i, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if (i + 1) % 5 == 0:
            print(f"  Batch {i + 1}/{len(data_loader)} loss: {np.mean(losses[-5:]):.4f}")

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# --- Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(DATA_PATH)
    
    # Map labels to IDs
    LABEL_MAP = {"Negatif": 0, "Netral": 1, "Positif": 2}
    df['label'] = df['sentiment_label'].map(LABEL_MAP)
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    train_data_loader = DataLoader(
        ReviewDataset(df_train.review_text_clean.to_numpy(), df_train.label.to_numpy(), tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True
    )
    
    test_data_loader = DataLoader(
        ReviewDataset(df_test.review_text_clean.to_numpy(), df_test.label.to_numpy(), tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    best_accuracy = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, test_data_loader, loss_fn, device, len(df_test))
        print(f"Val   loss {val_loss:.4f} accuracy {val_acc:.4f}")

        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            best_accuracy = val_acc

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.title('Gambar 4.5 Grafik Training dan Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "fig4_5_training_loss.png"))
    print(f"Training complete. Best Val Acc: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
