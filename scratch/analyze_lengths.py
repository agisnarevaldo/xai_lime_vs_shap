import pandas as pd
import json
import sys
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
df_train = pd.read_csv("data/processed/prdect_train.csv")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Compute token lengths
lengths = df_train['review_clean'].astype(str).apply(lambda x: len(tokenizer.encode(x)))

print("=== Sequence Length Statistics ===")
print("Min length    :", lengths.min())
print("Mean length   :", lengths.mean())
print("Median length :", lengths.median())
print("90th percentile:", lengths.quantile(0.90))
print("95th percentile:", lengths.quantile(0.95))
print("99th percentile:", lengths.quantile(0.99))
print("Max length    :", lengths.max())
