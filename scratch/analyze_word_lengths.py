import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/processed/prdect_clean.csv"

word_counts = []
char_lengths = []

try:
    with open(raw_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('review_clean', '')
            words = text.split()
            word_counts.append(len(words))
            char_lengths.append(len(text))
except FileNotFoundError:
    print("Processed dataset not found. Checking raw dataset...")
    raw_file = "data/raw/PRDECT-ID Dataset.csv"
    with open(raw_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('Customer Review', '')
            words = text.split()
            word_counts.append(len(words))
            char_lengths.append(len(text))

word_counts.sort()
char_lengths.sort()

n = len(word_counts)
print("=== Word Count Statistics ===")
print("Total rows:", n)
print("Min word count:", word_counts[0])
print("Mean word count:", sum(word_counts) / n)
print("Median word count:", word_counts[n // 2])
print("90th percentile:", word_counts[int(n * 0.90)])
print("95th percentile:", word_counts[int(n * 0.95)])
print("99th percentile:", word_counts[int(n * 0.99)])
print("Max word count:", word_counts[-1])

print("\n=== Character Count Statistics ===")
print("Min char count:", char_lengths[0])
print("Mean char count:", sum(char_lengths) / n)
print("Median char count:", char_lengths[n // 2])
print("90th percentile:", char_lengths[int(n * 0.90)])
print("95th percentile:", char_lengths[int(n * 0.95)])
print("99th percentile:", char_lengths[int(n * 0.99)])
print("Max char count:", char_lengths[-1])
