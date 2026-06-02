import csv
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

raw_file = "data/raw/PRDECT-ID Dataset.csv"

ratings_emotions = defaultdict(Counter)
sentiments_emotions = defaultdict(Counter)
emotion_counts = Counter()

with open(raw_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rating = row.get('Customer Rating')
        emotion = row.get('Emotion')
        sentiment = row.get('Sentiment')
        
        if rating and emotion:
            ratings_emotions[rating][emotion] += 1
        if sentiment and emotion:
            sentiments_emotions[sentiment][emotion] += 1
        if emotion:
            emotion_counts[emotion] += 1

print("Total raw rows:", sum(emotion_counts.values()))
print("\nEmotion distribution in raw data:")
for emo, count in emotion_counts.most_common():
    print(f"  {emo}: {count}")

print("\nCross-tab: Customer Rating vs Emotion:")
for rating in sorted(ratings_emotions.keys()):
    row_str = f"Rating {rating}: "
    for emo in sorted(emotion_counts.keys()):
        row_str += f"{emo}={ratings_emotions[rating][emo]} | "
    print(row_str)

print("\nCross-tab: Sentiment vs Emotion:")
for sent in sorted(sentiments_emotions.keys()):
    row_str = f"Sentiment {sent}: "
    for emo in sorted(emotion_counts.keys()):
        row_str += f"{emo}={sentiments_emotions[sent][emo]} | "
    print(row_str)
