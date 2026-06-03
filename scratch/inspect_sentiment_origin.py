import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Read raw dataset
df = pd.read_csv("data/raw/PRDECT-ID Dataset.csv")

print("=== RELATIONSHIP BETWEEN CUSTOMER RATING AND SENTIMENT ===")
crosstab = pd.crosstab(df['Customer Rating'], df['Sentiment'])
print(crosstab)
print("\nPercentages:")
crosstab_pct = pd.crosstab(df['Customer Rating'], df['Sentiment'], normalize='index') * 100
print(crosstab_pct.round(2))

print("\n=== SAMPLE REVIEWS FOR RATINGS WITH MIXED SENTIMENTS ===")
# Show examples of rating 3 or other ratings with mixed sentiments if any
mixed_ratings = []
for rating in sorted(df['Customer Rating'].unique()):
    counts = df[df['Customer Rating'] == rating]['Sentiment'].value_counts()
    if len(counts) > 1:
        mixed_ratings.append(rating)

if mixed_ratings:
    for rating in mixed_ratings:
        print(f"\nSamples for Rating {rating}:")
        pos_samples = df[(df['Customer Rating'] == rating) & (df['Sentiment'] == 'Positive')][['Customer Review', 'Sentiment']].head(3)
        neg_samples = df[(df['Customer Rating'] == rating) & (df['Sentiment'] == 'Negative')][['Customer Review', 'Sentiment']].head(3)
        print("--- Positive Labels ---")
        for i, row in pos_samples.iterrows():
            print(f"  - {repr(row['Customer Review'])}")
        print("--- Negative Labels ---")
        for i, row in neg_samples.iterrows():
            print(f"  - {repr(row['Customer Review'])}")
else:
    print("There are no ratings with mixed sentiment labels. The mapping is 100% deterministic based on rating.")
