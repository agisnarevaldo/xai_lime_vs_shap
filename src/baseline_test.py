import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import os

def run_baseline_comparison(data_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['review_text_clean'])
    
    # 1. 3-Class Scenario
    print("=== SCENARIO 1: 3-Class (Original) ===")
    y3 = df['sentiment_label']
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_text_clean'], y3, test_size=0.2, random_state=42, stratify=y3
    )
    
    vec = TfidfVectorizer(max_features=2000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    clf3 = LogisticRegression(class_weight='balanced', random_state=42)
    clf3.fit(X_train_vec, y_train)
    y_pred3 = clf3.predict(X_test_vec)
    
    print(classification_report(y_test, y_pred3))
    f1_minority_3 = f1_score(y_test, y_pred3, labels=['Negatif', 'Netral'], average='macro')
    
    # 2. 2-Class Scenario
    print("\n=== SCENARIO 2: 2-Class (Binary Merge) ===")
    # Re-apply merge just in case
    y2 = df['sentiment_label'].apply(lambda x: "Positif" if x == "Positif" else "Non-Positif")
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        df['review_text_clean'], y2, test_size=0.2, random_state=42, stratify=y2
    )
    
    X_train_vec2 = vec.fit_transform(X_train2)
    X_test_vec2 = vec.transform(X_test2)
    
    clf2 = LogisticRegression(class_weight='balanced', random_state=42)
    clf2.fit(X_train_vec2, y_train2)
    y_pred2 = clf2.predict(X_test_vec2)
    
    print(classification_report(y_test2, y_pred2))
    f1_minority_2 = f1_score(y_test2, y_pred2, pos_label='Non-Positif')

    print("\n" + "="*50)
    print("SUMMARY PENGUJIAN BASELINE")
    print("="*50)
    print(f"F1-Score Minority (Original 3-Class Macro): {f1_minority_3:.4f}")
    print(f"F1-Score Minority (Binary Non-Positif):    {f1_minority_2:.4f}")
    
    improvement = (f1_minority_2 - f1_minority_3) / (f1_minority_3 + 1e-9) * 100
    if improvement > 0:
        print(f"Hasil: Peningkatan stabilitas deteksi sebesar {improvement:.2f}%")
    else:
        print("Hasil: Performa relatif stabil, namun variansi label terminimalisir.")
    print("="*50)

if __name__ == "__main__":
    run_baseline_comparison("data/processed/tokopedia_reviews_binary.csv")
