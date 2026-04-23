import pandas as pd
import random
import re
import os

# --- EDA Functions ---

def get_synonyms(word):
    # A simple Indonesian synonym dictionary for common sentiment words
    synonyms_dict = {
        "bagus": ["mantap", "oke", "keren", "baik", "elok"],
        "jelek": ["buruk", "kurang", "payah", "hancur"],
        "cepat": ["kencang", "kilat", "gesit", "segera"],
        "lama": ["lambat", "lelet", "pelan", "molor"],
        "puas": ["senang", "bahagia", "lega"],
        "kecewa": ["kesal", "sedih", "nyesel", "dongkol"],
        "ramah": ["baik", "sopan", "friendly"],
        "mahal": ["tinggi", "berlebihan"],
        "murah": ["terjangkau", "ekonomis"],
        "original": ["asli", "ori", "authentic"],
        "palsu": ["kw", "tiruan", "fake"],
    }
    return synonyms_dict.get(word.lower(), [])

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if get_synonyms(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def random_deletion(words, p):
    if len(words) <= 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        return [random.choice(words)]
    return new_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter > 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
    words = sentence.split()
    if len(words) == 0:
        return [sentence] * num_aug
    num_words = len(words)
    augmented_sentences = []
    
    # Calculate how many sentences per technique
    n_per_tech = (num_aug // 4) + 1
    
    # SR
    n_sr = max(1, int(alpha_sr*num_words))
    for _ in range(n_per_tech):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))
        
    # RI
    n_ri = max(1, int(alpha_ri*num_words))
    for _ in range(n_per_tech):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))
        
    # RS
    n_rs = max(1, int(alpha_rs*num_words))
    for _ in range(n_per_tech):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))
        
    # RD
    for _ in range(n_per_tech):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))
        
    random.shuffle(augmented_sentences)
    if len(augmented_sentences) > num_aug:
        augmented_sentences = augmented_sentences[:num_aug]
        
    return augmented_sentences

# --- Execution ---

def main():
    INPUT_PATH = "data/processed/tokopedia_reviews_clean.csv"
    OUTPUT_PATH = "data/processed/tokopedia_reviews_balanced_eda.csv"
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    print("Initial distribution:")
    print(df['sentiment_label'].value_counts())
    
    target_count = 500
    
    # 1. Undersample Positif
    df_positif = df[df['sentiment_label'] == 'Positif'].sample(n=target_count, random_state=42)
    
    # 2. Augment Netral
    df_netral = df[df['sentiment_label'] == 'Netral']
    netral_list = df_netral.to_dict('records')
    new_netral = []
    
    num_to_augment_netral = target_count - len(df_netral)
    aug_per_sample_netral = (num_to_augment_netral // len(df_netral)) + 1
    
    for row in netral_list:
        augs = eda(row['review_text_clean'], num_aug=aug_per_sample_netral)
        for a in augs:
            new_row = row.copy()
            new_row['review_text_clean'] = a
            new_row['review_text_raw'] = f"[EDA] {a}"
            new_netral.append(new_row)
            if (len(new_netral) + len(df_netral)) >= target_count:
                break
        if (len(new_netral) + len(df_netral)) >= target_count:
            break
            
    df_netral_aug = pd.concat([df_netral, pd.DataFrame(new_netral)]).head(target_count)
    
    # 3. Augment Negatif
    df_negatif = df[df['sentiment_label'] == 'Negatif']
    negatif_list = df_negatif.to_dict('records')
    new_negatif = []
    
    num_to_augment_negatif = target_count - len(df_negatif)
    aug_per_sample_negatif = (num_to_augment_negatif // len(df_negatif)) + 1
    
    for row in negatif_list:
        augs = eda(row['review_text_clean'], num_aug=aug_per_sample_negatif)
        for a in augs:
            new_row = row.copy()
            new_row['review_text_clean'] = a
            new_row['review_text_raw'] = f"[EDA] {a}"
            new_negatif.append(new_row)
            if (len(new_negatif) + len(df_negatif)) >= target_count:
                break
        if (len(new_negatif) + len(df_negatif)) >= target_count:
            break
            
    df_negatif_aug = pd.concat([df_negatif, pd.DataFrame(new_negatif)]).head(target_count)
    
    # Combine
    df_final = pd.concat([df_positif, df_netral_aug, df_negatif_aug]).sample(frac=1, random_state=42)
    
    df_final.to_csv(OUTPUT_PATH, index=False)
    print("\nBalanced distribution (EDA):")
    print(df_final['sentiment_label'].value_counts())
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
