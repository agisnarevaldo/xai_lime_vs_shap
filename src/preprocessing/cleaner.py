"""
Text & structural cleaning pipeline for Tokopedia iPhone 17 reviews.
Used by notebooks/02_cleaning.ipynb
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Structural Cleaning
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows by review_id."""
    before = len(df)
    df = df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)
    print(f"  [dedupe]    {before:>5} → {len(df):>5}  (removed {before - len(df)})")
    return df


def drop_empty_reviews(df: pd.DataFrame, min_chars: int = 1) -> pd.DataFrame:
    """Drop rows where review_text is null, empty, or shorter than min_chars."""
    before = len(df)
    df = df[df["review_text"].notna()]
    df = df[df["review_text"].str.strip().str.len() >= min_chars]
    df = df.reset_index(drop=True)
    print(f"  [empty/short] {before:>5} → {len(df):>5}  (removed {before - len(df)}, min_chars={min_chars})")
    return df


def filter_valid_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with rating in [1, 2, 3, 4, 5]."""
    before = len(df)
    df = df[df["rating"].notna()]
    df = df[df["rating"].astype(float).between(1, 5)]
    df = df.reset_index(drop=True)
    print(f"  [rating]    {before:>5} → {len(df):>5}  (removed {before - len(df)})")
    return df


def clean_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Strip 'Varian: ' prefix from variant column."""
    df["variant"] = df["variant"].apply(
        lambda x: x.replace("Varian: ", "").strip() if isinstance(x, str) else None
    )
    return df


def add_sentiment_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map numeric rating → sentiment_label.
      4-5  → Positif
      3    → Netral
      1-2  → Negatif
    """
    def _map(r):
        r = int(float(r))
        if r >= 4:
            return "Positif"
        elif r == 3:
            return "Netral"
        else:
            return "Negatif"

    df["sentiment_label"] = df["rating"].apply(_map)
    dist = df["sentiment_label"].value_counts()
    print(f"  [sentiment] Positif={dist.get('Positif', 0)}  "
          f"Netral={dist.get('Netral', 0)}  "
          f"Negatif={dist.get('Negatif', 0)}")
    return df


# ---------------------------------------------------------------------------
# 2. Text Preprocessing
# ---------------------------------------------------------------------------

_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_NON_ALPHA  = re.compile(r"[^a-z0-9\s]")
_WS_RE      = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Pipeline:
      lowercase → strip URL/mention/hashtag →
      remove non-alphanumeric → normalise whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def apply_text_cleaning(df: pd.DataFrame, min_words: int = 1) -> pd.DataFrame:
    """
    Add `review_text_raw` (original) and `review_text_clean` (processed).
    Adds `text_length` (word count) and drops rows below min_words.
    """
    df = df.copy()
    df["review_text_raw"] = df["review_text"]
    df["review_text_clean"] = df["review_text"].apply(clean_text)
    df["text_length"] = df["review_text_clean"].str.split().str.len()

    before = len(df)
    df = df[df["text_length"] >= min_words].reset_index(drop=True)
    print(f"  [text_len]  {before:>5} → {len(df):>5}  (removed {before - len(df)}, min_words={min_words})")
    return df


# ---------------------------------------------------------------------------
# 3. Optional: Sastrawi Stemming
# ---------------------------------------------------------------------------

def apply_stemming(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply PySastrawi stemmer to `review_text_clean` → `review_text_stemmed`.
    Only use when PySastrawi is installed. Safe to skip for LIME/SHAP since
    human-readable tokens are preferred there.
    """
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df["review_text_stemmed"] = df["review_text_clean"].apply(stemmer.stem)
        print("  [stemming]  Done (PySastrawi)")
    except ImportError:
        print("  [stemming]  Skipped — PySastrawi not installed. Run: pip install PySastrawi")
    return df


# ---------------------------------------------------------------------------
# 4. Final Cleanup
# ---------------------------------------------------------------------------

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed for analysis."""
    drop_cols = [c for c in ["helpful_count", "scraped_at", "product_url"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"  [drop_cols] Removed: {drop_cols}")
    return df


# ---------------------------------------------------------------------------
# 5. Master Pipeline
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(input_path: str, output_path: str,
                           min_chars: int = 1, min_words: int = 1,
                           apply_stem: bool = False) -> pd.DataFrame:
    """
    Full cleaning pipeline end-to-end.
    Returns the cleaned DataFrame and saves it to output_path.
    """
    print("=" * 50)
    print("CLEANING PIPELINE")
    print("=" * 50)

    df = pd.read_csv(input_path)
    print(f"\nLoaded : {len(df)} rows  |  {df.shape[1]} columns")
    print("-" * 50)

    df = remove_duplicates(df)
    df = drop_empty_reviews(df, min_chars=min_chars)
    df = filter_valid_ratings(df)
    df = clean_variant(df)
    df = add_sentiment_label(df)
    df = apply_text_cleaning(df, min_words=min_words)
    if apply_stem:
        df = apply_stemming(df)
    df = drop_unused_columns(df)

    print("-" * 50)
    print(f"Final   : {len(df)} rows  |  {df.shape[1]} columns")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved → {output_path}")
    print("=" * 50)

    return df
