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
    print(f"  [dedupe]    {before:>5} -> {len(df):>5}  (removed {before - len(df)})")
    return df


def drop_empty_reviews(df: pd.DataFrame, min_chars: int = 1) -> pd.DataFrame:
    """Drop rows where review_text is null, empty, or shorter than min_chars."""
    before = len(df)
    df = df[df["review_text"].notna()]
    df = df[df["review_text"].str.strip().str.len() >= min_chars]
    df = df.reset_index(drop=True)
    print(f"  [empty/short] {before:>5} -> {len(df):>5}  (removed {before - len(df)}, min_chars={min_chars})")
    return df


def filter_valid_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with rating in [1, 2, 3, 4, 5]."""
    before = len(df)
    df = df[df["rating"].notna()]
    df = df[df["rating"].astype(float).between(1, 5)]
    df = df.reset_index(drop=True)
    print(f"  [rating]    {before:>5} -> {len(df):>5}  (removed {before - len(df)})")
    return df


def clean_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Strip 'Varian: ' prefix from variant column."""
    df["variant"] = df["variant"].apply(
        lambda x: x.replace("Varian: ", "").strip() if isinstance(x, str) else None
    )
    return df


def add_sentiment_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map numeric rating -> sentiment_label.
      4-5  -> Positif
      3    -> Netral
      1-2  -> Negatif
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
      lowercase -> strip URL/mention/hashtag ->
      remove non-alphanumeric -> normalise whitespace
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
    print(f"  [text_len]  {before:>5} -> {len(df):>5}  (removed {before - len(df)}, min_words={min_words})")
    return df


# ---------------------------------------------------------------------------
# 3. Optional: Sastrawi Stemming
# ---------------------------------------------------------------------------

def apply_stemming(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply PySastrawi stemmer to `review_text_clean` -> `review_text_stemmed`.
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
# 4. Class-Imbalance Helpers
# ---------------------------------------------------------------------------

def undersample_majority(
    df: pd.DataFrame,
    majority_label: str = "Positif",
    n: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Keep ALL minority-class rows; randomly sample n rows from the majority class.
    Useful when Positif >> Negatif+Netral.
    """
    df_maj = df[df["sentiment_label"] == majority_label]
    df_min = df[df["sentiment_label"] != majority_label]

    n_actual = min(n, len(df_maj))
    df_maj_sampled = df_maj.sample(n=n_actual, random_state=random_state)

    result = (
        pd.concat([df_maj_sampled, df_min], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    print(
        f"  [undersample] {majority_label}: {len(df_maj)} -> {n_actual}  "
        f"| minority kept: {len(df_min)}  "
        f"| total: {len(result)}"
    )
    return result


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `binary_label` column pooling Negatif + Netral -> 'Non-Positif'.
    Original `sentiment_label` (3-class) is preserved.
    """
    df = df.copy()
    df["binary_label"] = df["sentiment_label"].apply(
        lambda x: "Positif" if x == "Positif" else "Non-Positif"
    )
    dist = df["binary_label"].value_counts()
    print(
        f"  [binary_label] Positif={dist.get('Positif', 0)}  "
        f"Non-Positif={dist.get('Non-Positif', 0)}"
    )
    return df


# ---------------------------------------------------------------------------
# 5. Final Cleanup
# ---------------------------------------------------------------------------

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed for analysis."""
    drop_cols = [c for c in ["helpful_count", "scraped_at", "product_url"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"  [drop_cols] Removed: {drop_cols}")
    return df


# ---------------------------------------------------------------------------
# 6. Master Pipelines
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
    print(f"Saved -> {output_path}")
    print("=" * 50)

    return df


def run_balanced_pipeline(
    clean_csv: str,
    output_path: str,
    majority_label: str = "Positif",
    n_majority: int = 200,
    random_state: int = 42,
    add_binary: bool = True,
) -> pd.DataFrame:
    """
    Post-cleaning balancing pipeline.
    Reads an already-cleaned CSV, undersamples the majority class, and
    optionally adds a binary_label column.

    Saves result to output_path.
    """
    print("=" * 50)
    print("BALANCING PIPELINE")
    print("=" * 50)

    df = pd.read_csv(clean_csv)
    print(f"Loaded  : {len(df)} rows")
    print("Before:")
    print(df["sentiment_label"].value_counts().to_string())
    print("-" * 50)

    df = undersample_majority(df, majority_label=majority_label, n=n_majority, random_state=random_state)

    if add_binary:
        df = add_binary_label(df)

    df.to_csv(output_path, index=False, encoding="utf-8")
    print("-" * 50)
    print(f"Saved → {output_path}  ({len(df)} rows)")
    print("=" * 50)
    return df
