"""Targeted scraping for low ratings (1-3) to improve class balance quickly."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")
from src.scraping.runner import run_scraper


def load_urls(path: str = "link.txt") -> list:
    p = Path(path)
    if not p.exists():
        return []
    urls = [u.strip() for u in p.read_text(encoding="utf-8").splitlines() if u.strip().startswith("http")]
    return list(dict.fromkeys(urls))


def main():
    urls = load_urls("link.txt")
    if not urls:
        print("No URLs found in link.txt")
        return

    print(f"Loaded {len(urls)} unique URLs")
    print("Running targeted scrape for ratings [1,2,3]...")

    results = run_scraper(urls, max_reviews=300, target_ratings=[1, 2, 3])
    df = pd.DataFrame(results["reviews"])

    out_path = Path("data/raw/tokopedia_reviews_low_ratings.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Rows : {len(df)}")
    if len(df):
        print("Rating distribution:")
        print(df["rating"].value_counts().sort_index())


if __name__ == "__main__":
    main()
