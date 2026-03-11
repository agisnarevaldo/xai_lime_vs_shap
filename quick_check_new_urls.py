"""
Pass 1 — Quick rating distribution check for NEW URLs.

Scrapes up to 30 reviews per URL (no rating filter, stops early after 2 empty
pages) so the whole run is fast (~1-2 min per URL).

Outputs:
  data/raw/quick_check_new_urls.csv   — raw sample reviews
  data/raw/quick_check_summary.csv    — per-URL rating counts + "promising" flag
  promising_urls.txt                  — URLs with any non-5-star review seen

Run BEFORE scrape_new_urls_deep.py to identify which stores are worth a
full targeted scrape.
"""
import sys
import time
from pathlib import Path
from typing import Iterator
import pandas as pd

sys.path.insert(0, ".")
from src.scraping.runner import run_scraper


# ── helpers ──────────────────────────────────────────────────────────────────

NEW_SECTION_MARKER = "# ===== NEW URLs - Added 2026-03-12"


def chunked(items: list, size: int) -> Iterator[list]:
    for index in range(0, len(items), size):
        yield items[index:index + size]

def load_new_urls(path: str = "link.txt") -> list:
    """Return only URLs from the 2026-03-12 discovery section."""
    text = Path(path).read_text(encoding="utf-8")
    idx = text.find(NEW_SECTION_MARKER)
    if idx == -1:
        print("⚠️  Could not find new-URL section marker — loading ALL URLs instead")
        section = text
    else:
        section = text[idx:]

    urls = [
        line.strip()
        for line in section.splitlines()
        if line.strip().startswith("http")
    ]
    return list(dict.fromkeys(urls))   # deduplicate, preserve order


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    urls = load_new_urls("link.txt")
    if not urls:
        print("No new URLs found"); return

    print(f"Quick-checking {len(urls)} new URLs…")
    print("Each URL: max 30 reviews, no rating filter (fast mode)\n")

    batch_size = 8
    cooldown_seconds = 15
    all_reviews = []
    all_errors = []

    for batch_index, batch_urls in enumerate(chunked(urls, batch_size), start=1):
        print(f"[batch {batch_index}] scraping {len(batch_urls)} URLs")
        results = run_scraper(batch_urls, max_reviews=30, target_ratings=None)
        all_reviews.extend(results.get("reviews", []))
        all_errors.extend(results.get("errors", []))

        if batch_index * batch_size < len(urls):
            print(f"[batch {batch_index}] cooldown {cooldown_seconds}s\n")
            time.sleep(cooldown_seconds)

    reviews = all_reviews
    if not reviews:
        print("No reviews collected — check scraper output above"); return

    df = pd.DataFrame(reviews)

    # Save raw samples
    raw_out = Path("data/raw/quick_check_new_urls.csv")
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_out, index=False, encoding="utf-8-sig")
    print(f"\nSaved raw samples → {raw_out}  ({len(df)} rows)")

    # Per-URL summary
    summary_rows = []
    for url, grp in df.groupby("product_url"):
        counts = grp["rating"].value_counts().to_dict()
        total  = len(grp)
        non5   = sum(v for k, v in counts.items() if k != 5)
        summary_rows.append({
            "url":         url,
            "total":       total,
            "rating_1":    counts.get(1, 0),
            "rating_2":    counts.get(2, 0),
            "rating_3":    counts.get(3, 0),
            "rating_4":    counts.get(4, 0),
            "rating_5":    counts.get(5, 0),
            "non_5_total": non5,
            "promising":   non5 > 0,
        })

    summary = pd.DataFrame(summary_rows).sort_values("non_5_total", ascending=False)
    sum_out = Path("data/raw/quick_check_summary.csv")
    summary.to_csv(sum_out, index=False, encoding="utf-8-sig")
    print(f"Saved summary      → {sum_out}")

    if all_errors:
        err_out = Path("data/raw/quick_check_errors.csv")
        pd.DataFrame(all_errors).to_csv(err_out, index=False, encoding="utf-8-sig")
        print(f"Saved errors       → {err_out}  ({len(all_errors)} rows)")

    # Promising list
    promising = summary[summary["promising"]]["url"].tolist()
    prom_out = Path("promising_urls.txt")
    prom_out.write_text("\n".join(promising), encoding="utf-8")
    print(f"Saved promising    → {prom_out}  ({len(promising)} URLs)\n")

    print("=== TOP 20 by non-5★ count ===")
    print(summary.head(20).to_string(index=False))

    print(f"\n{'='*50}")
    print(f"Total new URLs scanned : {len(urls)}")
    print(f"URLs with ≥1 non-5★    : {len(promising)}")
    print(f"\nNext step → run: python scrape_new_urls_deep.py")


if __name__ == "__main__":
    main()
