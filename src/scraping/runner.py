"""
Browser-based Tokopedia Review Scraper with Full Pagination.
Navigates to /review page to get ALL reviews.
"""
import sys
import json
import argparse
import os
import time
import random
import re
import hashlib
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from playwright.sync_api import sync_playwright
from config import settings

# Timeouts
MAX_TIME_PER_URL = 300  # 5 minutes per URL for full pagination
MAX_PAGES = 50          # Max pagination pages


def apply_stealth(page):
    """Apply stealth settings."""
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['id-ID', 'id', 'en-US', 'en'] });
        window.chrome = { runtime: {} };
    """)


def extract_product_info(page, url: str) -> dict:
    """Extract product metadata."""
    product = {
        "url": url,
        "name": None,
        "price": None,
        "store_name": None,
        "rating": None,
        "review_count": None,
        "scraped_at": datetime.now().isoformat()
    }
    
    try:
        # Product Name
        elem = page.query_selector('h1[data-testid="lblPDPDetailProductName"]')
        if elem:
            product["name"] = elem.inner_text().strip()
        
        # Price
        elem = page.query_selector('[data-testid="lblPDPDetailProductPrice"]')
        if elem:
            product["price"] = elem.inner_text().strip()
        
        # Store Name - multiple selectors
        for sel in ['[data-testid="llbPDPFooterShopName"] h2', 'a[data-testid="llbPDPFooterShopName"]', '[data-testid="lblPDPShopName"]']:
            elem = page.query_selector(sel)
            if elem:
                product["store_name"] = elem.inner_text().strip()
                break
        
        # Rating
        elem = page.query_selector('[data-testid="lblPDPDetailProductRatingNumber"]')
        if elem:
            match = re.search(r'(\d+\.?\d*)', elem.inner_text())
            if match:
                product["rating"] = float(match.group(1))
        
        # Review Count
        elem = page.query_selector('[data-testid="lblPDPDetailProductRatingCounter"]')
        if elem:
            text = elem.inner_text().replace('.', '').replace(',', '')
            match = re.search(r'(\d+)', text)
            if match:
                product["review_count"] = int(match.group(1))
        
        print(f"  ✅ {product['name'][:50] if product['name'] else 'Unknown'}...", file=sys.stderr)
        
    except Exception as e:
        print(f"  ⚠️ Product extraction: {e}", file=sys.stderr)
    
    return product


def navigate_to_review_page(page, url: str) -> bool:
    """Navigate to the dedicated /review page for full review list."""
    try:
        # Construct review page URL
        review_url = url.rstrip('/') + '/review'
        print(f"  📖 Navigating to review page...", file=sys.stderr)
        
        page.goto(review_url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)
        
        # Wait for reviews to load
        page.wait_for_selector('article', timeout=10000)
        print(f"  ✅ Review page loaded", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"  ⚠️ Could not load review page: {e}", file=sys.stderr)
        # Fallback: try clicking "Lihat Semua Ulasan" on product page
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(2)
            page.evaluate("window.scrollTo(0, 2000)")
            time.sleep(1)
            
            btn = page.query_selector('button:has-text("Lihat Semua")')
            if btn:
                btn.click()
                time.sleep(2)
                return True
        except:
            pass
        return False


def extract_reviews_from_page(page, product: dict, seen_ids: set, target_ratings: set = None) -> list:
    """Extract reviews from current page view."""
    reviews = []
    
    articles = page.query_selector_all('article')
    
    for article in articles:
        try:
            review = extract_single_review(article, product, target_ratings=target_ratings)
            if review and review.get("review_text"):
                # Skip duplicates
                if review["review_id"] not in seen_ids:
                    seen_ids.add(review["review_id"])
                    reviews.append(review)
        except:
            continue
    
    return reviews


def extract_single_review(article, product: dict, target_ratings: set = None) -> dict:
    """Extract data from a single article."""
    try:
        # Review text - MOST IMPORTANT for sentiment classification
        review_text = ""
        text_elem = article.query_selector('[data-testid="lblItemUlasan"]')
        if text_elem:
            review_text = text_elem.inner_text().strip()
        
        # MUST have text for sentiment classification
        if not review_text or len(review_text) < 3:
            return None
        
        # Skip UI elements only (not reviews mentioning these words)
        skip_patterns = ["lihat semua", "image lainnya", "lihat lebih", "tampilkan lebih"]
        if any(x in review_text.lower() for x in skip_patterns):
            return None
        
        # Reviewer name (masked format like M***y)
        reviewer_name = "Anonymous"
        spans = article.query_selector_all('span, p')
        for span in spans:
            try:
                text = span.inner_text().strip()
                if text and re.match(r'^[A-Za-z]\*{2,}[a-z]$', text):
                    reviewer_name = text
                    break
            except:
                continue
        
        # Rating: never fallback to 5; skip review if rating cannot be extracted.
        rating = None
        try:
            # Most reliable selector on Tokopedia review card.
            stars = article.query_selector_all('[data-testid="icnStarScore"]')
            if stars and 1 <= len(stars) <= 5:
                rating = len(stars)

            # Fallback parse from rating containers / aria labels.
            if rating is None:
                rating_nodes = article.query_selector_all(
                    '[data-testid*="rating"], [aria-label*="bintang" i], [aria-label*="star" i], [class*="rating"]'
                )
                for node in rating_nodes:
                    try:
                        aria = node.get_attribute('aria-label') or ''
                        txt = node.inner_text() or ''
                        combined = f"{aria} {txt}".strip()
                        match = re.search(r'(\d+)', combined)
                        if not match:
                            continue
                        val = int(match.group(1))
                        if 1 <= val <= 5:
                            rating = val
                            break
                    except:
                        continue
        except:
            pass

        if rating is None:
            return None
        
        # Optional rating filter for targeted scraping (e.g., only 1-3 stars)
        if target_ratings and rating not in target_ratings:
            return None

        # Variant
        variant = None
        try:
            var_elem = article.query_selector('[data-testid="lblVarian"]')
            if var_elem:
                variant = var_elem.inner_text().strip()
        except:
            pass
        
        # Date
        review_date = None
        for span in spans:
            try:
                text = span.inner_text().strip()
                if text and ("lalu" in text or "hari" in text or "minggu" in text or "bulan" in text):
                    review_date = text
                    break
            except:
                continue
        
        # Generate unique ID
        id_content = f"{product['url']}|{reviewer_name}|{review_text[:50]}"
        review_id = hashlib.md5(id_content.encode()).hexdigest()[:12]
        
        return {
            "review_id": review_id,
            "product_url": product["url"],
            "product_name": product["name"] or "Unknown",
            "store_name": product["store_name"] or "Unknown",
            "reviewer_name": reviewer_name,
            "rating": rating,
            "review_text": review_text,
            "review_date": review_date,
            "variant": variant,
            "helpful_count": 0,
            "scraped_at": datetime.now().isoformat()
        }
        
    except:
        return None


def scrape_all_reviews_with_pagination(page, product: dict, max_reviews: int = 500, target_ratings: set = None) -> list:
    """Scrape all reviews using pagination with correct selectors."""
    all_reviews = []
    seen_ids = set()
    current_page = 1
    no_new_reviews_count = 0
    
    while current_page <= MAX_PAGES and len(all_reviews) < max_reviews:
        # Scroll up first to see reviews
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        
        # Extract reviews from current page
        reviews = extract_reviews_from_page(page, product, seen_ids, target_ratings=target_ratings)
        new_count = len(reviews)
        all_reviews.extend(reviews)
        
        if new_count > 0:
            print(f"  📄 Page {current_page}: +{new_count} reviews (total: {len(all_reviews)})", file=sys.stderr)
            no_new_reviews_count = 0
        else:
            # If filtering by target ratings, keep paginating because
            # early pages may not contain the desired stars.
            if target_ratings:
                print(
                    f"  📄 Page {current_page}: +0 reviews after rating filter {sorted(target_ratings)}",
                    file=sys.stderr,
                )
            else:
                no_new_reviews_count += 1
                if no_new_reviews_count >= 2:
                    print(f"  📊 No new reviews found for 2 pages, stopping", file=sys.stderr)
                    break
        
        # Scroll to bottom to find pagination
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.5)
        
        # Try to click "next page" button using aria-label
        next_btn = page.query_selector('button[aria-label="Laman berikutnya"]')
        
        if next_btn:
            try:
                # Check if button is disabled
                is_disabled = page.evaluate("(el) => el.disabled", next_btn)
                if is_disabled:
                    print(f"  📊 Reached last page (page {current_page})", file=sys.stderr)
                    break
                
                # Use JavaScript click to avoid scroll issues
                page.evaluate("(el) => el.click()", next_btn)
                time.sleep(1.5)  # Wait for new reviews to load
                current_page += 1
            except Exception as e:
                # Try alternative: scroll and retry
                try:
                    page.evaluate("window.scrollBy(0, -200)")  # Scroll up a bit
                    time.sleep(0.3)
                    page.evaluate("(el) => el.click()", next_btn)
                    time.sleep(1.5)
                    current_page += 1
                except:
                    print(f"  ⚠️ Could not click next: {e}", file=sys.stderr)
                    break
        else:
            # Fallback: try specific page number button
            next_page_num = current_page + 1
            page_btn = page.query_selector(f'button[aria-label="Laman {next_page_num}"]')
            
            if page_btn:
                try:
                    page.evaluate("(el) => el.click()", page_btn)
                    time.sleep(1.5)
                    current_page += 1
                except Exception as e:
                    print(f"  ⚠️ Could not click page {next_page_num}: {e}", file=sys.stderr)
                    break
            else:
                print(f"  📊 No more pagination buttons (stopped at page {current_page})", file=sys.stderr)
                break
    
    print(f"  ✅ Total: {len(all_reviews)} reviews from {current_page} pages", file=sys.stderr)
    return all_reviews


def scrape_url(context, url: str, max_reviews: int = 500, target_ratings: set = None) -> dict:
    """Scrape a single URL with retry logic."""
    start_time = time.time()
    result = {"product": None, "reviews": [], "error": None}
    
    MAX_RETRIES = 3
    
    for attempt in range(MAX_RETRIES):
        try:
            # Create fresh page for each attempt
            page = context.new_page()
            apply_stealth(page)
            
            # Step 1: Load product page and extract info
            print(f"\n🔍 Loading (attempt {attempt + 1}): {url[:50]}...", file=sys.stderr)
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(3)  # Wait for page to stabilize
            
            # Scroll to trigger lazy load
            page.evaluate("window.scrollTo(0, 500)")
            time.sleep(1)
            
            product = extract_product_info(page, url)
            result["product"] = product
            
            # Check if product loaded
            if not product.get("name"):
                print(f"  ⚠️ Product not loaded, retrying...", file=sys.stderr)
                page.close()
                time.sleep(2)
                continue
            
            # Step 2: Navigate to review page  
            if navigate_to_review_page(page, url):
                # Scroll down to load pagination
                page.evaluate("window.scrollTo(0, 2000)")
                time.sleep(1)
                
                # Step 3: Scrape all reviews with pagination
                reviews = scrape_all_reviews_with_pagination(
                    page, product, max_reviews, target_ratings=target_ratings
                )
                result["reviews"] = reviews
            else:
                # Fallback: extract from current page
                seen_ids = set()
                reviews = extract_reviews_from_page(
                    page, product, seen_ids, target_ratings=target_ratings
                )
                result["reviews"] = reviews
            
            elapsed = time.time() - start_time
            print(f"  ✅ Done in {elapsed:.1f}s - {len(result['reviews'])} reviews", file=sys.stderr)
            
            page.close()
            break  # Success, exit retry loop
            
        except Exception as e:
            result["error"] = str(e)
            print(f"  ❌ Attempt {attempt + 1} failed: {e}", file=sys.stderr)
            try:
                page.close()
            except:
                pass
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
            else:
                print(f"  ❌ All attempts failed for {url[:50]}", file=sys.stderr)
    
    return result


def run_scraper(urls: list, max_reviews: int = 500, target_ratings: list = None) -> dict:
    """Main scraper function."""
    results = {"products": [], "reviews": [], "errors": []}
    rating_filter = set(target_ratings) if target_ratings else None
    
    print(f"🚀 Starting browser scraper for {len(urls)} URLs", file=sys.stderr)
    print(f"📊 Max reviews per product: {max_reviews}", file=sys.stderr)
    if rating_filter:
        print(f"🎯 Target ratings filter: {sorted(rating_filter)}", file=sys.stderr)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=['--no-sandbox', '--disable-blink-features=AutomationControlled', '--disable-http2']
        )
        
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="id-ID",
            timezone_id="Asia/Jakarta",
            user_agent=settings.USER_AGENT
        )
        
        total = len(urls)
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*50}", file=sys.stderr)
            print(f"📌 [{i}/{total}] Processing", file=sys.stderr)
            print(f"{'='*50}", file=sys.stderr)
            
            # Pass context - scrape_url creates fresh page per attempt
            result = scrape_url(context, url, max_reviews, target_ratings=rating_filter)
            
            if result["product"]:
                results["products"].append(result["product"])
            results["reviews"].extend(result["reviews"])
            
            if result["error"]:
                results["errors"].append({"url": url, "error": result["error"]})
            
            if i < total:
                delay = random.uniform(2, 4)
                print(f"  ⏳ Waiting {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)
        
        browser.close()
    
    print(f"\n✅ Done! {len(results['products'])} products, {len(results['reviews'])} reviews", file=sys.stderr)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokopedia Review Scraper")
    parser.add_argument("--urls", type=str, required=True, help="JSON array of URLs")
    parser.add_argument("--max-reviews", type=int, default=500, help="Max reviews per product")
    parser.add_argument(
        "--target-ratings",
        type=str,
        default="",
        help="Comma-separated ratings filter, e.g. '1,2,3'",
    )
    
    args = parser.parse_args()
    
    urls = json.loads(args.urls)
    target_ratings = []
    if args.target_ratings.strip():
        target_ratings = [int(x.strip()) for x in args.target_ratings.split(",") if x.strip()]

    results = run_scraper(urls, max_reviews=args.max_reviews, target_ratings=target_ratings)
    
    print(json.dumps(results, ensure_ascii=False))
