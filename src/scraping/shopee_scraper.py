"""
Improved Shopee Review Scraper v2
More aggressive scrolling and review extraction
"""
import json
import time
import hashlib
import re
from datetime import datetime
from playwright.sync_api import sync_playwright

def scrape_shopee_reviews_v2(url: str, max_pages: int = 80) -> dict:
    """Scrape reviews with improved extraction."""
    print(" Shopee Review Scraper v2")
    print(f" Max pages: {max_pages}")
    
    result = {"product": None, "reviews": [], "errors": []}
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={'width': 1400, 'height': 900},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = context.new_page()
        
        try:
            print(" Loading page...")
            page.goto(url, wait_until='domcontentloaded', timeout=120000)
            time.sleep(8)  # Give Shopee more time to render
            
            # Get product title
            product_name = "Apple iPhone 17"
            try:
                page.wait_for_selector('span._44qnta', timeout=10000)
                title = page.query_selector('span._44qnta')
                if title:
                    product_name = title.inner_text()[:100]
            except:
                pass
            
            print(f"   Product: {product_name}")
            result["product"] = {
                "name": product_name,
                "url": url,
                "store_name": "Digimap Official Store",
                "source": "shopee"
            }
            
            # Scroll to reviews - need to go deep (about 15000px)
            print("   Scrolling to reviews section...")
            for i in range(20):
                page.evaluate(f"window.scrollBy(0, 1500)")
                time.sleep(0.3)
            
            time.sleep(2)
            
            # Click on "Dengan Komentar" filter to get reviews with text
            try:
                comment_filter = page.query_selector('div:has-text("Dengan Komentar")')
                if comment_filter:
                    page.evaluate("(el) => el.click()", comment_filter)
                    time.sleep(2)
                    print("   Filtered to comments only")
            except:
                pass
            
            all_reviews = []
            seen_ids = set()
            current_page = 1
            no_new_count = 0
            
            while current_page <= max_pages and len(all_reviews) < 800:
                # Extract reviews using JavaScript
                reviews_data = page.evaluate("""
                () => {
                    const reviews = [];
                    
                    // Method 1: Find all text blocks that look like reviews
                    const allText = document.body.innerText;
                    const lines = allText.split('\\n').filter(l => l.trim().length > 0);
                    
                    let currentReview = null;
                    
                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i].trim();
                        
                        // Username pattern (masked like g*****o)
                        if (line.match(/^[a-z]\\*+[a-z]$/i) && line.length < 15) {
                            if (currentReview && currentReview.text) {
                                reviews.push(currentReview);
                            }
                            currentReview = {username: line, rating: 5, text: '', date: '', variant: ''};
                        }
                        // Date pattern
                        else if (line.match(/^\\d{4}-\\d{2}-\\d{2}/)) {
                            if (currentReview) {
                                const parts = line.split('|');
                                currentReview.date = parts[0].trim();
                                if (parts[1] && parts[1].includes('Variasi:')) {
                                    currentReview.variant = parts[1].replace('Variasi:', '').trim();
                                }
                            }
                        }
                        // Review text (longer content, not a button/label)
                        else if (line.length > 30 && line.length < 2000 && 
                                 !line.includes('Rp') && 
                                 !line.includes('Lihat Semua') &&
                                 !line.includes('Tambah ke') &&
                                 !line.includes('Beli Sekarang')) {
                            if (currentReview && !currentReview.text) {
                                currentReview.text = line;
                            }
                        }
                    }
                    
                    if (currentReview && currentReview.text) {
                        reviews.push(currentReview);
                    }
                    
                    return reviews;
                }
                """)
                
                # Process extracted reviews
                new_count = 0
                for r in reviews_data:
                    if r.get('text') and len(r['text']) >= 20:
                        content = f"shopee|{r.get('username', '')}|{r['text'][:50]}"
                        review_id = hashlib.md5(content.encode()).hexdigest()[:12]
                        
                        if review_id not in seen_ids:
                            seen_ids.add(review_id)
                            all_reviews.append({
                                "review_id": review_id,
                                "product_url": url,
                                "product_name": product_name,
                                "store_name": "Digimap Official Store",
                                "reviewer_name": r.get('username', 'Anonymous'),
                                "rating": r.get('rating', 5),
                                "review_text": r['text'],
                                "review_date": r.get('date', ''),
                                "variant": r.get('variant', ''),
                                "helpful_count": 0,
                                "source": "shopee",
                                "scraped_at": datetime.now().isoformat()
                            })
                            new_count += 1
                
                if new_count > 0:
                    print(f"   Page {current_page}: +{new_count} reviews (total: {len(all_reviews)})")
                    no_new_count = 0
                else:
                    no_new_count += 1
                    if no_new_count >= 3:
                        print(f"   No new reviews for 3 pages, stopping")
                        break
                
                # Click next page button
                try:
                    # Look for the right arrow pagination button
                    next_btn = page.query_selector('button.shopee-icon-button--right')
                    if not next_btn:
                        # Try alternative selectors
                        next_btn = page.query_selector('[class*="next"]')
                    if not next_btn:
                        # Try the > button in pagination
                        next_btn = page.evaluate("""
                        () => {
                            const btns = document.querySelectorAll('button');
                            for (const b of btns) {
                                if (b.querySelector('svg') && b.closest('[class*="pagination"]')) {
                                    const rect = b.getBoundingClientRect();
                                    if (rect.right > window.innerWidth / 2) {
                                        return true;
                                    }
                                }
                            }
                            return false;
                        }
                        """)
                    
                    if next_btn:
                        if isinstance(next_btn, bool):
                            # Click the right-most pagination button
                            page.evaluate("""
                            () => {
                                const btns = Array.from(document.querySelectorAll('[class*="pagination"] button'));
                                if (btns.length > 0) {
                                    btns[btns.length - 1].click();
                                }
                            }
                            """)
                        else:
                            is_disabled = page.evaluate("(el) => el.disabled", next_btn)
                            if is_disabled:
                                print(f"   Reached last page")
                                break
                            page.evaluate("(el) => el.click()", next_btn)
                        
                        time.sleep(2)
                        current_page += 1
                    else:
                        print(f"   No pagination found")
                        break
                        
                except Exception as e:
                    print(f"   Pagination error: {e}")
                    break
            
            result["reviews"] = all_reviews
            print(f"\n Total extracted: {len(all_reviews)} reviews from {current_page} pages")
            
        except Exception as e:
            print(f" Error: {e}")
            result["errors"].append(str(e))
        finally:
            browser.close()
    
    return result


def main():
    url = "https://shopee.co.id/Apple-iPhone-17-i.255563049.28042411268"
    
    print("="*60)
    print("SHOPEE IPHONE 17 REVIEW SCRAPER V2")  
    print("="*60)
    
    result = scrape_shopee_reviews_v2(url, max_pages=100)
    
    # Save results
    if result["reviews"]:
        import pandas as pd
        
        with open('data/raw/shopee_reviews.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        df = pd.DataFrame(result["reviews"])
        df.to_csv('data/raw/shopee_reviews.csv', index=False, encoding='utf-8')
        
        print(f"\n Saved {len(result['reviews'])} reviews to:")
        print(f"   - data/raw/shopee_reviews.json")
        print(f"   - data/raw/shopee_reviews.csv")
    else:
        print("\n No reviews extracted")


if __name__ == "__main__":
    main()
