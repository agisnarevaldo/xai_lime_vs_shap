import sys
import json
import time
import random
import re
from datetime import datetime
from playwright.sync_api import sync_playwright

def apply_stealth(page):
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['id-ID', 'id', 'en-US', 'en'] });
        window.chrome = { runtime: {} };
    """)

def extract_product_data(page, url: str) -> dict:
    data = {
        "product_name": None, "price": None, "store_name": None,
        "store_location": None, "rating": None, "review_count": None,
        "sold_count": None, "stock_status": None, "url": url,
        "scraped_at": datetime.now().isoformat()
    }
    try:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        time.sleep(1)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        
        # Product Name
        for sel in ['h1[data-testid="lblPDPDetailProductName"]', 'h1']:
            elem = page.query_selector(sel)
            if elem:
                data["product_name"] = elem.inner_text()
                break
        
        # Price
        elem = page.query_selector('[data-testid="lblPDPDetailProductPrice"]')
        if elem:
            data["price"] = elem.inner_text().strip()
        
        # Store Name
        for sel in ['[data-testid="llbPDPFooterShopName"] h2', 'a[data-testid="llbPDPFooterShopName"]']:
            elem = page.query_selector(sel)
            if elem:
                data["store_name"] = elem.inner_text()
                break
        
        # Rating
        elem = page.query_selector('[data-testid="lblPDPDetailProductRatingNumber"]')
        if elem:
            match = re.search(r'(\d+\.?\d*)', elem.inner_text())
            if match:
                data["rating"] = float(match.group(1))
        
        # Review Count
        elem = page.query_selector('[data-testid="lblPDPDetailProductRatingCounter"]')
        if elem:
            text = elem.inner_text().replace('.', '').replace(',', '')
            match = re.search(r'(\d+)', text)
            if match:
                data["review_count"] = int(match.group(1))
        
        # Stock Status
        stock = page.query_selector('text="Stok Habis"')
        data["stock_status"] = "Habis" if stock else "Tersedia"
        
    except Exception as e:
        print(f"  ⚠️ Extract error: {e}", file=sys.stderr)
    
    return data

def scrape_tokopedia(urls: list, headless: bool = False) -> list:
    results = []
    
    try:
        with sync_playwright() as p:
            print(f"🚀 Launching browser (headless={headless})...", file=sys.stderr)
            browser = p.chromium.launch(
                headless=headless,
                args=['--no-sandbox', '--disable-blink-features=AutomationControlled']
            )
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                locale="id-ID",
                timezone_id="Asia/Jakarta",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            apply_stealth(page)
            
            for i, url in enumerate(urls, 1):
                print(f"\n🔍 [{i}/{len(urls)}] {url[:60]}...", file=sys.stderr)
                try:
                    page.goto(url, wait_until="networkidle", timeout=60000)
                    time.sleep(random.uniform(2, 4))
                    
                    data = extract_product_data(page, url)
                    results.append(data)
                    
                    name = data['product_name'][:40] if data['product_name'] else 'N/A'
                    print(f"  ✅ {name}", file=sys.stderr)
                    print(f"     💰 {data['price']} | ⭐ {data['rating']} | 📝 {data['review_count']}", file=sys.stderr)
                    
                    if i < len(urls):
                        delay = random.uniform(5, 10)
                        print(f"  ⏳ Waiting {delay:.1f}s...", file=sys.stderr)
                        time.sleep(delay)
                        
                except Exception as e:
                    print(f"  ❌ Failed: {e}", file=sys.stderr)
                    results.append({
                        "url": url, "error": str(e),
                        "product_name": None, "price": None, "store_name": None,
                        "rating": None, "review_count": None, "stock_status": None,
                        "scraped_at": datetime.now().isoformat()
                    })
            
            browser.close()
            print(f"\n✅ Scraping complete! {len(results)} products.", file=sys.stderr)
            
    except Exception as e:
        print(f"❌ Browser error: {e}", file=sys.stderr)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scraper.py '[urls_json]' [headless]", file=sys.stderr)
        sys.exit(1)
    
    urls = json.loads(sys.argv[1])
    headless = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
    
    results = scrape_tokopedia(urls, headless)
    
    # Output JSON to stdout (this is what the notebook reads)
    print(json.dumps(results))