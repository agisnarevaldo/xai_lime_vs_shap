"""
GraphQL-based Tokopedia Review Scraper.
Uses direct API calls to gql.tokopedia.com for complete review extraction.

Key Endpoint: https://gql.tokopedia.com/graphql/productReviewList
"""
import sys
import json
import argparse
import os
import requests
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


# GraphQL Configuration
GRAPHQL_ENDPOINT = "https://gql.tokopedia.com/graphql/productReviewList"
REVIEWS_PER_PAGE = 15
MAX_PAGES = 100  # Safety limit


def get_product_ids_from_page(page, url: str) -> dict:
    """Extract productID and shopID from product page."""
    result = {
        "url": url,
        "productID": None,
        "shopID": None,
        "productName": None,
        "shopName": None,
        "totalReviews": None
    }
    
    try:
        print(f"  📦 Loading: {url[:60]}...", file=sys.stderr)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)
        
        # Scroll to trigger data loading
        page.evaluate("window.scrollTo(0, 500)")
        time.sleep(1)
        
        # Extract IDs from page source
        html = page.content()
        
        # Product ID patterns
        pid_patterns = [
            r'"productID":\s*"(\d+)"',
            r'"product_id":\s*(\d+)',
            r'"id":\s*"(\d+)".*?"shop"',
        ]
        for pattern in pid_patterns:
            match = re.search(pattern, html)
            if match:
                result["productID"] = match.group(1)
                break
        
        # Shop ID patterns
        sid_patterns = [
            r'"shopID":\s*"(\d+)"',
            r'"shop_id":\s*(\d+)',
        ]
        for pattern in sid_patterns:
            match = re.search(pattern, html)
            if match:
                result["shopID"] = match.group(1)
                break
        
        # Product name
        name_elem = page.query_selector('h1[data-testid="lblPDPDetailProductName"]')
        if name_elem:
            result["productName"] = name_elem.inner_text().strip()
        
        # Shop name
        shop_elem = page.query_selector('[data-testid="llbPDPFooterShopName"] h2')
        if shop_elem:
            result["shopName"] = shop_elem.inner_text().strip()
        
        # Total reviews
        review_elem = page.query_selector('[data-testid="lblPDPDetailProductRatingCounter"]')
        if review_elem:
            text = review_elem.inner_text().replace('.', '').replace(',', '')
            match = re.search(r'(\d+)', text)
            if match:
                result["totalReviews"] = int(match.group(1))
        
        print(f"  ✅ Product: {result['productName'][:40] if result['productName'] else 'Unknown'}...", file=sys.stderr)
        print(f"  📊 ProductID: {result['productID']}, ShopID: {result['shopID']}, Reviews: {result['totalReviews']}", file=sys.stderr)
        
    except Exception as e:
        print(f"  ❌ Error extracting IDs: {e}", file=sys.stderr)
    
    return result


def fetch_reviews_graphql(product_id: str, shop_id: str, page_num: int = 1) -> dict:
    """Fetch reviews using GraphQL API."""
    
    # GraphQL query for product reviews
    query = """
    query productReviewList($productID: String!, $page: Int!, $limit: Int!, $sortBy: String, $filterBy: String) {
      productrevGetProductReviewList(productID: $productID, page: $page, limit: $limit, sortBy: $sortBy, filterBy: $filterBy) {
        productID
        list {
          feedbackID
          message
          productRating
          reviewCreateTime
          reviewCreateTimestamp
          isAnonymous
          isReportable
          variantName
          user {
            userID
            fullName
            image
          }
          imageAttachments {
            attachmentID
            imageThumbnailUrl
            imageUrl
          }
          videoAttachments {
            attachmentID
            videoUrl
          }
          stats {
            productID
            totalHelpful
          }
        }
        hasNext
        totalReviews
      }
    }
    """
    
    variables = {
        "productID": product_id,
        "page": page_num,
        "limit": REVIEWS_PER_PAGE,
        "sortBy": "create_time",
        "filterBy": ""
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": settings.USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://www.tokopedia.com",
        "Referer": "https://www.tokopedia.com/",
        "X-Source": "tokopedia-lite",
        "X-Tkpd-Lite-Service": "zeus",
        "X-Device": "desktop",
    }
    
    payload = {
        "operationName": "productReviewList",
        "variables": variables,
        "query": query
    }
    
    try:
        print(f"  📡 API call page {page_num}, productID={product_id}...", file=sys.stderr)
        response = requests.post(
            GRAPHQL_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"  📡 Response status: {response.status_code}", file=sys.stderr)
        
        if response.status_code != 200:
            print(f"  ⚠️ HTTP Error: {response.status_code}", file=sys.stderr)
            print(f"  Response: {response.text[:500]}", file=sys.stderr)
            return {"data": {"productrevGetProductReviewList": {"list": [], "hasNext": False, "totalReviews": 0}}}
        
        data = response.json()
        
        # Debug: show response structure
        if page_num == 1:
            if "errors" in data:
                print(f"  ⚠️ GraphQL Errors: {data['errors']}", file=sys.stderr)
            if "data" in data and data["data"]:
                review_data = data.get("data", {}).get("productrevGetProductReviewList", {})
                if review_data:
                    print(f"  ✅ API returned {len(review_data.get('list', []))} reviews, hasNext={review_data.get('hasNext')}", file=sys.stderr)
                else:
                    print(f"  ⚠️ No productrevGetProductReviewList in response", file=sys.stderr)
                    print(f"  Response keys: {list(data.get('data', {}).keys())}", file=sys.stderr)
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request error: {e}", file=sys.stderr)
        return {"data": {"productrevGetProductReviewList": {"list": [], "hasNext": False, "totalReviews": 0}}}
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON decode error: {e}", file=sys.stderr)
        return {"data": {"productrevGetProductReviewList": {"list": [], "hasNext": False, "totalReviews": 0}}}
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}", file=sys.stderr)
        return {"data": {"productrevGetProductReviewList": {"list": [], "hasNext": False, "totalReviews": 0}}}


def scrape_all_reviews_graphql(product_info: dict) -> list:
    """Scrape all reviews for a product using GraphQL API."""
    reviews = []
    product_id = product_info.get("productID")
    shop_id = product_info.get("shopID")
    
    if not product_id:
        print(f"  ⚠️ No productID found, skipping GraphQL", file=sys.stderr)
        return reviews
    
    page_num = 1
    has_next = True
    
    print(f"  🔄 Fetching reviews via GraphQL API...", file=sys.stderr)
    
    while has_next and page_num <= MAX_PAGES:
        result = fetch_reviews_graphql(product_id, shop_id, page_num)
        
        if not result or "data" not in result:
            print(f"  ⚠️ No data returned for page {page_num}", file=sys.stderr)
            break
        
        review_data = result.get("data", {}).get("productrevGetProductReviewList", {})
        review_list = review_data.get("list", [])
        has_next = review_data.get("hasNext", False)
        total_reviews = review_data.get("totalReviews", 0)
        
        if page_num == 1:
            print(f"  📊 Total reviews available: {total_reviews}", file=sys.stderr)
        
        for item in review_list:
            review = parse_graphql_review(item, product_info)
            if review:
                reviews.append(review)
        
        if len(review_list) > 0:
            print(f"  📄 Page {page_num}: {len(review_list)} reviews (total: {len(reviews)})", file=sys.stderr)
        
        page_num += 1
        
        # Rate limiting
        time.sleep(random.uniform(0.5, 1.0))
    
    print(f"  ✅ Extracted {len(reviews)} reviews via GraphQL", file=sys.stderr)
    return reviews


def parse_graphql_review(item: dict, product_info: dict) -> dict:
    """Parse a single review from GraphQL response."""
    try:
        message = item.get("message", "")
        if not message or len(message) < 3:
            return None
        
        user_info = item.get("user", {})
        reviewer_name = user_info.get("fullName", "Anonymous")
        if item.get("isAnonymous"):
            reviewer_name = "Anonymous"
        
        # Generate unique ID
        content = f"{product_info['url']}|{reviewer_name}|{message[:50]}"
        review_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        return {
            "review_id": review_id,
            "feedback_id": item.get("feedbackID"),
            "product_url": product_info.get("url", ""),
            "product_name": product_info.get("productName", "Unknown"),
            "store_name": product_info.get("shopName", "Unknown"),
            "reviewer_name": reviewer_name,
            "rating": item.get("productRating", 5),
            "review_text": message,
            "review_date": item.get("reviewCreateTime"),
            "variant": item.get("variantName"),
            "helpful_count": item.get("stats", {}).get("totalHelpful", 0) if item.get("stats") else 0,
            "has_images": len(item.get("imageAttachments", [])) > 0,
            "has_video": len(item.get("videoAttachments", [])) > 0,
            "scraped_at": datetime.now().isoformat()
        }
    except Exception as e:
        return None


def run_graphql_scraper(urls: list, max_reviews_per_product: int = 500) -> dict:
    """Main scraper using GraphQL API."""
    results = {
        "products": [],
        "reviews": [],
        "errors": []
    }
    
    print(f"🚀 Starting GraphQL scraper for {len(urls)} URLs", file=sys.stderr)
    print(f"📊 Max reviews per product: {max_reviews_per_product}", file=sys.stderr)
    
    with sync_playwright() as p:
        print(f"🌐 Launching browser (visible mode)...", file=sys.stderr)
        
        browser = p.chromium.launch(
            headless=False,  # Non-headless to avoid blocking
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-http2',  # Force HTTP/1.1
            ]
        )
        
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="id-ID",
            timezone_id="Asia/Jakarta",
            user_agent=settings.USER_AGENT
        )
        
        page = context.new_page()
        
        # Stealth
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)
        
        total = len(urls)
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*50}", file=sys.stderr)
            print(f"📌 [{i}/{total}] Processing", file=sys.stderr)
            print(f"{'='*50}", file=sys.stderr)
            
            try:
                # Step 1: Extract product IDs from page (with retry)
                product_info = None
                for attempt in range(3):
                    try:
                        product_info = get_product_ids_from_page(page, url)
                        if product_info and product_info.get("productID"):
                            break
                    except Exception as e:
                        print(f"  ⚠️ Attempt {attempt + 1} failed: {e}", file=sys.stderr)
                        time.sleep(2)
                
                if product_info and product_info.get("productID"):
                    results["products"].append({
                        "url": url,
                        "name": product_info.get("productName"),
                        "store_name": product_info.get("shopName"),
                        "product_id": product_info.get("productID"),
                        "shop_id": product_info.get("shopID"),
                        "total_reviews": product_info.get("totalReviews"),
                        "scraped_at": datetime.now().isoformat()
                    })
                    
                    # Step 2: Fetch all reviews via GraphQL
                    reviews = scrape_all_reviews_graphql(product_info)
                    
                    # Limit per product
                    reviews = reviews[:max_reviews_per_product]
                    results["reviews"].extend(reviews)
                else:
                    results["errors"].append({
                        "url": url,
                        "error": "Could not extract productID"
                    })
                
            except Exception as e:
                results["errors"].append({
                    "url": url,
                    "error": str(e)
                })
                print(f"  ❌ Failed: {e}", file=sys.stderr)
            
            # Delay between URLs
            if i < total:
                delay = random.uniform(2, 4)
                print(f"  ⏳ Waiting {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)
        
        browser.close()
    
    print(f"\n✅ Done! {len(results['products'])} products, {len(results['reviews'])} reviews", file=sys.stderr)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokopedia GraphQL Review Scraper")
    parser.add_argument("--urls", type=str, required=True, help="JSON array of URLs")
    parser.add_argument("--max-reviews", type=int, default=500, help="Max reviews per product")
    
    args = parser.parse_args()
    
    urls = json.loads(args.urls)
    results = run_graphql_scraper(urls, max_reviews_per_product=args.max_reviews)
    
    # Output JSON to stdout
    print(json.dumps(results, ensure_ascii=False))

