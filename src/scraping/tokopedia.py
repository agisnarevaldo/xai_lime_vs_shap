"""
Tokopedia Review Scraper using Playwright Async API for Jupyter compatibility.
"""
import asyncio
import time
import random
import re
from typing import List, Optional
from datetime import datetime

from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Add config path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

from .models import Product, Review, ScrapedData


class TokopediaScraper:
    """Async scraper for Tokopedia product pages and reviews."""
    
    def __init__(self, headless: bool = None):
        self.headless = headless if headless is not None else settings.HEADLESS
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._playwright = None
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def start(self):
        """Start browser instance."""
        self._playwright = await async_playwright().start()
        print(f"🚀 Launching browser (headless={self.headless})...")
        
        self.browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="id-ID",
            timezone_id="Asia/Jakarta",
            user_agent=settings.USER_AGENT
        )
        
        self.page = await self.context.new_page()
        await self._apply_stealth()
    
    async def close(self):
        """Close browser instance."""
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
        print("🔒 Browser closed")
    
    async def _apply_stealth(self):
        """Apply stealth settings to avoid bot detection."""
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['id-ID', 'id', 'en-US', 'en'] });
            window.chrome = { runtime: {} };
            
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
    
    async def _random_delay(self, min_s: float = None, max_s: float = None):
        """Random delay to mimic human behavior."""
        min_delay = min_s or settings.MIN_DELAY
        max_delay = max_s or settings.MAX_DELAY
        delay = random.uniform(min_delay, max_delay)
        await asyncio.sleep(delay)
    
    async def _wait_for_page_ready(self, timeout: int = None):
        """Wait for page to be ready with improved strategy."""
        timeout = timeout or settings.PAGE_TIMEOUT
        try:
            # Wait for DOM content first (faster)
            await self.page.wait_for_load_state("domcontentloaded", timeout=timeout)
            
            # Then wait for specific Tokopedia elements
            selectors = [
                'h1[data-testid="lblPDPDetailProductName"]',
                'div[data-testid="pdpLayoutReview"]',
                'h1'
            ]
            
            for selector in selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=10000)
                    break
                except:
                    continue
            
            # Small additional wait for dynamic content
            await asyncio.sleep(1.5)
            
        except Exception as e:
            print(f"  ⚠️ Page load warning: {e}")
    
    async def _scroll_page(self, scroll_count: int = 3):
        """Scroll page to trigger lazy loading."""
        for i in range(scroll_count):
            await self.page.evaluate(f"window.scrollTo(0, {(i + 1) * 500})")
            await asyncio.sleep(0.5)
        await self.page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.3)
    
    async def scrape_product_info(self, url: str) -> Product:
        """Scrape product metadata from URL."""
        product = Product(url=url)
        
        try:
            print(f"  📦 Extracting product info...")
            await self._scroll_page(2)
            
            # Product Name
            for sel in ['h1[data-testid="lblPDPDetailProductName"]', 'h1']:
                elem = await self.page.query_selector(sel)
                if elem:
                    product.name = (await elem.inner_text()).strip()
                    break
            
            # Price
            elem = await self.page.query_selector('[data-testid="lblPDPDetailProductPrice"]')
            if elem:
                product.price = (await elem.inner_text()).strip()
            
            # Store Name
            for sel in ['[data-testid="llbPDPFooterShopName"] h2', 'a[data-testid="llbPDPFooterShopName"]']:
                elem = await self.page.query_selector(sel)
                if elem:
                    product.store_name = (await elem.inner_text()).strip()
                    break
            
            # Rating
            elem = await self.page.query_selector('[data-testid="lblPDPDetailProductRatingNumber"]')
            if elem:
                text = await elem.inner_text()
                match = re.search(r'(\d+\.?\d*)', text)
                if match:
                    product.rating = float(match.group(1))
            
            # Review Count
            elem = await self.page.query_selector('[data-testid="lblPDPDetailProductRatingCounter"]')
            if elem:
                text = (await elem.inner_text()).replace('.', '').replace(',', '')
                match = re.search(r'(\d+)', text)
                if match:
                    product.review_count = int(match.group(1))
            
            # Stock Status
            stock_elem = await self.page.query_selector('text="Stok Habis"')
            product.stock_status = "Habis" if stock_elem else "Tersedia"
            
            print(f"  ✅ {product.name[:50] if product.name else 'Unknown'}...")
            
        except Exception as e:
            print(f"  ⚠️ Product extraction error: {e}")
        
        return product
    
    async def _navigate_to_reviews(self) -> bool:
        """Navigate to review section of the page."""
        try:
            # Try to find and click review tab/section
            review_selectors = [
                '[data-testid="pdpLayoutReview"]',
                'text="Ulasan"',
                'button:has-text("Ulasan")',
                '[data-testid="btnPDPFooterReview"]'
            ]
            
            for sel in review_selectors:
                try:
                    elem = await self.page.query_selector(sel)
                    if elem:
                        await elem.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)
                        return True
                except:
                    continue
            
            # Fallback: scroll to bottom where reviews usually are
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
            await asyncio.sleep(1)
            return True
            
        except Exception as e:
            print(f"  ⚠️ Could not navigate to reviews: {e}")
            return False
    
    async def _scroll_to_load_reviews(self, max_reviews: int = None) -> int:
        """Scroll to load more reviews (lazy loading)."""
        max_reviews = max_reviews or settings.MAX_REVIEWS_PER_PRODUCT
        loaded_count = 0
        scroll_attempts = 0
        last_count = 0
        
        while scroll_attempts < settings.MAX_SCROLL_ATTEMPTS:
            # Count current reviews
            reviews = await self.page.query_selector_all('[data-testid*="review"], .review-item, [class*="ReviewItem"]')
            current_count = len(reviews)
            
            if current_count >= max_reviews:
                print(f"  📊 Reached max reviews limit ({max_reviews})")
                break
            
            if current_count == last_count:
                scroll_attempts += 1
                if scroll_attempts >= 5:
                    # Try clicking "Load More" button
                    try:
                        load_more = await self.page.query_selector('button:has-text("Lihat"), button:has-text("Muat")')
                        if load_more:
                            await load_more.click()
                            await asyncio.sleep(1)
                            scroll_attempts = 0
                            continue
                    except:
                        pass
            else:
                scroll_attempts = 0
            
            last_count = current_count
            loaded_count = current_count
            
            # Scroll down
            await self.page.evaluate("window.scrollBy(0, 500)")
            await asyncio.sleep(settings.REVIEW_SCROLL_DELAY)
        
        print(f"  📊 Loaded {loaded_count} review elements")
        return loaded_count
    
    async def _extract_reviews(self, product: Product) -> List[Review]:
        """Extract all visible reviews from the page."""
        reviews = []
        
        # Common review container selectors for Tokopedia
        container_selectors = [
            '[data-testid*="review"]',
            '.review-item',
            '[class*="ReviewItem"]',
            '[class*="ulasan"]'
        ]
        
        # Try each selector
        review_elements = []
        for sel in container_selectors:
            elements = await self.page.query_selector_all(sel)
            if elements:
                review_elements = elements
                break
        
        print(f"  🔍 Found {len(review_elements)} review elements to extract")
        
        for elem in review_elements:
            try:
                review = await self._extract_single_review(elem, product)
                if review and review.review_text:
                    reviews.append(review)
            except Exception as e:
                continue  # Skip problematic reviews
        
        return reviews
    
    async def _extract_single_review(self, elem, product: Product) -> Optional[Review]:
        """Extract data from a single review element."""
        try:
            # Reviewer name
            reviewer_name = "Anonymous"
            name_selectors = ['[data-testid*="name"]', '.user-name', '[class*="username"]', 'span:first-child']
            for sel in name_selectors:
                name_elem = await elem.query_selector(sel)
                if name_elem:
                    text = (await name_elem.inner_text()).strip()
                    if text and len(text) < 50:  # Reasonable name length
                        reviewer_name = text
                        break
            
            # Rating (count stars or find rating number)
            rating = 5  # Default
            rating_elem = await elem.query_selector('[data-testid*="rating"], [class*="rating"], [class*="star"]')
            if rating_elem:
                # Try to find rating number
                rating_text = (await rating_elem.get_attribute('aria-label')) or (await rating_elem.inner_text())
                match = re.search(r'(\d)', rating_text or '')
                if match:
                    rating = int(match.group(1))
                else:
                    # Count filled stars
                    stars = await elem.query_selector_all('[class*="star-filled"], [class*="starFilled"]')
                    if stars:
                        rating = len(stars)
            
            # Review text
            review_text = ""
            text_selectors = [
                '[data-testid*="content"]',
                '[data-testid*="text"]',
                '.review-text',
                '[class*="review-content"]',
                'p',
                'span:not([class*="name"]):not([class*="date"])'
            ]
            for sel in text_selectors:
                text_elem = await elem.query_selector(sel)
                if text_elem:
                    text = (await text_elem.inner_text()).strip()
                    if len(text) > 10:  # Minimum length for actual review
                        review_text = text
                        break
            
            if not review_text:
                return None
            
            # Date
            review_date = None
            date_selectors = ['[data-testid*="date"]', '[class*="date"]', 'time']
            for sel in date_selectors:
                date_elem = await elem.query_selector(sel)
                if date_elem:
                    review_date = (await date_elem.inner_text()).strip()
                    break
            
            # Variant
            variant = None
            variant_selectors = ['[data-testid*="variant"]', '[class*="variant"]']
            for sel in variant_selectors:
                var_elem = await elem.query_selector(sel)
                if var_elem:
                    variant = (await var_elem.inner_text()).strip()
                    break
            
            return Review(
                product_url=product.url,
                product_name=product.name or "Unknown Product",
                store_name=product.store_name or "Unknown Store",
                reviewer_name=reviewer_name,
                rating=rating,
                review_text=review_text,
                review_date=review_date,
                variant=variant
            )
            
        except Exception as e:
            return None
    
    async def scrape_url(self, url: str, max_reviews: int = None) -> tuple:
        """Scrape a single URL for product info and reviews."""
        try:
            print(f"\n🔍 Navigating to: {url[:60]}...")
            await self.page.goto(url, wait_until="domcontentloaded", timeout=settings.PAGE_TIMEOUT)
            await self._wait_for_page_ready()
            await self._random_delay(2, 4)
            
            # Scrape product info
            product = await self.scrape_product_info(url)
            
            # Navigate to and scrape reviews
            reviews = []
            if await self._navigate_to_reviews():
                await self._scroll_to_load_reviews(max_reviews)
                reviews = await self._extract_reviews(product)
                print(f"  ✅ Extracted {len(reviews)} reviews")
            
            return product, reviews
            
        except Exception as e:
            print(f"  ❌ Failed to scrape: {e}")
            return Product(url=url), []
    
    async def scrape_urls(self, urls: List[str], max_reviews_per_product: int = None) -> ScrapedData:
        """Scrape multiple URLs."""
        data = ScrapedData()
        total = len(urls)
        
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*60}")
            print(f"📌 [{i}/{total}] Processing URL")
            print(f"{'='*60}")
            
            product, reviews = await self.scrape_url(url, max_reviews_per_product)
            data.add_product(product)
            data.add_reviews(reviews)
            
            # Rate limiting between URLs
            if i < total:
                await self._random_delay()
        
        print(f"\n✅ Scraping complete! {len(data.products)} products, {len(data.reviews)} reviews")
        return data


def load_urls_from_file(filepath: str = None) -> List[str]:
    """Load URLs from text file."""
    path = filepath or settings.LINKS_FILE
    urls = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('http'):
                urls.append(line)
    
    print(f"📋 Loaded {len(urls)} URLs from {path}")
    return urls


# Helper function for running in Jupyter
async def scrape_single_url(url: str, headless: bool = False, max_reviews: int = 20):
    """Helper function to scrape a single URL (for Jupyter)."""
    async with TokopediaScraper(headless=headless) as scraper:
        return await scraper.scrape_url(url, max_reviews)


async def scrape_multiple_urls(urls: List[str], headless: bool = False, max_reviews: int = 100):
    """Helper function to scrape multiple URLs (for Jupyter)."""
    async with TokopediaScraper(headless=headless) as scraper:
        return await scraper.scrape_urls(urls, max_reviews)
