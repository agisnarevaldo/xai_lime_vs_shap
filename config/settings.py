# Configuration settings for Tokopedia scraper

# Timeout settings (milliseconds)
PAGE_TIMEOUT = 90000  # 90 seconds for page load
ELEMENT_TIMEOUT = 30000  # 30 seconds for element wait
NAVIGATION_TIMEOUT = 60000

# Browser settings
HEADLESS = False  # Set to True for production
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Rate limiting (seconds)
MIN_DELAY = 3
MAX_DELAY = 8
REVIEW_SCROLL_DELAY = 1.5

# Review scraping settings
MAX_REVIEWS_PER_PRODUCT = 500  # Limit per product
MAX_SCROLL_ATTEMPTS = 50  # Stop scrolling after this many attempts

# File paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LINKS_FILE = os.path.join(BASE_DIR, "link.txt")

# Output settings
OUTPUT_CSV = os.path.join(DATA_RAW_DIR, "tokopedia_reviews.csv")
OUTPUT_JSON = os.path.join(DATA_RAW_DIR, "tokopedia_reviews.json")
