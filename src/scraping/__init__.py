# Scraping module init
from .tokopedia import TokopediaScraper
from .models import Product, Review, ScrapedData
from .data_manager import DataManager

__all__ = ['TokopediaScraper', 'Product', 'Review', 'ScrapedData', 'DataManager']
