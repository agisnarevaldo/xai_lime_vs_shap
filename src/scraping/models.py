"""
Data models for Tokopedia scraping.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List
import hashlib


@dataclass
class Review:
    """Individual product review."""
    product_url: str
    product_name: str
    store_name: str
    reviewer_name: str
    rating: int
    review_text: str
    review_date: Optional[str] = None
    variant: Optional[str] = None
    helpful_count: int = 0
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    review_id: str = field(default="")
    
    def __post_init__(self):
        """Generate unique review_id based on content hash."""
        if not self.review_id:
            content = f"{self.product_url}|{self.reviewer_name}|{self.review_text[:100]}"
            self.review_id = hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Product:
    """Product metadata."""
    url: str
    name: Optional[str] = None
    price: Optional[str] = None
    store_name: Optional[str] = None
    store_location: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    sold_count: Optional[int] = None
    stock_status: str = "Unknown"
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ScrapedData:
    """Container for all scraped data."""
    products: List[Product] = field(default_factory=list)
    reviews: List[Review] = field(default_factory=list)
    
    def add_product(self, product: Product):
        self.products.append(product)
    
    def add_review(self, review: Review):
        self.reviews.append(review)
    
    def add_reviews(self, reviews: List[Review]):
        self.reviews.extend(reviews)
    
    def get_reviews_dict(self) -> List[dict]:
        return [r.to_dict() for r in self.reviews]
    
    def get_products_dict(self) -> List[dict]:
        return [p.to_dict() for p in self.products]
