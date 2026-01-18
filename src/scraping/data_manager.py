"""
Data Manager for handling review data storage, deduplication, and export.
"""
import json
import csv
import os
from typing import List, Dict, Optional, Set
from datetime import datetime
from .models import Review, ScrapedData

# Add config path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings


class DataManager:
    """Handles data storage, deduplication, and export."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or settings.DATA_RAW_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self._seen_ids: Set[str] = set()
    
    def deduplicate_reviews(self, reviews: List[Review]) -> List[Review]:
        """Remove duplicate reviews based on review_id."""
        unique_reviews = []
        for review in reviews:
            if review.review_id not in self._seen_ids:
                self._seen_ids.add(review.review_id)
                unique_reviews.append(review)
        return unique_reviews
    
    def load_existing_ids(self, json_path: Optional[str] = None) -> Set[str]:
        """Load existing review IDs from JSON file to avoid duplicates."""
        path = json_path or settings.OUTPUT_JSON
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for review in data.get('reviews', []):
                        self._seen_ids.add(review.get('review_id', ''))
                print(f"📂 Loaded {len(self._seen_ids)} existing review IDs")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Could not load existing data: {e}")
        return self._seen_ids
    
    def save_to_json(self, data: ScrapedData, filename: Optional[str] = None) -> str:
        """Save scraped data to JSON file."""
        path = filename or settings.OUTPUT_JSON
        
        output = {
            'metadata': {
                'scraped_at': datetime.now().isoformat(),
                'total_products': len(data.products),
                'total_reviews': len(data.reviews)
            },
            'products': data.get_products_dict(),
            'reviews': data.get_reviews_dict()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved {len(data.reviews)} reviews to {path}")
        return path
    
    def save_to_csv(self, data: ScrapedData, filename: Optional[str] = None) -> str:
        """Save reviews to CSV file."""
        path = filename or settings.OUTPUT_CSV
        reviews = data.get_reviews_dict()
        
        if not reviews:
            print("⚠️ No reviews to save")
            return path
        
        fieldnames = [
            'review_id', 'product_name', 'product_url', 'store_name',
            'reviewer_name', 'rating', 'review_text', 'review_date',
            'variant', 'helpful_count', 'scraped_at'
        ]
        
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(reviews)
        
        print(f"💾 Saved {len(reviews)} reviews to {path}")
        return path
    
    def save_all(self, data: ScrapedData) -> Dict[str, str]:
        """Save to both CSV and JSON."""
        return {
            'json': self.save_to_json(data),
            'csv': self.save_to_csv(data)
        }
    
    def append_reviews(self, new_reviews: List[Review], deduplicate: bool = True) -> int:
        """Append new reviews to existing data files."""
        # Load existing
        self.load_existing_ids()
        
        # Deduplicate
        if deduplicate:
            new_reviews = self.deduplicate_reviews(new_reviews)
        
        if not new_reviews:
            print("ℹ️ No new unique reviews to add")
            return 0
        
        # Load existing JSON
        existing_data = {'products': [], 'reviews': []}
        if os.path.exists(settings.OUTPUT_JSON):
            try:
                with open(settings.OUTPUT_JSON, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                pass
        
        # Append new reviews
        existing_data['reviews'].extend([r.to_dict() for r in new_reviews])
        existing_data['metadata'] = {
            'last_updated': datetime.now().isoformat(),
            'total_reviews': len(existing_data['reviews'])
        }
        
        # Save back
        with open(settings.OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        # Also update CSV
        data = ScrapedData()
        for r in existing_data['reviews']:
            data.reviews.append(Review(**{k: v for k, v in r.items() if k in Review.__dataclass_fields__}))
        self.save_to_csv(data)
        
        print(f"✅ Added {len(new_reviews)} new reviews")
        return len(new_reviews)
