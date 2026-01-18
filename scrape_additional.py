"""
Scrape additional URLs to reach 2000+ reviews
"""
import sys
import json

sys.path.insert(0, '.')
from src.scraping.runner import run_scraper

# New URLs from browser search
new_urls = [
    "https://www.tokopedia.com/putra-siregar-store/spesial-putra-siregar-iphone-17-pro-new-launching-new-produk-garansi-resmi-indonesia-1733343861743978301",
    "https://www.tokopedia.com/tradem/apple-iphone-17-pro-original-1733512360637203461",
    "https://www.tokopedia.com/kungfu-gadget/kg-apple-iphone-17-pro-series-garansi-resmi-ibox-gdn-1733195417491047609",
    "https://www.tokopedia.com/royalltech/apple-iphone-17-pro-1tb-all-collors-resmi-apple-indonesia-segel-greenpeel-1733546747226129911",
    "https://www.tokopedia.com/agresponsel/iphone-17-pro-256gb-kamera-48mp-baterai-3998-mah-garansi-resmi-indonesia-1733361538893842199",
    "https://www.tokopedia.com/ljsofficial/iphone-17-pro-256-gb-512-gb-1tb-garansi-resmi-indonesia-1732870514079991198",
    "https://www.tokopedia.com/topsellofficial/iphon-17-pro-256gb-512gb-1tb-new-original-garansi-resmi-indonesia-1-tahun-1732814659663922545",
    "https://www.tokopedia.com/dmzstoreofficial/iphone-17-pro-256-gb-cosmic-orange-like-new-garansi-resmi-1733490936703780825",
    "https://www.tokopedia.com/4gcell45/new-iphone-17-pro-256gb-garansi-resmi-ibox-1733489510024644021",
    "https://www.tokopedia.com/r-shop-cahaya/iphone-17-pro-256gb-512gb-1tb-garansi-resmi-indonesia-ibox-paa-ida-saa-new-segel-original-garansi-1-tahun-1733155221240776623",
    "https://www.tokopedia.com/ayrivshop/apple-iphone-17-pro-1tb-512gb-256gb-a19-pro-chip-silver-cosmic-orange-deep-blue-resmi-256gb-deep-blue-1733589700784915631",
    "https://www.tokopedia.com/nins-gadget/iphone-17-pro-256gb-paa-bnib-masih-segel-greenpeel-kualitas-terbaik-1733719989327529076",
    "https://www.tokopedia.com/jullindo-jis/apple-new-iphone-17-pro-17-pro-max-chip-a19-pro-256gb-512gb-1tb-2tb-garansi-resmi-1733388533953430818",
    "https://www.tokopedia.com/imartmedan/iphone-17-pro-256gb-garansi-resmi-lengkap-bonus-warna-silver-orange-blue-1733246861609830165",
    "https://www.tokopedia.com/mj18phone/iph-17-pro-256gb-512gb-1tb-garansi-resmi-indonesia-bnib-1733149607951107401",
]

print(f"Scraping {len(new_urls)} new URLs...")
print("="*60)

result = run_scraper(new_urls, max_reviews=300)

print("\n" + "="*60)
print("ADDITIONAL SCRAPING RESULTS")
print("="*60)
print(f"Products: {len(result['products'])}")
print(f"Reviews: {len(result['reviews'])}")

# Save to new file
import pandas as pd

with open('data/raw/tokopedia_reviews_additional.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

if result['reviews']:
    df = pd.DataFrame(result['reviews'])
    df.to_csv('data/raw/tokopedia_reviews_additional.csv', index=False, encoding='utf-8')
    print(f"\nSaved to:")
    print(f"  - data/raw/tokopedia_reviews_additional.json")
    print(f"  - data/raw/tokopedia_reviews_additional.csv")
