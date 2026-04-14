"""
SDGZero business directory scraper.
Uses the GeoDirectory REST API: /wp-json/geodir/v2/businesses
"""

import time
import requests
from scraper.models import Business

BASE_URL = "https://sdgzero.com/wp-json/geodir/v2/businesses"
DEFAULT_DELAY = 1.0  # seconds between requests (be polite)


_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SDGZero-Scraper/1.0)"}


def fetch_page(page: int, per_page: int = 100) -> list[dict]:
    """Fetch one page of businesses from the API."""
    params = {"page": page, "per_page": per_page}
    response = requests.get(BASE_URL, params=params, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def scrape_all(per_page: int = 100, delay: float = DEFAULT_DELAY) -> list[Business]:
    """
    Paginate through all businesses.
    Stops when a page returns fewer results than per_page.
    """
    businesses = []
    page = 1

    print(f"Starting scrape from {BASE_URL}")

    while True:
        print(f"  Fetching page {page}...", end=" ", flush=True)
        raw_items = fetch_page(page=page, per_page=per_page)

        if not raw_items:
            print("empty — done.")
            break

        parsed = []
        for item in raw_items:
            try:
                parsed.append(Business.model_validate(item))
            except Exception as e:
                print(f"\n  [WARN] Failed to parse item id={item.get('id')}: {e}")

        businesses.extend(parsed)
        print(f"got {len(parsed)} businesses (total: {len(businesses)})")

        # Stop if we got fewer than per_page — last page
        if len(raw_items) < per_page:
            break

        page += 1
        time.sleep(delay)

    return businesses
