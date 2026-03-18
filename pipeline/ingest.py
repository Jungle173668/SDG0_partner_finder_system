"""
Ingest pipeline: scrape → parse → embed → store.
Run this once to populate the ChromaDB database.

Usage:
    python -m pipeline.ingest
"""

import sys
import os

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.spider import scrape_all
from db.chroma_store import BusinessStore


def run():
    print("=" * 50)
    print("SDGZero Ingest Pipeline — Phase 1")
    print("=" * 50)

    # Step 1: Scrape
    print("\n[1/2] Scraping SDGZero directory...")
    businesses = scrape_all(per_page=100, delay=1.0)
    print(f"  Total businesses scraped: {len(businesses)}")

    if not businesses:
        print("No businesses found. Exiting.")
        return

    # Step 2: Embed + store
    print("\n[2/2] Embedding and storing in ChromaDB...")
    store = BusinessStore(persist_dir="./chroma_db")
    store.upsert(businesses)

    print(f"\nDone! ChromaDB now has {store.count()} businesses.")
    print("Run demo_search.py to test semantic search.")


if __name__ == "__main__":
    run()
