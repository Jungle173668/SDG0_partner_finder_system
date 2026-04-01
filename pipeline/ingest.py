"""
Ingest pipeline: scrape → parse → embed → store.
Run this to populate/refresh the PostgreSQL + pgvector database.

Usage:
    python -m pipeline.ingest
"""

import sys
import os

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.spider import scrape_all
from db.pg_store import PGStore


def run():
    print("=" * 50)
    print("SDG: Zero Ingest Pipeline")
    print("=" * 50)

    # Step 1: Scrape
    print("\n[1/3] Scraping SDG: Zero directory...")
    businesses = scrape_all(per_page=100, delay=1.0)
    print(f"  Total businesses scraped: {len(businesses)}")

    if not businesses:
        print("No businesses found. Exiting.")
        return

    # Step 2: Embed
    print("\n[2/3] Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"  Encoding {len(businesses)} businesses...")
    rows = []
    for b in businesses:
        meta = b.to_metadata()
        doc = b.to_embedding_text()
        embedding = encoder.encode([doc], normalize_embeddings=True)[0].tolist()
        rows.append({
            **meta,
            "document": doc,
            "predicted_sdg_tags": "",   # populated later by fix_sdg_tags.py
            "embedding": embedding,
        })

    # Step 3: Upsert into PostgreSQL
    print("\n[3/3] Upserting into PostgreSQL...")
    store = PGStore()
    store.upsert_batch(rows)

    print(f"\nDone! Upserted {len(rows)} businesses into PostgreSQL.")

    # Invalidate schema disk cache so next API call reflects new data
    from agent.schema_cache import invalidate_cache
    invalidate_cache()
    print("Schema disk cache cleared.")
    print("Run 'python scripts/fix_sdg_tags.py' to classify predicted SDG tags.")


if __name__ == "__main__":
    run()
