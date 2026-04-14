"""
Incremental update pipeline: scrape → embed → upsert only new/changed businesses.

Compares scraped_at timestamps to detect new or updated companies.
Only re-embeds companies that are genuinely new or have changed content.

Usage:
    python -m pipeline.update             # incremental update
    python -m pipeline.update --dry-run   # show what would be updated, don't write
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(dry_run: bool = False) -> dict:
    from scraper.spider import scrape_all
    from db.pg_store import PGStore

    store = PGStore()

    # -------------------------------------------------------------------------
    # Step 1: Get existing scraped_at timestamps from DB
    # -------------------------------------------------------------------------
    print("[1/4] Fetching existing business timestamps from DB...")
    conn = store._pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, scraped_at FROM businesses")
        existing = {str(row[0]): row[1] for row in cur.fetchall()}
        cur.close()
    finally:
        store._pool.putconn(conn)

    print(f"  Existing in DB: {len(existing)} businesses")

    # -------------------------------------------------------------------------
    # Step 2: Scrape all from SDGZero
    # -------------------------------------------------------------------------
    print("\n[2/4] Scraping SDGZero directory...")
    businesses = scrape_all(per_page=100, delay=1.0)
    print(f"  Scraped: {len(businesses)} businesses")

    if not businesses:
        print("No businesses scraped. Exiting.")
        return {"total_scraped": 0, "new": 0, "updated": 0, "unchanged": 0}

    # -------------------------------------------------------------------------
    # Step 3: Filter to only new or updated businesses
    # -------------------------------------------------------------------------
    new_businesses = []
    updated_businesses = []
    unchanged_count = 0

    for b in businesses:
        bid = str(b.id)
        scraped_at = b.scraped_at  # ISO string from API

        if bid not in existing:
            new_businesses.append(b)
        elif scraped_at and existing[bid] and str(scraped_at) > str(existing[bid]):
            updated_businesses.append(b)
        else:
            unchanged_count += 1

    to_process = new_businesses + updated_businesses
    print(f"\n  New      : {len(new_businesses)}")
    print(f"  Updated  : {len(updated_businesses)}")
    print(f"  Unchanged: {unchanged_count}")

    if not to_process:
        print("\nNothing to update.")
        return {
            "total_scraped": len(businesses),
            "new": 0,
            "updated": 0,
            "unchanged": unchanged_count,
        }

    if dry_run:
        print("\n[DRY RUN] Would process:")
        for b in to_process[:10]:
            print(f"  {'NEW' if b in new_businesses else 'UPD'} {b.name}")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
        return {
            "total_scraped": len(businesses),
            "new": len(new_businesses),
            "updated": len(updated_businesses),
            "unchanged": unchanged_count,
            "dry_run": True,
        }

    # -------------------------------------------------------------------------
    # Step 4: Embed and upsert only new/updated businesses
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Embedding {len(to_process)} businesses...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    rows = []
    for b in to_process:
        meta = b.to_metadata()
        doc = b.to_embedding_text()
        embedding = encoder.encode([doc], normalize_embeddings=True)[0].tolist()
        rows.append({
            **meta,
            "document": doc,
            "predicted_sdg_tags": "",
            "embedding": embedding,
        })

    print(f"\n[4/4] Upserting {len(rows)} businesses into PostgreSQL...")
    store.upsert_batch(rows)

    # Invalidate schema cache so next API call reflects new data
    from agent.schema_cache import invalidate_cache
    invalidate_cache()
    print("Schema cache cleared.")

    result = {
        "total_scraped": len(businesses),
        "new": len(new_businesses),
        "updated": len(updated_businesses),
        "unchanged": unchanged_count,
    }
    print(f"\nDone. {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental SDGZero update")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing to DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
