"""
Quick ChromaDB inspector — browse and filter stored businesses.

Usage examples:
    python inspect_db.py                         # summary stats
    python inspect_db.py --sample 5             # show 5 random records
    python inspect_db.py --sdg                  # show only businesses WITH sdg_tags
    python inspect_db.py --city London          # filter by city
    python inspect_db.py --country "United Kingdom" --sdg --sample 10
    python inspect_db.py --fields               # list all metadata keys
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db.chroma_store import BusinessStore


def print_record(meta: dict, i: int) -> None:
    print(f"\n[{i}] {meta.get('name', '?')}  (id={meta.get('id')})")
    print(f"     Location : {', '.join(filter(None, [meta.get('city'), meta.get('region'), meta.get('country')]))}")
    print(f"     Category : {meta.get('categories') or '-'}")
    print(f"     Sector   : {meta.get('job_sector') or '-'}")
    print(f"     SDG tags : {meta.get('sdg_tags') or '(none)'}")
    print(f"     SDG pred : {meta.get('predicted_sdg_tags') or '(none)'}")
    print(f"     SDG slugs: {meta.get('sdg_slugs') or '(none)'}")
    print(f"     Tier     : {meta.get('membership_tier') or '-'}")
    print(f"     Website  : {meta.get('website') or '-'}")
    print(f"     LinkedIn : {meta.get('linkedin') or '-'}")
    print(f"     TikTok   : {meta.get('tiktok') or '-'}")
    print(f"     Claimed  : {meta.get('claimed') or '-'}")
    print(f"     Rating   : {meta.get('rating')}  ({meta.get('rating_count')} reviews)")


def main():
    parser = argparse.ArgumentParser(description="Inspect ChromaDB contents")
    parser.add_argument("--sample", type=int, default=0, help="Show N records")
    parser.add_argument("--sdg", action="store_true", help="Only records with SDG tags")
    parser.add_argument("--city", type=str, default="", help="Filter by city")
    parser.add_argument("--country", type=str, default="", help="Filter by country")
    parser.add_argument("--fields", action="store_true", help="List all metadata field names")
    args = parser.parse_args()

    store = BusinessStore(persist_dir="./chroma_db")
    all_data = store.collection.get(include=["metadatas"])
    metadatas = all_data["metadatas"]

    if not metadatas:
        print("Database is empty. Run: python -m pipeline.ingest")
        return

    # --- stats ---
    total = len(metadatas)
    with_sdg = [m for m in metadatas if m.get("sdg_tags", "").strip() or m.get("predicted_sdg_tags", "").strip()]
    with_website = [m for m in metadatas if m.get("website", "").strip()]
    with_linkedin = [m for m in metadatas if m.get("linkedin", "").strip()]
    claimed = [m for m in metadatas if m.get("claimed", "").strip()]
    countries = sorted(set(m.get("country", "") for m in metadatas if m.get("country")))

    print(f"\n{'='*50}")
    print(f"  ChromaDB Summary")
    print(f"{'='*50}")
    print(f"  Total businesses : {total}")
    print(f"  With SDG tags    : {len(with_sdg)}  ({len(with_sdg)*100//total}%)")
    print(f"  With website     : {len(with_website)}")
    print(f"  With LinkedIn    : {len(with_linkedin)}")
    print(f"  Claimed profiles : {len(claimed)}")
    print(f"  Countries ({len(countries)})   : {', '.join(countries[:10])}{'...' if len(countries)>10 else ''}")

    if args.fields:
        keys = sorted(set(k for m in metadatas for k in m.keys()))
        print(f"\n  Metadata fields ({len(keys)}):")
        for k in keys:
            print(f"    - {k}")
        return

    # --- filter ---
    subset = metadatas
    if args.sdg:
        subset = [m for m in subset if m.get("sdg_tags", "").strip()]
    if args.city:
        subset = [m for m in subset if args.city.lower() in m.get("city", "").lower()]
    if args.country:
        subset = [m for m in subset if args.country.lower() in m.get("country", "").lower()]

    print(f"\n  Matching records : {len(subset)}")

    if args.sample and subset:
        import random
        sample = random.sample(subset, min(args.sample, len(subset)))
        for i, m in enumerate(sample, 1):
            print_record(m, i)


if __name__ == "__main__":
    main()
