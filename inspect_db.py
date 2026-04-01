"""
PostgreSQL DB inspector — browse and filter stored businesses.

Usage examples:
    python inspect_db.py                         # summary stats
    python inspect_db.py --sample 5             # show 5 random records
    python inspect_db.py --sdg                  # show only businesses WITH sdg_tags
    python inspect_db.py --city London          # filter by city
    python inspect_db.py --country "United Kingdom" --sdg --sample 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_record(row: dict, i: int) -> None:
    print(f"\n[{i}] {row.get('name', '?')}  (id={row.get('id')})")
    print(f"     Location : {', '.join(filter(None, [row.get('city'), row.get('region'), row.get('country')]))}")
    print(f"     Category : {row.get('categories') or '-'}")
    print(f"     Sector   : {row.get('job_sector') or '-'}")
    print(f"     SDG tags : {row.get('sdg_tags') or '(none)'}")
    print(f"     SDG pred : {row.get('predicted_sdg_tags') or '(none)'}")
    print(f"     SDG slugs: {row.get('sdg_slugs') or '(none)'}")
    print(f"     Tier     : {row.get('membership_tier') or '-'}")
    print(f"     Website  : {row.get('website') or '-'}")
    print(f"     LinkedIn : {row.get('linkedin') or '-'}")
    print(f"     TikTok   : {row.get('tiktok') or '-'}")
    print(f"     Claimed  : {row.get('claimed') or '-'}")
    print(f"     Rating   : {row.get('rating')}  ({row.get('rating_count')} reviews)")


def main():
    parser = argparse.ArgumentParser(description="Inspect PostgreSQL DB contents")
    parser.add_argument("--sample", type=int, default=0, help="Show N records")
    parser.add_argument("--sdg", action="store_true", help="Only records with SDG tags")
    parser.add_argument("--city", type=str, default="", help="Filter by city")
    parser.add_argument("--country", type=str, default="", help="Filter by country")
    args = parser.parse_args()

    from db.pg_store import PGStore
    store = PGStore()

    with store._cursor(dict_rows=True) as cur:
        cur.execute(
            "SELECT id, name, city, region, country, categories, job_sector, "
            "sdg_tags, predicted_sdg_tags, sdg_slugs, membership_tier, "
            "website, linkedin, tiktok, claimed, rating, rating_count "
            "FROM businesses"
        )
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        print("Database is empty. Run: python -m pipeline.ingest")
        return

    # --- stats ---
    total = len(rows)
    with_sdg = [r for r in rows if (r.get("sdg_tags") or "").strip() or (r.get("predicted_sdg_tags") or "").strip()]
    with_website = [r for r in rows if (r.get("website") or "").strip()]
    with_linkedin = [r for r in rows if (r.get("linkedin") or "").strip()]
    claimed = [r for r in rows if (r.get("claimed") or "").strip()]
    countries = sorted(set(r.get("country", "") for r in rows if r.get("country")))

    print(f"\n{'='*50}")
    print(f"  PostgreSQL Summary")
    print(f"{'='*50}")
    print(f"  Total businesses : {total}")
    print(f"  With SDG tags    : {len(with_sdg)}  ({len(with_sdg)*100//total}%)")
    print(f"  With website     : {len(with_website)}")
    print(f"  With LinkedIn    : {len(with_linkedin)}")
    print(f"  Claimed profiles : {len(claimed)}")
    print(f"  Countries ({len(countries)})   : {', '.join(countries[:10])}{'...' if len(countries)>10 else ''}")

    # --- filter ---
    subset = rows
    if args.sdg:
        subset = [r for r in subset if (r.get("sdg_tags") or "").strip()]
    if args.city:
        subset = [r for r in subset if args.city.lower() in (r.get("city") or "").lower()]
    if args.country:
        subset = [r for r in subset if args.country.lower() in (r.get("country") or "").lower()]

    print(f"\n  Matching records : {len(subset)}")

    if args.sample and subset:
        import random
        sample = random.sample(subset, min(args.sample, len(subset)))
        for i, r in enumerate(sample, 1):
            print_record(r, i)


if __name__ == "__main__":
    main()
