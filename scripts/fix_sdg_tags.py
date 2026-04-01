"""
Fix corrupted SDG tags in existing PostgreSQL records.

Scans every row's sdg_tags and predicted_sdg_tags columns,
applies normalize_sdg_tags(), and writes back only the rows that changed.

Usage:
    python scripts/fix_sdg_tags.py [--dry-run]

Options:
    --dry-run  Print what would change without writing to DB.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.sdg_normalize import normalize_sdg_tags


def fix_tag_field(raw: str) -> tuple[str, bool]:
    """
    Normalise a comma-separated SDG tag string.
    Returns (fixed_string, was_changed).
    """
    if not raw or not raw.strip():
        return raw, False
    original = [t.strip() for t in raw.split(",") if t.strip()]
    fixed = normalize_sdg_tags(original)
    fixed_str = ", ".join(fixed)
    original_str = ", ".join(original)
    return fixed_str, fixed_str != original_str


def main():
    parser = argparse.ArgumentParser(description="Fix corrupted SDG tags in PostgreSQL.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = parser.parse_args()

    from db.pg_store import PGStore
    store = PGStore()

    # Fetch all records (id + tag fields only)
    with store._cursor(dict_rows=True) as cur:
        cur.execute("SELECT id, name, sdg_tags, predicted_sdg_tags FROM businesses")
        rows = list(cur.fetchall())

    print(f"Connected to PostgreSQL — {len(rows)} records")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    updates = []   # list of (fixed_sdg_tags, fixed_predicted, id)
    total_fields_fixed = 0

    for row in rows:
        sdg_fixed, sdg_changed = fix_tag_field(row["sdg_tags"] or "")
        pred_fixed, pred_changed = fix_tag_field(row["predicted_sdg_tags"] or "")

        if sdg_changed or pred_changed:
            name = row["name"] or str(row["id"])
            if sdg_changed:
                print(f"  [sdg_tags] {name}")
                print(f"    Before: {row['sdg_tags']}")
                print(f"    After:  {sdg_fixed}\n")
                total_fields_fixed += 1
            if pred_changed:
                print(f"  [predicted_sdg_tags] {name}")
                print(f"    Before: {row['predicted_sdg_tags']}")
                print(f"    After:  {pred_fixed}\n")
                total_fields_fixed += 1
            updates.append((sdg_fixed, pred_fixed, row["id"]))

    print(f"Records to update: {len(updates)}  (fields fixed: {total_fields_fixed})")

    if not updates:
        print("Nothing to fix — DB is clean.")
        return

    if args.dry_run:
        print("\nDry run complete. No changes written.")
        return

    # Write back in batches of 100
    batch_size = 100
    with store._cursor(dict_rows=False) as cur:
        from psycopg2.extras import execute_batch
        execute_batch(
            cur,
            "UPDATE businesses SET sdg_tags = %s, predicted_sdg_tags = %s WHERE id = %s",
            updates,
            page_size=batch_size,
        )
    print(f"\nDone. {len(updates)} records fixed in PostgreSQL.")

    # Invalidate schema disk cache so next API call rebuilds from fresh DB data
    from agent.schema_cache import invalidate_cache
    invalidate_cache()
    print("Schema disk cache cleared — will rebuild on next API request.")


if __name__ == "__main__":
    main()
