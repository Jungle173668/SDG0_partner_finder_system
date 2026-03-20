"""
One-time script: fix corrupted SDG tags in existing ChromaDB records.

Scans every document's sdg_tags and predicted_sdg_tags metadata fields,
applies normalize_sdg_tags(), and writes back only the records that changed.

Usage:
    python scripts/fix_sdg_tags.py [--dry-run] [--chroma-dir ./chroma_db]

Options:
    --dry-run     Print what would change without writing to DB.
    --chroma-dir  Path to ChromaDB directory (default: ./chroma_db).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

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
    parser = argparse.ArgumentParser(description="Fix corrupted SDG tags in ChromaDB.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="ChromaDB directory path.")
    args = parser.parse_args()

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=args.chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection("sdgzero_businesses")

    print(f"Connected to ChromaDB at '{args.chroma_dir}' — {collection.count()} docs")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    # Fetch all records
    raw = collection.get(include=["metadatas"], limit=10000)
    ids = raw["ids"]
    metas = raw["metadatas"]

    changed_ids = []
    changed_metas = []
    total_fields_fixed = 0

    for doc_id, meta in zip(ids, metas):
        updated = dict(meta)
        doc_changed = False

        for field in ("sdg_tags", "predicted_sdg_tags"):
            original_val = meta.get(field, "")
            fixed_val, was_changed = fix_tag_field(original_val)
            if was_changed:
                name = meta.get("name", doc_id)
                print(f"  [{field}] {name}")
                print(f"    Before: {original_val}")
                print(f"    After:  {fixed_val}\n")
                updated[field] = fixed_val
                doc_changed = True
                total_fields_fixed += 1

        if doc_changed:
            changed_ids.append(doc_id)
            changed_metas.append(updated)

    print(f"Records to update: {len(changed_ids)}  (fields fixed: {total_fields_fixed})")

    if not changed_ids:
        print("Nothing to fix — DB is clean.")
        return

    if args.dry_run:
        print("\nDry run complete. No changes written.")
        return

    # Write back in batches of 100
    batch_size = 100
    for i in range(0, len(changed_ids), batch_size):
        batch_ids = changed_ids[i : i + batch_size]
        batch_metas = changed_metas[i : i + batch_size]
        collection.update(ids=batch_ids, metadatas=batch_metas)
        print(f"  Updated {min(i + batch_size, len(changed_ids))}/{len(changed_ids)} records")

    print(f"\nDone. {len(changed_ids)} records fixed in '{args.chroma_dir}'.")

    # Invalidate schema disk cache so next API call rebuilds from fresh DB data
    from agent.schema_cache import invalidate_cache
    invalidate_cache()
    print("Schema disk cache cleared — will rebuild on next API request.")


if __name__ == "__main__":
    main()
