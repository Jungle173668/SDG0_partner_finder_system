"""
Schema cache — valid filter values from ChromaDB.

Queries the database once and caches the results in memory so that
the frontend / CLI can show users only values that actually exist.

Usage:
    from agent.schema_cache import get_schema
    schema = get_schema()
    # schema["city"]          → ["London", "Preston", ...]
    # schema["business_type"] → ["B2B", "B2C", "Both"]
    # schema["sdg_tags"]      → ["Climate Action", ...]

Note: called automatically on first use (lazy, cached). Re-call
      get_schema(refresh=True) if the database has been updated.
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# business_type: DB stores verbose form, expose short form to callers
_BTYPE_DISPLAY = {
    "Business2Business (B2B)": "B2B",
    "Business2Consumer (B2C)": "B2C",
    "Both": "Both",
}

# job_sector: DB stores "Private Sector" etc., normalise for display
_SECTOR_DISPLAY = {
    "Private Sector": "Private",
    "Public Sector":  "Public",
    "Agencies":       "Agencies",
}


@lru_cache(maxsize=1)
def get_schema() -> dict:
    """
    Return a dict of valid filter values, keyed by field name.

    Keys:
        city            list[str]  — exact values to pass to filters["city"]
        business_type   list[str]  — "B2B" | "B2C" | "Both"
        job_sector      list[str]  — "Private" | "Public" | "Agencies"
        company_size    list[str]  — e.g. "SME (Less than 250) Staff"
        categories      list[str]  — e.g. "Energy & Renewables"
        sdg_tags        list[str]  — e.g. "Climate Action"

    Returns cached result after first call; call get_schema.cache_clear()
    to force a refresh.
    """
    import os
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    from db.chroma_store import BusinessStore

    store = BusinessStore(persist_dir=os.getenv("CHROMA_DIR", "./chroma_db"))
    raw = store.collection.get(include=["metadatas"], limit=600)
    metas = raw["metadatas"]

    # Simple scalar fields — unique non-empty values
    cities = sorted({m.get("city", "") for m in metas if m.get("city", "").strip()})

    business_types_raw = sorted(
        {m.get("business_type", "") for m in metas if m.get("business_type", "").strip()}
    )
    business_types = [_BTYPE_DISPLAY.get(b, b) for b in business_types_raw]

    job_sectors_raw = sorted(
        {m.get("job_sector", "") for m in metas if m.get("job_sector", "").strip()}
    )
    job_sectors = [_SECTOR_DISPLAY.get(s, s) for s in job_sectors_raw]

    company_sizes = sorted(
        {m.get("company_size", "") for m in metas if m.get("company_size", "").strip()}
    )

    # Multi-value fields — split comma-separated strings
    categories: set = set()
    for m in metas:
        for c in (m.get("categories", "") or "").split(","):
            c = c.strip()
            if c:
                categories.add(c)

    sdg_tags: set = set()
    for m in metas:
        for field in ("sdg_tags", "predicted_sdg_tags"):
            for tag in (m.get(field, "") or "").split(","):
                tag = tag.strip()
                if tag:
                    sdg_tags.add(tag)

    schema = {
        "city": cities,
        "business_type": business_types,
        "job_sector": job_sectors,
        "company_size": company_sizes,
        "categories": sorted(categories),
        "sdg_tags": sorted(sdg_tags),
    }

    logger.debug(
        f"Schema loaded — "
        f"cities={len(schema['city'])}, "
        f"categories={len(schema['categories'])}, "
        f"sdg_tags={len(schema['sdg_tags'])}"
    )
    return schema


def print_schema() -> None:
    """Pretty-print all valid filter values (useful for CLI inspection)."""
    schema = get_schema()
    print("\n=== Valid Filter Values ===\n")
    for field, values in schema.items():
        print(f"  {field} ({len(values)} options):")
        for v in values:
            print(f"    • {v}")
        print()
