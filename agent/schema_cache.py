"""
Schema cache — valid filter values from ChromaDB.

Strategy (fast restarts):
  1. On first call, check for schema_cache.json on disk.
     If present and not stale (default TTL: 7 days), load it instantly.
  2. If missing or stale, query ChromaDB metadata directly (no embedding
     model needed — bypasses the slow SentenceTransformer load).
  3. Save result to schema_cache.json for next restart.
  4. Cache in-process memory for the lifetime of the server process.

Invalidation:
  - Call get_schema(refresh=True) after DB updates (ingest / fix scripts).
  - Delete schema_cache.json to force a cold rebuild on next restart.

Usage:
    from agent.schema_cache import get_schema
    schema = get_schema()
    # schema["city"]          → ["London", "Preston", ...]
    # schema["business_type"] → ["B2B", "B2C", "Both"]
    # schema["sdg_tags"]      → ["Climate Action", ...]
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(os.getenv("SCHEMA_CACHE_FILE", "./schema_cache.json"))
_CACHE_TTL_DAYS = int(os.getenv("SCHEMA_CACHE_TTL_DAYS", "7"))

# In-process memory cache (cleared on process restart, but that's fine —
# we'll just read from disk next time instead of hitting ChromaDB)
_mem_cache: Optional[dict] = None


def _load_from_disk() -> Optional[dict]:
    """Return schema from disk cache if it exists and is fresh enough."""
    if not _CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        age_days = (time.time() - data.get("_ts", 0)) / 86400
        if age_days > _CACHE_TTL_DAYS:
            logger.info(f"Schema disk cache expired ({age_days:.1f}d > {_CACHE_TTL_DAYS}d), rebuilding")
            return None
        logger.info(f"Schema loaded from disk cache ({_CACHE_FILE}, {age_days:.1f}d old)")
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception as e:
        logger.warning(f"Failed to read schema disk cache: {e}")
        return None


def _save_to_disk(schema: dict) -> None:
    """Persist schema to disk with a timestamp."""
    try:
        payload = {"_ts": time.time(), **schema}
        _CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Schema saved to disk cache: {_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to write schema disk cache: {e}")


def _build_from_db() -> dict:
    """Query PostgreSQL directly — replaces ChromaDB metadata scan."""
    from db.pg_store import PGStore
    store = PGStore()
    schema = store.build_schema_data()
    logger.info(
        f"Schema built from PostgreSQL — cities={len(schema['city'])}, "
        f"categories={len(schema['categories'])}, sdg_tags={len(schema['sdg_tags'])}"
    )
    return schema


def get_schema(refresh: bool = False) -> dict:
    """
    Return a dict of valid filter values, keyed by field name.

    Keys:
        city            list[str]  — exact values to pass to filters["city"]
        business_type   list[str]  — "B2B" | "B2C" | "Both"
        job_sector      list[str]  — "Private" | "Public" | "Agencies"
        company_size    list[str]  — e.g. "SME (Less than 250) Staff"
        categories      list[str]  — e.g. "Energy & Renewables"
        sdg_tags        list[str]  — e.g. "Climate Action"

    Args:
        refresh: Force rebuild from ChromaDB and update disk cache.
    """
    global _mem_cache

    if not refresh and _mem_cache is not None:
        return _mem_cache

    if not refresh:
        cached = _load_from_disk()
        if cached is not None:
            _mem_cache = cached
            return _mem_cache

    schema = _build_from_db()
    _save_to_disk(schema)
    _mem_cache = schema
    return _mem_cache


def invalidate_cache() -> None:
    """Clear both in-process and disk caches. Call after DB updates."""
    global _mem_cache
    _mem_cache = None
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
        logger.info(f"Schema disk cache deleted: {_CACHE_FILE}")


def print_schema() -> None:
    """Pretty-print all valid filter values (useful for CLI inspection)."""
    schema = get_schema()
    print("\n=== Valid Filter Values ===\n")
    for field, values in schema.items():
        print(f"  {field} ({len(values)} options):")
        for v in values:
            print(f"    • {v}")
        print()
