"""
GET /api/schema — returns valid filter values for frontend dropdowns.

Reads from schema_cache.py which queries the live ChromaDB.
Response is cached in-memory for 1 hour to avoid repeated DB reads.
"""

import logging
import time
from functools import lru_cache
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

_schema_cache: dict = {}
_cache_ts: float = 0
_CACHE_TTL = 3600  # 1 hour


@router.get("/schema")
def get_schema():
    """
    Returns available filter values from the database.

    Response shape:
    {
      "city": ["London", "Edinburgh", ...],
      "business_type": ["B2B", "B2C", "Both"],
      "job_sector": ["Private", "Public", "Third Sector"],
      "company_size": ["Micro", "SME", ...],
      "categories": ["Energy & Renewables", ...],
      "sdg_tags": ["Climate Action", ...]
    }
    """
    global _schema_cache, _cache_ts

    now = time.time()
    if _schema_cache and (now - _cache_ts) < _CACHE_TTL:
        return _schema_cache

    try:
        from agent.schema_cache import get_schema as _get_schema
        schema = _get_schema()
        _schema_cache = schema
        _cache_ts = now
        return schema
    except Exception as e:
        logger.error(f"schema route: failed to load schema — {e}")
        if _schema_cache:
            return _schema_cache  # stale but better than error
        raise HTTPException(status_code=500, detail=f"Failed to load schema: {e}")
