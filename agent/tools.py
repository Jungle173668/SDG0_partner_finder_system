"""
Search tools for the SearchAgent.

Three tools map to three search modes:
  semantic_search  — vector similarity (ChromaDB)
  sql_filter       — exact metadata filter (ChromaDB where clause)
  hybrid_search    — vector + metadata filter combined

All tools are implemented as plain functions now.
Step 2 will add @tool decorators and LLM-based tool routing.

The shared BusinessStore and SentenceTransformer are module-level singletons
(loaded once on first call) to avoid re-loading the model on every request.
"""

import os
import logging
import numpy as np
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons — loaded once, reused across calls
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_store():
    """Lazy-load BusinessStore singleton."""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    from db.chroma_store import BusinessStore
    return BusinessStore(persist_dir=os.getenv("CHROMA_DIR", "./chroma_db"))


@lru_cache(maxsize=1)
def _get_encoder():
    """Lazy-load SentenceTransformer singleton (same model as ChromaDB)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Internal helper: format ChromaDB query results
# (mirrors BusinessStore.search() output format)
# ---------------------------------------------------------------------------

def _format_results(ids, metadatas, documents, distances) -> list[dict]:
    output = []
    for mid, meta, doc, dist in zip(ids, metadatas, documents, distances):
        output.append({
            # identity
            "id": mid,
            "slug": meta.get("slug", ""),
            "name": meta.get("name", ""),
            "url": meta.get("url", ""),
            # location
            "street": meta.get("street", ""),
            "city": meta.get("city", ""),
            "region": meta.get("region", ""),
            "country": meta.get("country", ""),
            # contact
            "phone": meta.get("phone", ""),
            "website": meta.get("website", ""),
            "linkedin": meta.get("linkedin", ""),
            # business info
            "categories": meta.get("categories", ""),
            "job_sector": meta.get("job_sector", ""),
            "business_type": meta.get("business_type", ""),
            "company_size": meta.get("company_size", ""),
            "package_id": meta.get("package_id", 1),
            "claimed": meta.get("claimed", ""),
            "founder_name": meta.get("founder_name", ""),
            # sdg
            "sdg_tags": meta.get("sdg_tags", ""),
            "predicted_sdg_tags": meta.get("predicted_sdg_tags", ""),
            "membership_tier": meta.get("membership_tier", ""),
            # score
            "similarity": round(1 - dist, 4),
            # full embedding text (for Agent RAG context)
            "document": doc,
        })
    return output


# ---------------------------------------------------------------------------
# Tool 1: semantic_search
# Used in Step 1 (and always as part of hybrid_search)
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    n_results: int = 20,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Search businesses by semantic similarity to a natural language query.

    Uses the same all-MiniLM-L6-v2 embedding model as the stored vectors,
    so the query embedding lands in the same vector space.

    Args:
        query:     Natural language description to search for.
        n_results: Max number of results (default 20 for candidate pool).
        where:     Optional ChromaDB metadata filter to apply alongside vector search.

    Returns:
        List of business dicts sorted by cosine similarity (highest first).
    """
    store = _get_store()
    return store.search(query, n_results=n_results, where=where)


def semantic_search_from_embedding(
    embedding: list[float],
    n_results: int = 20,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Search from a pre-computed embedding vector (used for HyDE averaged embedding).

    Bypasses re-encoding — the caller already has the averaged HyDE vector.

    Args:
        embedding: 384-dim float list (pre-computed, normalised).
        n_results: Max results.
        where:     Optional metadata filter.

    Returns:
        List of business dicts sorted by cosine similarity.
    """
    store = _get_store()
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": n_results,
        "include": ["metadatas", "documents", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = store.collection.query(**kwargs)
    return _format_results(
        raw["ids"][0],
        raw["metadatas"][0],
        raw["documents"][0],
        raw["distances"][0],
    )


# ---------------------------------------------------------------------------
# Tool 2: sql_filter
# ---------------------------------------------------------------------------

def sql_filter(
    filters: dict,
    n_results: int = 100,
) -> list[dict]:
    """
    Filter businesses by exact metadata conditions (no vector similarity).

    Uses ChromaDB's where clause for city/business_type/sector/claimed filters.
    SDG tag filtering (sdg_tags / predicted_sdg_tags) uses Python post-processing
    because ChromaDB doesn't support substring matching on comma-separated strings.

    Results are sorted by package_id (membership tier) descending as a quality proxy.

    Args:
        filters:   SearchFilters dict (city, business_type, claimed, sdg_tags, ...).
        n_results: Max results to return (default 100 — no vector cap needed here).

    Returns:
        List of business dicts matching all filter conditions.

    PostgreSQL migration (Phase 4):
        Replace with:  SELECT ... FROM businesses
                       WHERE city = %s AND sdg_tags LIKE '%Climate Action%'
                          OR predicted_sdg_tags LIKE '%Climate Action%'
                       ORDER BY package_id DESC
    """
    store = _get_store()
    where = build_chroma_where(filters)

    kwargs = {
        "include": ["metadatas", "documents"],
        "limit": 500,   # pull more, post-filter SDG, then trim
    }
    if where:
        kwargs["where"] = where

    raw = store.collection.get(**kwargs)

    if not raw["ids"]:
        return []

    results = []
    for mid, meta, doc in zip(raw["ids"], raw["metadatas"], raw["documents"]):
        results.append({
            "id": mid,
            "slug": meta.get("slug", ""),
            "name": meta.get("name", ""),
            "url": meta.get("url", ""),
            "street": meta.get("street", ""),
            "city": meta.get("city", ""),
            "region": meta.get("region", ""),
            "country": meta.get("country", ""),
            "phone": meta.get("phone", ""),
            "website": meta.get("website", ""),
            "linkedin": meta.get("linkedin", ""),
            "categories": meta.get("categories", ""),
            "job_sector": meta.get("job_sector", ""),
            "business_type": meta.get("business_type", ""),
            "company_size": meta.get("company_size", ""),
            "package_id": meta.get("package_id", 1),
            "claimed": meta.get("claimed", ""),
            "founder_name": meta.get("founder_name", ""),
            "sdg_tags": meta.get("sdg_tags", ""),
            "predicted_sdg_tags": meta.get("predicted_sdg_tags", ""),
            "membership_tier": meta.get("membership_tier", ""),
            "similarity": None,   # no vector score for pure filter
            "document": doc,
        })

    # Python post-filter for multi-value fields (substring match on comma-separated strings)
    category_required = filters.get("categories", "")
    if category_required:
        results = post_filter_categories(results, category_required)

    sdg_required = filters.get("sdg_tags", [])
    if sdg_required:
        results = post_filter_sdg(results, sdg_required)

    # Sort by package_id desc (higher = premium member = more likely quality result)
    results.sort(key=lambda r: r.get("package_id") or 1, reverse=True)
    return results[:n_results]


# ---------------------------------------------------------------------------
# Tool 3: hybrid_search
# ---------------------------------------------------------------------------

def hybrid_search(
    embedding: list[float],
    filters: dict,
    n_results: int = 20,
) -> list[dict]:
    """
    Combine vector similarity search with exact metadata filtering.

    Passes the ChromaDB where clause directly into the vector query so both
    happen in a single round-trip (not sequential filter-then-rank).

    When the combined query returns too few results, the caller in search_agent.py
    applies three-level fallback: relax filters → pure vector.

    Args:
        embedding: Pre-computed 384-dim query vector (averaged HyDE embeddings).
        filters:   SearchFilters dict.
        n_results: Max results.

    Returns:
        List of business dicts satisfying all filters, ranked by cosine similarity.

    PostgreSQL migration (Phase 4):
        Becomes a single query:
            SELECT *, embedding <=> %s AS dist FROM businesses
            WHERE city = %s AND (sdg_tags LIKE '%X%' OR predicted_sdg_tags LIKE '%X%')
            ORDER BY dist LIMIT 20
    """
    # TODO Phase 4: Replace with a single pgvector query (SQL-first, then rank):
    #   SELECT *, embedding <=> %s AS dist FROM businesses
    #   WHERE city = %s AND categories LIKE '%X%'
    #     AND (sdg_tags LIKE '%Y%' OR predicted_sdg_tags LIKE '%Y%')
    #   ORDER BY dist LIMIT n
    # Current approach: vector search first (with scalar WHERE), then Python
    # post-filter categories/sdg_tags. Semantically correct but relies on a
    # large pool to survive post-filtering. Correct order should be:
    # SQL filter all conditions first → rank survivors by similarity.
    where = build_chroma_where(filters)
    has_post_filters = bool(filters.get("categories") or filters.get("sdg_tags"))
    pool_size = n_results * 15 if has_post_filters else n_results * 3
    results = semantic_search_from_embedding(embedding, n_results=pool_size, where=where)

    # Python post-filter for multi-value fields
    category_required = filters.get("categories", "")
    if category_required:
        results = post_filter_categories(results, category_required)

    sdg_required = filters.get("sdg_tags", [])
    if sdg_required:
        results = post_filter_sdg(results, sdg_required)

    return results[:n_results]


# ---------------------------------------------------------------------------
# Filter builder (used by sql_filter and hybrid_search)
# ---------------------------------------------------------------------------

def build_chroma_where(filters: dict) -> Optional[dict]:
    """
    Convert a SearchFilters dict to a ChromaDB where clause.

    ChromaDB supports: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or
    SDG tag filtering (comma-separated string) is handled by Python post-processing.

    Args:
        filters: SearchFilters dict with optional keys.

    Returns:
        ChromaDB where dict, or None if no filterable conditions.

    Examples:
        {"city": "London", "claimed": True}
        → {"$and": [{"city": {"$eq": "London"}}, {"claimed": {"$eq": "Yes"}}]}
    """
    # business_type normalisation: DB stores verbose form
    _BTYPE_MAP = {
        "B2B": "Business2Business (B2B)",
        "B2C": "Business2Consumer (B2C)",
        "Both": "Both",
        "b2b": "Business2Business (B2B)",
        "b2c": "Business2Consumer (B2C)",
    }

    conditions = []

    if filters.get("city"):
        conditions.append({"city": {"$eq": filters["city"]}})

    if filters.get("business_type"):
        raw_bt = filters["business_type"]
        bt = _BTYPE_MAP.get(raw_bt, raw_bt)   # map short form → DB form
        conditions.append({"business_type": {"$eq": bt}})

    if filters.get("job_sector"):
        conditions.append({"job_sector": {"$eq": filters["job_sector"]}})

    if filters.get("company_size"):
        conditions.append({"company_size": {"$eq": filters["company_size"]}})

    if filters.get("claimed") is True or filters.get("claimed") == "Yes":
        conditions.append({"claimed": {"$eq": "Yes"}})

    # SDG tag and categories: ChromaDB can't do substring match on comma-separated
    # strings — both are handled by Python post-processing below.

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def post_filter_categories(results: list[dict], category: str) -> list[dict]:
    """
    Python-side category filter (comma-separated field, substring match).

    Args:
        results:  List of business dicts.
        category: Single category string, e.g. "Energy & Renewables".

    Returns:
        Filtered list (companies whose categories field contains the value).
    """
    if not category:
        return results
    cat_lower = category.lower()
    return [r for r in results if cat_lower in (r.get("categories", "") or "").lower()]


def post_filter_sdg(results: list[dict], sdg_tags: list[str]) -> list[dict]:
    """
    Python-side SDG tag filter for ChromaDB results.

    Keeps businesses where sdg_tags OR predicted_sdg_tags contains
    at least one of the requested SDG tags (case-insensitive substring match).

    Args:
        results:  List of business dicts (from semantic_search or sql_filter).
        sdg_tags: List of SDG tag names to require.

    Returns:
        Filtered list.
    """
    if not sdg_tags:
        return results

    filtered = []
    for r in results:
        all_tags = (r.get("sdg_tags", "") or "") + "," + (r.get("predicted_sdg_tags", "") or "")
        all_tags_lower = all_tags.lower()
        if any(tag.lower() in all_tags_lower for tag in sdg_tags):
            filtered.append(r)
    return filtered
