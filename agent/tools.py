"""
Search tools for the SearchAgent — PostgreSQL + pgvector backend.

Three tools map to three search modes:
  semantic_search              — pure vector similarity (all businesses)
  semantic_search_from_embedding — same, accepts pre-computed embedding
  sql_filter                   — pure SQL metadata filter, no vector
  hybrid_search                — SQL WHERE + vector ORDER BY (single query)

All filtering (city, sdg_tags, categories, etc.) now happens natively in SQL.
No Python post-processing is needed — pgvector's ILIKE handles substring match.

The shared PGStore and SentenceTransformer are module-level singletons
(loaded once on first call) to avoid re-initialising on every request.
"""

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singletons — loaded once, reused across calls
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_store():
    """Lazy-load PGStore singleton."""
    from db.pg_store import PGStore
    return PGStore()


@lru_cache(maxsize=1)
def _get_encoder():
    """Lazy-load SentenceTransformer singleton (384-dim, same model as stored embeddings)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Tool 1: semantic_search
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    n_results: int = 20,
    where: Optional[dict] = None,   # kept for signature compatibility, unused
) -> list[dict]:
    """
    Search businesses by semantic similarity to a natural language query.

    Encodes the query with all-MiniLM-L6-v2, then searches the pgvector index.
    No SQL filters — returns the globally most similar businesses.

    Args:
        query:     Natural language description to search for.
        n_results: Max number of results.
        where:     Unused (ChromaDB legacy parameter — ignored).

    Returns:
        List of business dicts sorted by cosine similarity (highest first).
    """
    encoder = _get_encoder()
    embedding = encoder.encode([query], normalize_embeddings=True)[0].tolist()
    return _get_store().semantic_search(embedding, n_results=n_results)


# ---------------------------------------------------------------------------
# Tool 2: semantic_search_from_embedding
# ---------------------------------------------------------------------------

def semantic_search_from_embedding(
    embedding: list[float],
    n_results: int = 20,
    where: Optional[dict] = None,   # kept for signature compatibility, unused
) -> list[dict]:
    """
    Search from a pre-computed embedding vector.

    Used by SearchAgent after HyDE averaging — skips re-encoding.

    Args:
        embedding: 384-dim float list (pre-computed, normalised).
        n_results: Max results.
        where:     Unused (ChromaDB legacy parameter — ignored).

    Returns:
        List of business dicts sorted by cosine similarity.
    """
    return _get_store().semantic_search(embedding, n_results=n_results)


# ---------------------------------------------------------------------------
# Tool 3: sql_filter
# ---------------------------------------------------------------------------

def sql_filter(
    filters: dict,
    n_results: int = 100,
) -> list[dict]:
    """
    Filter businesses by metadata conditions — no vector similarity.

    Used when the user provides filters but no company description.
    Results sorted by package_id DESC (premium members first).

    Handles all filter types natively in SQL:
      city, business_type, job_sector, company_size, claimed — exact match
      sdg_tags   — ILIKE on sdg_tags OR predicted_sdg_tags
      categories — ILIKE on categories field

    Args:
        filters:   SearchFilters dict.
        n_results: Max results.

    Returns:
        List of business dicts matching all conditions (similarity=None).
    """
    return _get_store().sql_filter(filters, n_results=n_results)


# ---------------------------------------------------------------------------
# Tool 4: hybrid_search
# ---------------------------------------------------------------------------

def hybrid_search(
    embedding: list[float],
    filters: dict,
    n_results: int = 20,
) -> list[dict]:
    """
    SQL-first hybrid search: filter with SQL WHERE, then rank by vector similarity.

    Single PostgreSQL query — no Python post-processing needed:
        SELECT ..., 1 - (embedding <=> %s) AS similarity
        FROM businesses
        WHERE city = %s AND sdg_tags ILIKE '%X%'
        ORDER BY embedding <=> %s
        LIMIT n

    This is more correct than the previous ChromaDB approach (which did
    vector-first with a large pool, then Python post-filtered), because:
      1. Hard filter constraints are always respected
      2. No pool-size guessing (n×15) — SQL filters first, then rank survivors
      3. sdg_tags and categories use native ILIKE (no Python substring matching)

    Args:
        embedding: Pre-computed 384-dim query vector (averaged HyDE embeddings).
        filters:   SearchFilters dict.
        n_results: Max results.

    Returns:
        List of business dicts satisfying all filters, ranked by cosine similarity.
    """
    return _get_store().hybrid_search(embedding, filters, n_results=n_results)
