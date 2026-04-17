"""
PostgreSQL + pgvector storage layer for SDGZero businesses.

Replaces db/chroma_store.py.

Key improvements over ChromaDB:
  - Single SQL query for hybrid search (SQL WHERE + vector ORDER BY)
  - Native ILIKE for sdg_tags/categories — no Python post-filtering needed
  - ThreadedConnectionPool for concurrent FastAPI requests
  - HNSW index for fast ANN vector search (works on empty table, grows incrementally)

Usage:
    from db.pg_store import PGStore
    store = PGStore()
    store.init_schema()
    results = store.semantic_search(embedding, n_results=10)
"""

import logging
import os
from contextlib import contextmanager
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions — single source of truth
# ---------------------------------------------------------------------------

# All non-id, non-embedding columns in exact table order
_META_COLS = (
    "slug", "name", "url", "scraped_at",
    "street", "city", "region", "country", "zip", "latitude", "longitude",
    "phone", "website", "linkedin", "facebook", "twitter", "tiktok", "instagram", "logo",
    "business_type", "job_sector", "company_size", "package_id", "claimed", "founder_name",
    "categories", "sdg_tags", "sdg_slugs", "predicted_sdg_tags", "membership_tier",
    "rating", "rating_count", "document",
)

_SELECT_COLS = ", ".join(["id"] + list(_META_COLS))

# business_type: form sends short form, DB stores verbose form
_BTYPE_MAP = {
    "B2B": "Business2Business (B2B)",
    "B2C": "Business2Consumer (B2C)",
    "Both": "Both",
    "b2b": "Business2Business (B2B)",
    "b2c": "Business2Consumer (B2C)",
}

# job_sector: display name → DB stored value
_SECTOR_MAP = {
    "Private":        "Private Sector",
    "Public":         "Public Sector",
    "private":        "Private Sector",
    "public":         "Public Sector",
    "Private Sector": "Private Sector",
    "Public Sector":  "Public Sector",
    "Agencies":       "Agencies",
    "agencies":       "Agencies",
}


# ---------------------------------------------------------------------------
# Filter builder — converts SearchFilters dict → SQL WHERE + params
# ---------------------------------------------------------------------------

def build_pg_where(filters: dict) -> tuple[str, list]:
    """
    Convert a SearchFilters dict to a SQL WHERE clause + params list.

    Handles:
      city, business_type, job_sector, company_size, claimed — exact match
      sdg_tags   — ILIKE on sdg_tags OR predicted_sdg_tags (comma-separated)
      categories — ILIKE on categories field (comma-separated)

    Returns:
        (where_sql, params)
        where_sql: "WHERE ..." string, or "" if no conditions
        params:    list of values to bind (positional %s)
    """
    conditions = []
    params = []

    if filters.get("city"):
        conditions.append("city = %s")
        params.append(filters["city"])

    if filters.get("business_type"):
        bt = _BTYPE_MAP.get(filters["business_type"], filters["business_type"])
        conditions.append("business_type = %s")
        params.append(bt)

    if filters.get("job_sector"):
        js = _SECTOR_MAP.get(filters["job_sector"], filters["job_sector"])
        conditions.append("job_sector = %s")
        params.append(js)

    if filters.get("company_size"):
        conditions.append("company_size = %s")
        params.append(filters["company_size"])

    if filters.get("claimed") is True or filters.get("claimed") == "Yes":
        conditions.append("claimed = %s")
        params.append("Yes")

    # SDG tags — ANY of the requested tags must appear in sdg_tags OR predicted_sdg_tags
    sdg_required = filters.get("sdg_tags", [])
    if sdg_required:
        if isinstance(sdg_required, str):
            sdg_required = [sdg_required]
        sdg_conds = []
        for tag in sdg_required:
            sdg_conds.append("(sdg_tags ILIKE %s OR predicted_sdg_tags ILIKE %s)")
            params.extend([f"%{tag}%", f"%{tag}%"])
        if sdg_conds:
            conditions.append(f"({' OR '.join(sdg_conds)})")

    # Categories — ANY of the requested categories must appear
    cat_required = filters.get("categories", "")
    if cat_required:
        cats = [cat_required] if isinstance(cat_required, str) else cat_required
        cat_conds = []
        for cat in cats:
            cat_conds.append("categories ILIKE %s")
            params.append(f"%{cat}%")
        if cat_conds:
            conditions.append(f"({' OR '.join(cat_conds)})")

    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where_sql, params


# ---------------------------------------------------------------------------
# PGStore
# ---------------------------------------------------------------------------

class PGStore:
    """
    PostgreSQL + pgvector storage for SDGZero businesses.

    Singleton usage via tools._get_store() — do not instantiate directly elsewhere.
    """

    _pool = None

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv(
            "DATABASE_URL",
            "postgresql://sdgzero:sdgzero@localhost:5432/sdgzero",
        )
        self._init_pool()

    def _init_pool(self) -> None:
        import psycopg2.pool
        if PGStore._pool is None:
            PGStore._pool = psycopg2.pool.ThreadedConnectionPool(1, 5, self.dsn)
            logger.info(f"PGStore: connection pool created → {self.dsn.split('@')[-1]}")

    @contextmanager
    def _cursor(self, dict_rows: bool = True):
        """
        Context manager: borrow a connection from the pool, yield a cursor,
        commit on success / rollback on error, return connection to pool.

        Args:
            dict_rows: If True, use RealDictCursor (rows as dicts).
                       If False, use plain cursor (for upsert with execute_values).
        """
        conn = PGStore._pool.getconn()
        try:
            try:
                from pgvector.psycopg2 import register_vector
                register_vector(conn)
            except Exception:
                pass  # vector extension not yet created (first init_schema call)
            if dict_rows:
                from psycopg2.extras import RealDictCursor
                cur = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cur = conn.cursor()
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            PGStore._pool.putconn(conn)

    # -----------------------------------------------------------------------
    # Schema
    # -----------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create the businesses table and all indexes if they don't exist."""
        with self._cursor(dict_rows=False) as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS businesses (
                    id                 INTEGER PRIMARY KEY,
                    slug               TEXT NOT NULL DEFAULT '',
                    name               TEXT NOT NULL DEFAULT '',
                    url                TEXT DEFAULT '',
                    scraped_at         TEXT DEFAULT '',
                    street             TEXT DEFAULT '',
                    city               TEXT DEFAULT '',
                    region             TEXT DEFAULT '',
                    country            TEXT DEFAULT '',
                    zip                TEXT DEFAULT '',
                    latitude           FLOAT DEFAULT 0,
                    longitude          FLOAT DEFAULT 0,
                    phone              TEXT DEFAULT '',
                    website            TEXT DEFAULT '',
                    linkedin           TEXT DEFAULT '',
                    facebook           TEXT DEFAULT '',
                    twitter            TEXT DEFAULT '',
                    tiktok             TEXT DEFAULT '',
                    instagram          TEXT DEFAULT '',
                    logo               TEXT DEFAULT '',
                    business_type      TEXT DEFAULT '',
                    job_sector         TEXT DEFAULT '',
                    company_size       TEXT DEFAULT '',
                    package_id         INTEGER DEFAULT 1,
                    claimed            TEXT DEFAULT '',
                    founder_name       TEXT DEFAULT '',
                    categories         TEXT DEFAULT '',
                    sdg_tags           TEXT DEFAULT '',
                    sdg_slugs          TEXT DEFAULT '',
                    predicted_sdg_tags TEXT DEFAULT '',
                    membership_tier    TEXT DEFAULT '',
                    rating             FLOAT DEFAULT 0,
                    rating_count       INTEGER DEFAULT 0,
                    document           TEXT DEFAULT '',
                    embedding          vector(384),
                    inserted_at        TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # HNSW vector index — works on empty table, grows incrementally
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_businesses_embedding
                ON businesses USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            # B-tree indexes for common filter columns
            for col in ("city", "business_type", "job_sector", "company_size", "claimed", "package_id"):
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_businesses_{col}
                    ON businesses ({col})
                """)

        logger.info("PGStore: schema initialised")

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def upsert_batch(self, rows: list[dict], batch_size: int = 100) -> None:
        """
        Upsert a list of business dicts into PostgreSQL.

        Each dict must contain all _META_COLS fields plus 'id' and 'embedding'.
        Uses ON CONFLICT (id) DO UPDATE so re-running the migration is safe.

        Args:
            rows:       List of business dicts (from migration script).
            batch_size: Number of rows per INSERT batch.
        """
        from psycopg2.extras import execute_values

        all_cols = ("id",) + _META_COLS + ("embedding",)
        update_cols = _META_COLS + ("embedding",)
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        col_list = ", ".join(all_cols)

        total = len(rows)
        with self._cursor(dict_rows=False) as cur:
            for start in range(0, total, batch_size):
                batch = rows[start : start + batch_size]
                values = []
                for r in batch:
                    emb = r["embedding"]
                    if not isinstance(emb, np.ndarray):
                        emb = np.array(emb, dtype=np.float32)
                    row_vals = tuple(
                        emb if col == "embedding" else r.get(col, "")
                        for col in all_cols
                    )
                    values.append(row_vals)

                execute_values(
                    cur,
                    f"""
                    INSERT INTO businesses ({col_list})
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET {set_clause}
                    """,
                    values,
                )
                print(f"  Upserted {min(start + batch_size, total)}/{total}")

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def semantic_search(
        self,
        query_embedding: list,
        n_results: int = 20,
    ) -> list[dict]:
        """
        Pure vector similarity search — no SQL filters.

        Used for:
          - Simple fallback (no user filters)
          - Padding when filtered results are too few
          - CRAG retry global escape
        """
        emb = np.array(query_embedding, dtype=np.float32)
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT {_SELECT_COLS},
                       1 - (embedding <=> %s::vector) AS similarity
                FROM businesses
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (emb, emb, n_results),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def hybrid_search(
        self,
        query_embedding: list,
        filters: dict,
        n_results: int = 20,
    ) -> list[dict]:
        """
        SQL-first hybrid search: filter candidates with SQL WHERE, then rank by vector.

        Replaces the old ChromaDB approach of:
          vector search (large pool) → Python post-filter sdg/categories → truncate

        Now a single SQL query handles everything:
          WHERE city = %s AND sdg_tags ILIKE '%X%' ORDER BY embedding <=> %s LIMIT n

        Args:
            query_embedding: Pre-computed 384-dim vector.
            filters:         SearchFilters dict (city, business_type, sdg_tags, ...).
            n_results:       Max results.
        """
        emb = np.array(query_embedding, dtype=np.float32)
        where_sql, params = build_pg_where(filters)

        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT {_SELECT_COLS},
                       1 - (embedding <=> %s::vector) AS similarity
                FROM businesses
                {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (emb, *params, emb, n_results),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def sql_filter(
        self,
        filters: dict,
        n_results: int = 100,
    ) -> list[dict]:
        """
        Pure SQL filter — no vector search.

        Used when user provides filters but no company description.
        Results sorted by package_id DESC (premium members first).

        Args:
            filters:   SearchFilters dict.
            n_results: Max results.
        """
        where_sql, params = build_pg_where(filters)

        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT {_SELECT_COLS}, NULL::float AS similarity
                FROM businesses
                {where_sql}
                ORDER BY package_id DESC
                LIMIT %s
                """,
                (*params, n_results),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of businesses in the database."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM businesses")
            return cur.fetchone()["n"]

    def build_schema_data(self) -> dict:
        """
        Build the filter schema dict from PostgreSQL — replaces ChromaDB metadata scan.

        Returns the same shape as agent/schema_cache.py expects:
            {"city": [...], "business_type": [...], ...}
        """
        _BTYPE_DISPLAY = {
            "Business2Business (B2B)": "B2B",
            "Business2Consumer (B2C)": "B2C",
            "Both": "Both",
        }
        _SECTOR_DISPLAY = {
            "Private Sector": "Private",
            "Public Sector": "Public",
            "Agencies": "Agencies",
        }

        with self._cursor() as cur:
            # Scalar distinct fields
            cur.execute("SELECT DISTINCT city FROM businesses WHERE city != '' ORDER BY city")
            cities = [r["city"] for r in cur.fetchall()]

            cur.execute("SELECT DISTINCT business_type FROM businesses WHERE business_type != '' ORDER BY business_type")
            business_types = [_BTYPE_DISPLAY.get(r["business_type"], r["business_type"]) for r in cur.fetchall()]

            cur.execute("SELECT DISTINCT job_sector FROM businesses WHERE job_sector != '' ORDER BY job_sector")
            job_sectors = [_SECTOR_DISPLAY.get(r["job_sector"], r["job_sector"]) for r in cur.fetchall()]

            cur.execute("SELECT DISTINCT company_size FROM businesses WHERE company_size != '' ORDER BY company_size")
            company_sizes = [r["company_size"] for r in cur.fetchall()]

            # Comma-separated fields — unnest with PostgreSQL
            cur.execute("""
                SELECT DISTINCT trim(cat) AS cat
                FROM businesses,
                     unnest(string_to_array(categories, ',')) AS cat
                WHERE categories != '' AND trim(cat) != ''
                ORDER BY cat
            """)
            categories = [r["cat"] for r in cur.fetchall()]

            cur.execute("""
                SELECT DISTINCT trim(tag) AS tag
                FROM businesses,
                     unnest(string_to_array(sdg_tags || ',' || predicted_sdg_tags, ',')) AS tag
                WHERE trim(tag) != ''
                ORDER BY tag
            """)
            sdg_tags = [r["tag"] for r in cur.fetchall()]

        logger.info(
            f"PGStore.build_schema_data — cities={len(cities)}, "
            f"categories={len(categories)}, sdg_tags={len(sdg_tags)}"
        )
        return {
            "city": cities,
            "business_type": business_types,
            "job_sector": job_sectors,
            "company_size": company_sizes,
            "categories": categories,
            "sdg_tags": sdg_tags,
        }
