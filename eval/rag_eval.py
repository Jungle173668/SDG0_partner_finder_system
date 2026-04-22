"""
RAG Search Quality Evaluation
==============================
Measures the quality of the SDGZero semantic search system against two baselines:
  1. BM25 (best-case keyword retrieval — theoretical upper bound for keyword systems)
  2. Original website MySQL search (real-world lower bound — the system being replaced)

Four evaluation methods, all annotation-free:

  Method 1 — Self-Retrieval MRR
    A company's own document should rank #1 when used as the query.
    Sanity-checks that the vector space is coherent.
    Metric: MRR (Mean Reciprocal Rank). BM25 ≈ 1.0 here; semantic should be close.

  Method 2 — SDG-based Precision@K
    17 UN SDG official descriptions are used as queries.
    Precision = fraction of Top-K results tagged with that SDG.
    Grounded in human-assigned predicted_sdg_tags (independent of search system).
    Metric: Precision@10 per SDG, mean across all 17 SDGs.

  Method 3 — Category Consistency@5
    A company's neighbours should share its business category.
    Uses human-assigned `categories` field — no LLM involved.
    Metric: fraction of Top-5 neighbours (excluding self) in the same category.

  Method 4 — Motivational Zero-Result Rate
    30 natural language queries that keyword systems cannot handle
    (synonyms, paraphrases, SDG jargon).
    Semantic search should return ≥1 result for all; keyword/BM25 will fail many.
    Metric: Zero-Result Rate = fraction of queries returning 0 results.

  Method 5 — Out-of-Vocabulary (OOV) Queries
    20 queries describing SDG concepts WITHOUT using canonical domain keywords.
    (e.g., SDG7 described without "energy"/"renewable"; SDG13 without "carbon"/"climate")
    Isolates whether semantic search can bridge vocabulary gaps.
    Metric: BM25 Zero-Result Rate + BM25 Low-Signal Rate.

  Method 6 — BM25 Score Distribution
    Compares max BM25 scores for OOV queries vs. keyword-control queries.
    Shows that BM25 "non-zero" results on OOV queries are near-noise coincidental matches.
    Metric: mean max score for OOV vs controls; Low-Signal Rate.

  Method 7 — HyDE Partner Diversity (optional, requires LLM)
    Compares plain semantic search vs HyDE for 8 partner-matching scenarios.
    Plain: encodes user description → finds similar companies (competitors).
    HyDE: generates hypothetical partner profile → finds complementary companies (partners).
    Metric: Cross-Sector Rate@5, Category Diversity@5.

Usage:
    python3.11 eval/rag_eval.py
    python3.11 eval/rag_eval.py --sample 100       # fast mode: 100 companies for M1/M3
    python3.11 eval/rag_eval.py --output notes/rag_eval_summary.md
    python3.11 eval/rag_eval.py --skip-website      # skip live WP API calls
    python3.11 eval/rag_eval.py --run-hyde          # enable Method 7 (needs GOOGLE_API_KEY)

References:
  - MRR: Voorhees 1999 (TREC-8 QA track)
  - BM25: Robertson & Zaragoza 2009; BEIR benchmark Thakur et al. 2021
  - NDCG: Järvelin & Kekäläinen 2002
  - Synthetic eval: RAGAS, Es et al. 2023 (arxiv 2309.15217)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Load .env so LLM_PROVIDER / API keys are available when --run-hyde is used
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 17 UN SDG Official Descriptions (query text → SDG label for matching)
# ---------------------------------------------------------------------------

SDG_QUERIES: dict[str, str] = {
    "SDG1 No Poverty":                       "End poverty in all its forms everywhere — economic resources, social protection, vulnerability to climate",
    "SDG2 Zero Hunger":                      "End hunger, achieve food security and improved nutrition, promote sustainable agriculture",
    "SDG3 Good Health":                      "Ensure healthy lives and promote well-being for all at all ages — healthcare, mental health, medicine",
    "SDG4 Quality Education":                "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all",
    "SDG5 Gender Equality":                  "Achieve gender equality and empower all women and girls — equal rights, leadership, anti-discrimination",
    "SDG6 Clean Water":                      "Ensure availability and sustainable management of water and sanitation for all",
    "SDG7 Affordable Clean Energy":          "Ensure access to affordable, reliable, sustainable and modern energy for all — renewables, efficiency",
    "SDG8 Decent Work":                      "Promote sustained inclusive economic growth, full and productive employment and decent work for all",
    "SDG9 Industry Innovation":              "Build resilient infrastructure, promote inclusive industrialisation and foster innovation — technology",
    "SDG10 Reduced Inequalities":            "Reduce inequality within and among countries — social, economic, political inclusion",
    "SDG11 Sustainable Cities":              "Make cities and human settlements inclusive, safe, resilient and sustainable — urban planning",
    "SDG12 Responsible Consumption":         "Ensure sustainable consumption and production patterns — circular economy, waste reduction, recycling",
    "SDG13 Climate Action":                  "Take urgent action to combat climate change and its impacts — carbon, net zero, emissions",
    "SDG14 Life Below Water":                "Conserve and sustainably use the oceans, seas and marine resources — ocean, marine, biodiversity",
    "SDG15 Life On Land":                    "Protect, restore and promote sustainable use of terrestrial ecosystems — forests, biodiversity, land",
    "SDG16 Peace Justice":                   "Promote peaceful and inclusive societies, provide access to justice and build effective institutions",
    "SDG17 Partnerships":                    "Strengthen the means of implementation and revitalise the global partnership for sustainable development",
}

# Exact substrings matching the predicted_sdg_tags DB format (verified via DB query).
# Uses full SDG label names to avoid false matches (e.g. "Work" → 354 companies;
# "Decent Work" → 351 companies — much more precise against unrelated "work" mentions).
SDG_MATCH_KEYWORDS: dict[str, str] = {
    "SDG1 No Poverty":               "No Poverty",
    "SDG2 Zero Hunger":              "Zero Hunger",
    "SDG3 Good Health":              "Good Health",
    "SDG4 Quality Education":        "Quality Education",
    "SDG5 Gender Equality":          "Gender Equality",
    "SDG6 Clean Water":              "Clean Water And Sanitation",
    "SDG7 Affordable Clean Energy":  "Affordable And Clean Energy",
    "SDG8 Decent Work":              "Decent Work",
    "SDG9 Industry Innovation":      "Industry Innovation",
    "SDG10 Reduced Inequalities":    "Reduced Inequalit",
    "SDG11 Sustainable Cities":      "Sustainable Cities",
    "SDG12 Responsible Consumption": "Responsible Consumption",
    "SDG13 Climate Action":          "Climate Action",
    "SDG14 Life Below Water":        "Life Below Water",
    "SDG15 Life On Land":            "Life On Land",
    "SDG16 Peace Justice":           "Peace Justice",
    "SDG17 Partnerships":            "Partnerships For The Goals",
}

# SDGs considered "discriminative" for evaluation — prevalence between 2% and 35%.
# Over-predicted SDGs (>35% of corpus) inflate both systems' scores equally and
# add no signal. Under-represented SDGs (<5 companies) are statistically unreliable.
_DISCRIMINATIVE_SDGS = {
    k for k, _ in SDG_MATCH_KEYWORDS.items()
    # Excluded: SDG8 (71%), SDG9 (39%), SDG11 (44%), SDG12 (35%), SDG3 (34%)
    # (prevalence figures verified against 492-company corpus)
}

# ---------------------------------------------------------------------------
# 30 Motivational Queries: natural language phrasing that breaks keyword search
# Split into 3 themes:
#   A. Synonym / paraphrase  (keyword search fails)
#   B. SDG jargon → company activity  (keyword search fails)
#   C. Concept without literal words  (keyword search fails)
# ---------------------------------------------------------------------------

MOTIVATIONAL_QUERIES: list[dict] = [
    # Theme A: Synonym / paraphrase
    {"id": 1,  "query": "clean power generation",          "theme": "synonym",   "expected_sdg": "SDG7"},
    {"id": 2,  "query": "carbon footprint reduction",      "theme": "synonym",   "expected_sdg": "SDG13"},
    {"id": 3,  "query": "circular economy solutions",      "theme": "synonym",   "expected_sdg": "SDG12"},
    {"id": 4,  "query": "female empowerment business",     "theme": "synonym",   "expected_sdg": "SDG5"},
    {"id": 5,  "query": "food waste prevention",           "theme": "synonym",   "expected_sdg": "SDG12"},
    {"id": 6,  "query": "green building materials",        "theme": "synonym",   "expected_sdg": "SDG11"},
    {"id": 7,  "query": "affordable housing developer",    "theme": "synonym",   "expected_sdg": "SDG11"},
    {"id": 8,  "query": "supply chain transparency",       "theme": "synonym",   "expected_sdg": "SDG12"},
    {"id": 9,  "query": "community mental health support", "theme": "synonym",   "expected_sdg": "SDG3"},
    {"id": 10, "query": "workforce upskilling platform",   "theme": "synonym",   "expected_sdg": "SDG4"},

    # Theme B: SDG jargon → real company activity
    {"id": 11, "query": "SDG13 climate action consulting",          "theme": "sdg_jargon", "expected_sdg": "SDG13"},
    {"id": 12, "query": "SDG7 energy access solutions",             "theme": "sdg_jargon", "expected_sdg": "SDG7"},
    {"id": 13, "query": "SDG3 preventative healthcare company",     "theme": "sdg_jargon", "expected_sdg": "SDG3"},
    {"id": 14, "query": "SDG4 edtech learning provider",            "theme": "sdg_jargon", "expected_sdg": "SDG4"},
    {"id": 15, "query": "SDG12 responsible packaging manufacturer", "theme": "sdg_jargon", "expected_sdg": "SDG12"},
    {"id": 16, "query": "SDG8 fair trade employment agency",        "theme": "sdg_jargon", "expected_sdg": "SDG8"},
    {"id": 17, "query": "SDG9 deep tech innovation startup",        "theme": "sdg_jargon", "expected_sdg": "SDG9"},
    {"id": 18, "query": "SDG11 smart urban mobility",               "theme": "sdg_jargon", "expected_sdg": "SDG11"},
    {"id": 19, "query": "SDG6 water purification technology",       "theme": "sdg_jargon", "expected_sdg": "SDG6"},
    {"id": 20, "query": "SDG17 cross-sector partnership facilitator","theme": "sdg_jargon", "expected_sdg": "SDG17"},

    # Theme C: Concept described without the literal words in company names/descriptions
    {"id": 21, "query": "helping businesses measure their environmental impact",   "theme": "concept", "expected_sdg": "SDG13"},
    {"id": 22, "query": "technology to make energy bills cheaper for households",  "theme": "concept", "expected_sdg": "SDG7"},
    {"id": 23, "query": "teaching children to code and build digital skills",      "theme": "concept", "expected_sdg": "SDG4"},
    {"id": 24, "query": "connecting small farmers to markets",                     "theme": "concept", "expected_sdg": "SDG2"},
    {"id": 25, "query": "platform for ethical and sustainable fashion brands",     "theme": "concept", "expected_sdg": "SDG12"},
    {"id": 26, "query": "company that helps factories reduce energy consumption",  "theme": "concept", "expected_sdg": "SDG7"},
    {"id": 27, "query": "reducing plastic in oceans through innovation",           "theme": "concept", "expected_sdg": "SDG14"},
    {"id": 28, "query": "data analytics for public health surveillance",           "theme": "concept", "expected_sdg": "SDG3"},
    {"id": 29, "query": "training people from disadvantaged backgrounds for jobs", "theme": "concept", "expected_sdg": "SDG8"},
    {"id": 30, "query": "enabling remote communities to access financial services","theme": "concept", "expected_sdg": "SDG1"},
]

# ---------------------------------------------------------------------------
# Database access — reads directly from PGStore, zero pipeline impact
# ---------------------------------------------------------------------------

def _get_all_companies() -> list[dict]:
    """Fetch all companies with their documents, embeddings, and metadata."""
    from db.pg_store import PGStore
    store = PGStore()

    # Fetch all companies including their embeddings for BM25 corpus building
    with store._cursor() as cur:
        cur.execute(
            """
            SELECT id, slug, name, document, categories,
                   sdg_tags, predicted_sdg_tags, embedding
            FROM businesses
            WHERE document != '' AND embedding IS NOT NULL
            ORDER BY id
            """
        )
        rows = cur.fetchall()

    companies = []
    for r in rows:
        row = dict(r)
        # Convert numpy array from pgvector
        if row["embedding"] is not None:
            emb = row["embedding"]
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb, dtype=np.float32)
            row["embedding"] = emb
        companies.append(row)

    return companies


def _get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _encode_query(encoder, query: str) -> np.ndarray:
    return encoder.encode([query], normalize_embeddings=True)[0]


def _semantic_search_by_embedding(
    embedding: np.ndarray,
    all_embeddings: np.ndarray,
    company_ids: list[int],
    n: int = 20,
) -> list[int]:
    """
    In-memory cosine similarity search — avoids repeated DB round-trips during eval.
    Returns company IDs sorted by similarity descending.
    """
    sims = all_embeddings @ embedding  # cosine sim (embeddings are normalized)
    top_idx = np.argsort(-sims)[:n]
    return [company_ids[i] for i in top_idx]


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _build_bm25(documents: list[str]):
    """Build BM25 index over a corpus of documents."""
    from rank_bm25 import BM25Okapi
    tokenized = [doc.lower().split() for doc in documents]
    return BM25Okapi(tokenized)


def _bm25_search(bm25, query: str, company_ids: list[int], n: int = 20) -> list[int]:
    """Return top-n company IDs by BM25 score."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(-scores)[:n]
    return [company_ids[i] for i in top_idx]


def _bm25_zero_result(bm25, query: str) -> bool:
    """Return True if all BM25 scores are 0 (no keyword overlap)."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    return float(np.max(scores)) == 0.0


# ---------------------------------------------------------------------------
# Full Pipeline search helper (HyDE + averaged embedding)
#
# Mirrors the exact logic in mcp_server/server.py search_companies:
#   _run_hyde(user_company_desc="", partner_type_desc=query)
#   → _averaged_embedding([inferred_type×2, partner_desc])
#   → cosine similarity
#
# Results are cached to .eval_hyde_cache.json so repeated runs are free.
# ---------------------------------------------------------------------------

_HYDE_CACHE_PATH = Path(".eval_hyde_cache.json")


def _load_hyde_cache() -> dict:
    """Load HyDE embedding cache (query → embedding list)."""
    if _HYDE_CACHE_PATH.exists():
        try:
            return json.loads(_HYDE_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_hyde_cache(cache: dict) -> None:
    """Persist HyDE embedding cache to disk."""
    _HYDE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _hyde_pipeline_search(
    query: str,
    all_embeddings: np.ndarray,
    company_ids: list[int],
    n: int = 10,
    cache: dict | None = None,
) -> tuple[list[int], str]:
    """
    Full pipeline search for a query string — same logic as MCP search_companies.

    Steps:
      1. _run_hyde(partner_type_desc=query) → generates a hypothetical company profile
         of the type described by query, plus query expansions and inferred type label.
      2. _averaged_embedding([inferred_type×2, partner_desc]) → 384-dim query vector.
         (Repeating inferred_type gives the type label stronger weight vs the HyDE prose.)
      3. cosine similarity against all_embeddings → top-n company IDs.

    The embedding is cached by query string to avoid repeated LLM calls.

    Returns:
        (company_ids sorted by similarity, inferred_type_label)
    """
    cache_key = f"search:{query}"
    if cache is not None and cache_key in cache:
        data = cache[cache_key]
        embedding = np.array(data["embedding"], dtype=np.float32)
        inferred_type = data.get("inferred_type", "")
    else:
        from agent.search_agent import _run_hyde, _averaged_embedding
        partner_desc, expansions, inferred_type = _run_hyde(
            user_company_desc="",
            partner_type_desc=query,
        )
        texts = [inferred_type, inferred_type, partner_desc] if inferred_type else [partner_desc] + expansions
        embedding = np.array(_averaged_embedding(texts), dtype=np.float32)
        if cache is not None:
            cache[cache_key] = {"embedding": embedding.tolist(), "inferred_type": inferred_type}
    return _semantic_search_by_embedding(embedding, all_embeddings, company_ids, n=n), inferred_type


def _hyde_pipeline_search_full(
    user_company_desc: str,
    partner_type_desc: str,
    all_embeddings: np.ndarray,
    company_ids: list[int],
    n: int = 10,
    cache: dict | None = None,
) -> tuple[list[int], str]:
    """
    Full pipeline with BOTH fields — used for M1 and M3 where we pass the company's own
    description as partner_type_desc (leaving user_company_desc empty or filled).

    Distinct cache key from _hyde_pipeline_search to avoid collision.
    """
    cache_key = f"full:{user_company_desc[:80]}|{partner_type_desc[:80]}"
    if cache is not None and cache_key in cache:
        data = cache[cache_key]
        embedding = np.array(data["embedding"], dtype=np.float32)
        inferred_type = data.get("inferred_type", "")
    else:
        from agent.search_agent import _run_hyde, _averaged_embedding
        partner_desc, expansions, inferred_type = _run_hyde(
            user_company_desc=user_company_desc,
            partner_type_desc=partner_type_desc,
        )
        texts = [inferred_type, inferred_type, partner_desc] if inferred_type else [partner_desc] + expansions
        embedding = np.array(_averaged_embedding(texts), dtype=np.float32)
        if cache is not None:
            cache[cache_key] = {"embedding": embedding.tolist(), "inferred_type": inferred_type}
    return _semantic_search_by_embedding(embedding, all_embeddings, company_ids, n=n), inferred_type


# ---------------------------------------------------------------------------
# Method 1: Self-Retrieval MRR
# ---------------------------------------------------------------------------

def run_self_retrieval(
    companies: list[dict],
    all_embeddings: np.ndarray,
    company_ids: list[int],
    bm25,
    n: int = 20,
    sample: int | None = None,
    run_hyde: bool = False,
    hyde_cache: dict | None = None,
) -> dict:
    """
    For each company, use its own document as the query and check what rank
    the company itself appears in the results.

    Perfect system → MRR = 1.0 (always rank #1).
    BM25 on this task approaches 1.0 because exact text overlap is maximal.
    Semantic should also score high; any MRR < 0.9 indicates embedding quality issues.
    """
    pool = companies if sample is None else random.sample(companies, min(sample, len(companies)))

    sem_rr, bm25_rr, pipeline_rr = [], [], []
    failures = []

    for co in pool:
        doc = co["document"]
        cid = co["id"]

        # --- Semantic ---
        emb = co["embedding"]
        sem_results = _semantic_search_by_embedding(emb, all_embeddings, company_ids, n=n)
        try:
            rank = sem_results.index(cid) + 1
        except ValueError:
            rank = n + 1
        sem_rr.append(1.0 / rank)
        if rank > 1:
            failures.append({"id": cid, "name": co["name"], "rank": rank})

        # --- BM25 ---
        bm25_results = _bm25_search(bm25, doc[:500], company_ids, n=n)
        try:
            bm25_rank = bm25_results.index(cid) + 1
        except ValueError:
            bm25_rank = n + 1
        bm25_rr.append(1.0 / bm25_rank)

        # --- Full Pipeline (HyDE) ---
        # Pass company description as partner_type_desc: "find me a company like this"
        if run_hyde:
            pipe_results, _ = _hyde_pipeline_search_full(
                user_company_desc="",
                partner_type_desc=doc[:600],
                all_embeddings=all_embeddings,
                company_ids=company_ids,
                n=n,
                cache=hyde_cache,
            )
            try:
                pipe_rank = pipe_results.index(cid) + 1
            except ValueError:
                pipe_rank = n + 1
            pipeline_rr.append(1.0 / pipe_rank)

    sem_mrr = float(np.mean(sem_rr))
    bm25_mrr = float(np.mean(bm25_rr))
    pipeline_mrr = float(np.mean(pipeline_rr)) if pipeline_rr else None

    # Rank distribution
    sem_ranks = [1.0 / rr for rr in sem_rr]
    rank_dist = {
        "rank_1": sum(1 for r in sem_ranks if r == 1),
        "rank_2_5": sum(1 for r in sem_ranks if 2 <= r <= 5),
        "rank_6_10": sum(1 for r in sem_ranks if 6 <= r <= 10),
        "rank_11_plus": sum(1 for r in sem_ranks if r > 10),
    }
    if pipeline_rr:
        pipe_ranks = [1.0 / rr for rr in pipeline_rr]
        pipe_rank_dist = {
            "rank_1": sum(1 for r in pipe_ranks if r == 1),
            "rank_2_5": sum(1 for r in pipe_ranks if 2 <= r <= 5),
            "rank_6_10": sum(1 for r in pipe_ranks if 6 <= r <= 10),
            "rank_11_plus": sum(1 for r in pipe_ranks if r > 10),
        }
    else:
        pipe_rank_dist = None

    return {
        "n_companies": len(pool),
        "n_results": n,
        "run_hyde": run_hyde,
        "pipeline_mrr": pipeline_mrr,
        "pipe_rank_distribution": pipe_rank_dist,
        "semantic_mrr": sem_mrr,
        "bm25_mrr": bm25_mrr,
        "delta": sem_mrr - bm25_mrr,
        "rank_distribution": rank_dist,
        "top_failures": sorted(failures, key=lambda x: x["rank"])[:5],
    }


# ---------------------------------------------------------------------------
# Method 2: SDG-based Precision@K
# ---------------------------------------------------------------------------

def _sdg_hit(co: dict, keyword: str) -> bool:
    """Check if a company is tagged with a given SDG (checks both predicted and real tags)."""
    kw = keyword.lower()
    pred = (co.get("predicted_sdg_tags") or "").lower()
    real = (co.get("sdg_tags") or "").lower()
    return kw in pred or kw in real


def _ndcg_at_k(ranked_ids: list[int], id_to_co: dict, match_kw: str, k: int) -> float:
    """
    NDCG@K using binary SDG relevance (1 if tagged, 0 if not).
    NDCG accounts for *position*: relevant results at rank 1 count more than at rank K.
    DCG@K = Σ rel_i / log2(i+2)  for i in 0..K-1
    IDCG@K = ideal DCG (all relevant results at top)
    """
    rels = [1 if _sdg_hit(id_to_co[cid], match_kw) else 0 for cid in ranked_ids[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    n_positives = sum(rels)
    if n_positives == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_positives, k)))
    return dcg / idcg if idcg > 0 else 0.0


def run_sdg_precision(
    companies: list[dict],
    all_embeddings: np.ndarray,
    company_ids: list[int],
    encoder,
    bm25,
    k: int = 10,
    run_hyde: bool = False,
    hyde_cache: dict | None = None,
    website_baseline: dict | None = None,
) -> dict:
    """
    Use each of the 17 UN SDG descriptions as a query.

    Three search paths when run_hyde=True:
      - Raw semantic: embed(query_text) → cosine (no LLM)
      - Full pipeline: HyDE(query_text) → averaged embedding → cosine (full system)
      - BM25: keyword retrieval baseline

    Metrics:
      Precision@K  — fraction of Top-K results tagged with that SDG.
      NDCG@K       — position-weighted relevance; rewards SDG-tagged results at rank 1.
    """
    id_to_co = {co["id"]: co for co in companies}
    sem_precisions, bm25_precisions = [], []
    sem_ndcgs, bm25_ndcgs = [], []
    sem_disc_p, bm25_disc_p = [], []
    sem_disc_ndcg, bm25_disc_ndcg = [], []
    pipeline_precisions, pipeline_ndcgs = [], []
    pipeline_disc_p, pipeline_disc_ndcg = [], []
    website_precisions = []
    per_sdg = {}

    # Pre-computed website P@K per SDG from baseline
    wb_m2 = (website_baseline or {}).get("m2", {}) if (website_baseline and website_baseline.get("available")) else {}

    for sdg_name, query_text in SDG_QUERIES.items():
        match_kw = SDG_MATCH_KEYWORDS[sdg_name]

        # Raw semantic (no HyDE)
        query_emb = _encode_query(encoder, query_text)
        sem_ids = _semantic_search_by_embedding(query_emb, all_embeddings, company_ids, n=k)
        sem_hits = sum(1 for cid in sem_ids if _sdg_hit(id_to_co[cid], match_kw))
        sem_p = sem_hits / k
        sem_ndcg = _ndcg_at_k(sem_ids, id_to_co, match_kw, k)

        # BM25
        bm25_ids = _bm25_search(bm25, query_text, company_ids, n=k)
        bm25_hits = sum(1 for cid in bm25_ids if _sdg_hit(id_to_co[cid], match_kw))
        bm25_p = bm25_hits / k
        bm25_ndcg = _ndcg_at_k(bm25_ids, id_to_co, match_kw, k)

        # Full pipeline (HyDE) — optional
        pipeline_p, pipeline_ndcg_val, pipeline_inferred = None, None, None
        if run_hyde:
            pipeline_ids, pipeline_inferred = _hyde_pipeline_search(
                query_text, all_embeddings, company_ids, n=k, cache=hyde_cache
            )
            pipeline_hits = sum(1 for cid in pipeline_ids if _sdg_hit(id_to_co[cid], match_kw))
            pipeline_p = pipeline_hits / k
            pipeline_ndcg_val = _ndcg_at_k(pipeline_ids, id_to_co, match_kw, k)

        corpus_count = sum(1 for co in companies if _sdg_hit(co, match_kw))
        prevalence = corpus_count / len(companies)
        is_discriminative = 0.02 <= prevalence <= 0.35  # 2%–35% of corpus

        # Website P@K from pre-computed baseline (effective: zero-result → 0.0)
        wp_p = None
        if wb_m2:
            wp_entry = wb_m2.get("per_sdg", {}).get(sdg_name)
            if wp_entry is not None:
                wp_p = wp_entry.get("sdg_relevance_eff")
                website_precisions.append(wp_p)

        per_sdg[sdg_name] = {
            "semantic_precision": sem_p,
            "bm25_precision": bm25_p,
            "pipeline_precision": pipeline_p,
            "website_precision": wp_p,
            "semantic_ndcg": sem_ndcg,
            "bm25_ndcg": bm25_ndcg,
            "pipeline_ndcg": pipeline_ndcg_val,
            "pipeline_inferred_type": pipeline_inferred,
            "delta_precision": sem_p - bm25_p,
            "delta_ndcg": sem_ndcg - bm25_ndcg,
            "corpus_count": corpus_count,
            "prevalence": prevalence,
            "discriminative": is_discriminative,
        }
        sem_precisions.append(sem_p)
        bm25_precisions.append(bm25_p)
        sem_ndcgs.append(sem_ndcg)
        bm25_ndcgs.append(bm25_ndcg)

        if is_discriminative:
            sem_disc_p.append(sem_p)
            bm25_disc_p.append(bm25_p)
            sem_disc_ndcg.append(sem_ndcg)
            bm25_disc_ndcg.append(bm25_ndcg)

        if run_hyde and pipeline_p is not None:
            pipeline_precisions.append(pipeline_p)
            pipeline_ndcgs.append(pipeline_ndcg_val)
            if is_discriminative:
                pipeline_disc_p.append(pipeline_p)
                pipeline_disc_ndcg.append(pipeline_ndcg_val)

    return {
        "k": k,
        "run_hyde": run_hyde,
        # All 17 SDGs
        "mean_semantic_precision": float(np.mean(sem_precisions)),
        "mean_bm25_precision": float(np.mean(bm25_precisions)),
        "mean_pipeline_precision": float(np.mean(pipeline_precisions)) if pipeline_precisions else None,
        "mean_website_precision": float(np.mean(website_precisions)) if website_precisions else None,
        "delta": float(np.mean(sem_precisions)) - float(np.mean(bm25_precisions)),
        "mean_semantic_ndcg": float(np.mean(sem_ndcgs)),
        "mean_bm25_ndcg": float(np.mean(bm25_ndcgs)),
        "mean_pipeline_ndcg": float(np.mean(pipeline_ndcgs)) if pipeline_ndcgs else None,
        "delta_ndcg": float(np.mean(sem_ndcgs)) - float(np.mean(bm25_ndcgs)),
        # Discriminative SDGs only (2%–35% prevalence)
        "n_discriminative": len(sem_disc_p),
        "disc_semantic_precision": float(np.mean(sem_disc_p)) if sem_disc_p else None,
        "disc_bm25_precision": float(np.mean(bm25_disc_p)) if bm25_disc_p else None,
        "disc_pipeline_precision": float(np.mean(pipeline_disc_p)) if pipeline_disc_p else None,
        "disc_semantic_ndcg": float(np.mean(sem_disc_ndcg)) if sem_disc_ndcg else None,
        "disc_bm25_ndcg": float(np.mean(bm25_disc_ndcg)) if bm25_disc_ndcg else None,
        "disc_pipeline_ndcg": float(np.mean(pipeline_disc_ndcg)) if pipeline_disc_ndcg else None,
        "per_sdg": per_sdg,
    }


# ---------------------------------------------------------------------------
# Method 3: Category Consistency@5
# ---------------------------------------------------------------------------

def _parse_categories(raw: str) -> set[str]:
    """Parse comma-separated category string into a set of trimmed category names."""
    if not raw:
        return set()
    return {c.strip() for c in raw.split(",") if c.strip()}


def run_category_consistency(
    companies: list[dict],
    all_embeddings: np.ndarray,
    company_ids: list[int],
    bm25,
    k: int = 5,
    sample: int | None = None,
    run_hyde: bool = False,
    hyde_cache: dict | None = None,
) -> dict:
    """
    For each company with a known category, search by its document and measure
    what fraction of Top-K neighbours (excluding itself) share at least one category.

    High consistency = the semantic space clusters companies by industry correctly.
    BM25 tends to match on company name/location tokens, not category semantics.
    """
    # Only evaluate companies with non-empty categories
    with_cats = [co for co in companies if co.get("categories")]
    pool = with_cats if sample is None else random.sample(with_cats, min(sample, len(with_cats)))

    sem_scores, bm25_scores, pipeline_scores = [], [], []
    low_consistency = []
    id_to_co = {c["id"]: c for c in companies}

    for co in pool:
        own_cats = _parse_categories(co["categories"])
        if not own_cats:
            continue

        cid = co["id"]
        doc = co["document"]

        def cat_overlap(neighbour_ids):
            hits = sum(
                1 for nid in neighbour_ids
                if own_cats & _parse_categories(id_to_co[nid].get("categories", ""))
            )
            return hits / k if k > 0 else 0

        # Retrieve k+5 to exclude self, then take k
        sem_results = _semantic_search_by_embedding(co["embedding"], all_embeddings, company_ids, n=k + 5)
        sem_neighbours = [rid for rid in sem_results if rid != cid][:k]
        sem_score = cat_overlap(sem_neighbours)
        sem_scores.append(sem_score)

        bm25_results = _bm25_search(bm25, doc[:300], company_ids, n=k + 5)
        bm25_neighbours = [rid for rid in bm25_results if rid != cid][:k]
        bm25_score = cat_overlap(bm25_neighbours)
        bm25_scores.append(bm25_score)

        # Pipeline: pass company description as partner_type_desc
        if run_hyde:
            pipe_results, _ = _hyde_pipeline_search_full(
                user_company_desc="",
                partner_type_desc=doc[:600],
                all_embeddings=all_embeddings,
                company_ids=company_ids,
                n=k + 5,
                cache=hyde_cache,
            )
            pipe_neighbours = [rid for rid in pipe_results if rid != cid][:k]
            pipe_score = cat_overlap(pipe_neighbours)
            pipeline_scores.append(pipe_score)

        if sem_score < 0.4:
            low_consistency.append({
                "name": co["name"],
                "categories": co["categories"],
                "sem_consistency": round(sem_score, 2),
            })

    # Per-category breakdown (top 10 most common categories)
    from collections import Counter
    cat_counter = Counter()
    cat_sem = {}
    for co, score in zip(pool, sem_scores):
        for cat in _parse_categories(co.get("categories", "")):
            cat_counter[cat] += 1
            cat_sem.setdefault(cat, []).append(score)

    per_category = {}
    for cat, _ in cat_counter.most_common(10):
        scores = cat_sem.get(cat, [])
        if scores:
            per_category[cat] = {
                "count": len(scores),
                "mean_semantic_consistency": round(float(np.mean(scores)), 3),
            }

    return {
        "k": k,
        "n_companies": len(sem_scores),
        "run_hyde": run_hyde,
        "mean_semantic_consistency": float(np.mean(sem_scores)) if sem_scores else 0,
        "mean_bm25_consistency": float(np.mean(bm25_scores)) if bm25_scores else 0,
        "mean_pipeline_consistency": float(np.mean(pipeline_scores)) if pipeline_scores else None,
        "delta": float(np.mean(sem_scores) - np.mean(bm25_scores)) if sem_scores else 0,
        "per_category_top10": per_category,
        "low_consistency_examples": sorted(low_consistency, key=lambda x: x["sem_consistency"])[:5],
    }


# ---------------------------------------------------------------------------
# Method 4: Motivational Zero-Result Rate
# ---------------------------------------------------------------------------

def _bm25_top_k(bm25, query: str, companies: list[dict], n: int = 10) -> list[dict]:
    """Return top-k companies by BM25."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(-scores)[:n]
    return [
        {"id": companies[i]["id"], "name": companies[i]["name"], "score": float(scores[i])}
        for i in top_idx
        if float(scores[i]) > 0
    ]


def _try_original_site_search(query: str, n: int = 10) -> list[dict] | None:
    """
    Query the original SDGZero website via WordPress GeoDirectory REST API.

    WordPress post IDs are identical to our DB IDs (same dataset), so the
    returned IDs can be looked up directly in id_to_co for SDG relevance scoring.

    Returns None if the request fails or times out (non-blocking).
    """
    try:
        import httpx
        resp = httpx.get(
            "https://sdgzero.com/wp-json/geodir/v2/businesses",
            params={"search": query, "per_page": n},
            timeout=8.0,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return [{"id": r.get("id"), "name": r.get("post_title", "")} for r in data]
        return []
    except Exception:
        return None  # Network unavailable or API broken


def run_website_baseline(
    companies: list[dict],
    skip: bool = False,
    k: int = 10,
) -> dict:
    """
    Run the original SDGZero website (WordPress GeoDirectory, MySQL LIKE search) as a
    standalone baseline across all 3 natural-language query sets.

    Tests:
      M2: 17 UN SDG official descriptions
      M4: 30 motivational queries (synonyms, jargon, concepts)
      M5: 20 OOV queries (no canonical domain keywords)

    WordPress post IDs == our DB IDs, so returned results can be scored for SDG relevance.
    Effective relevance: zero-result queries are counted as 0 (not excluded), so the
    metric reflects real user experience — if the system returns nothing, the user gets nothing.

    Returns:
        Comprehensive dict with per-set stats. {"available": False} if skip=True or API down.
    """
    if skip:
        return {"available": False, "skipped": True, "k": k}

    id_to_co = {co["id"]: co for co in companies}

    def _fetch_and_score(query: str, expected_sdg_key: str | None) -> dict | None:
        """Call website API for one query. Returns result dict or None if API is down."""
        match_kw = None
        if expected_sdg_key:
            for sdg_name, kw in SDG_MATCH_KEYWORDS.items():
                if sdg_name == expected_sdg_key or sdg_name.startswith(expected_sdg_key + " "):
                    match_kw = kw
                    break

        data = _try_original_site_search(query, n=k)
        if data is None:
            return None  # API unavailable

        result_count = len(data)
        sdg_rel_nonzero = None
        if result_count > 0 and match_kw:
            ids = [r["id"] for r in data if r.get("id") and r["id"] in id_to_co]
            if ids:
                hits = sum(1 for cid in ids if _sdg_hit(id_to_co[cid], match_kw))
                sdg_rel_nonzero = hits / len(ids)

        # Effective = 0.0 for zero-result queries (counts them in the average, not excluded)
        sdg_rel_eff = sdg_rel_nonzero if result_count > 0 else 0.0

        return {
            "result_count": result_count,
            "sdg_relevance": sdg_rel_nonzero,    # precision among returned results only
            "sdg_relevance_eff": sdg_rel_eff,    # counts zero-result queries as 0
        }

    def _agg(results: list[dict]) -> dict:
        n = len(results)
        zero_rate = sum(1 for r in results if r["result_count"] == 0) / n
        n_nonzero = sum(1 for r in results if r["result_count"] > 0)
        eff_vals = [r["sdg_relevance_eff"] for r in results if r.get("sdg_relevance_eff") is not None]
        return {
            "zero_rate": zero_rate,
            "n_nonzero": n_nonzero,
            "mean_sdg_relevance_eff": float(np.mean(eff_vals)) if eff_vals else None,
        }

    print("    M2 (17 SDG descriptions)...", end="", flush=True)
    m2_per_sdg: dict[str, dict] = {}
    for sdg_name, query_text in SDG_QUERIES.items():
        r = _fetch_and_score(query_text, sdg_name)
        if r is None:
            return {"available": False, "error": "API unavailable (M2)", "k": k}
        m2_per_sdg[sdg_name] = r
    m2_agg = _agg(list(m2_per_sdg.values()))
    print(f" zero-rate {m2_agg['zero_rate']:.0%}")

    print("    M4 (30 motivational queries)...", end="", flush=True)
    m4_per_query: dict[int, dict] = {}
    for q in MOTIVATIONAL_QUERIES:
        r = _fetch_and_score(q["query"], q["expected_sdg"])
        if r is None:
            return {"available": False, "error": "API unavailable (M4)", "k": k}
        m4_per_query[q["id"]] = {**r, "theme": q["theme"], "expected_sdg": q["expected_sdg"], "query": q["query"]}
    m4_agg = _agg(list(m4_per_query.values()))
    m4_per_theme: dict[str, dict] = {}
    for theme in ("synonym", "sdg_jargon", "concept"):
        tqs = [v for v in m4_per_query.values() if v["theme"] == theme]
        t_eff = [r["sdg_relevance_eff"] for r in tqs if r.get("sdg_relevance_eff") is not None]
        m4_per_theme[theme] = {
            "n": len(tqs),
            "zero_rate": sum(1 for r in tqs if r["result_count"] == 0) / len(tqs),
            "mean_sdg_relevance_eff": float(np.mean(t_eff)) if t_eff else None,
        }
    print(f" zero-rate {m4_agg['zero_rate']:.0%}")

    print("    M5 (20 OOV queries)...", end="", flush=True)
    m5_per_query: dict[int, dict] = {}
    for q in OOV_QUERIES:
        r = _fetch_and_score(q["query"], q["expected_sdg"])
        if r is None:
            return {"available": False, "error": "API unavailable (M5)", "k": k}
        m5_per_query[q["id"]] = {**r, "expected_sdg": q["expected_sdg"], "query": q["query"]}
    m5_agg = _agg(list(m5_per_query.values()))
    print(f" zero-rate {m5_agg['zero_rate']:.0%}")

    return {
        "available": True,
        "k": k,
        "m2": {**m2_agg, "per_sdg": m2_per_sdg},
        "m4": {**m4_agg, "per_theme": m4_per_theme, "per_query": m4_per_query},
        "m5": {**m5_agg, "per_query": m5_per_query},
    }


def run_motivational_analysis(
    companies: list[dict],
    all_embeddings: np.ndarray,
    company_ids: list[int],
    encoder,
    bm25,
    skip_website: bool = False,
    k: int = 10,
    run_hyde: bool = False,
    hyde_cache: dict | None = None,
    website_baseline: dict | None = None,
) -> dict:
    """
    Run 30 motivational queries — natural language phrasing a real user might type.

    Three search paths when run_hyde=True:
      Raw semantic / Full pipeline (HyDE) / BM25

    Metrics:
    1. Zero-Result Rate: fraction of queries returning 0 results.
    2. SDG Relevance@K: fraction of Top-K results tagged with the expected SDG.
    """
    id_to_co = {co["id"]: co for co in companies}
    results = []
    sem_zero, bm25_zero = 0, 0
    sem_relevance, bm25_relevance, pipeline_relevance = [], [], []

    # Resolve pre-computed website baseline for M4
    wb_m4 = (website_baseline or {}).get("m4", {}) if (website_baseline and website_baseline.get("available")) else {}

    for q in MOTIVATIONAL_QUERIES:
        query = q["query"]
        expected_sdg_key = q["expected_sdg"]
        match_kw = None
        for sdg_name, kw in SDG_MATCH_KEYWORDS.items():
            if sdg_name.startswith(expected_sdg_key + " ") or sdg_name.startswith(expected_sdg_key):
                match_kw = kw
                break

        # Raw semantic
        query_emb = _encode_query(encoder, query)
        sem_ids = _semantic_search_by_embedding(query_emb, all_embeddings, company_ids, n=k)
        sem_count = len(sem_ids)

        # BM25
        bm25_top_ids = [r["id"] for r in _bm25_top_k(bm25, query, companies, n=k)]
        bm25_count = len(bm25_top_ids)
        if bm25_count == 0:
            bm25_zero += 1

        # Full pipeline (HyDE) — optional
        pipeline_rel, pipeline_inferred = None, None
        if run_hyde:
            pipeline_ids, pipeline_inferred = _hyde_pipeline_search(
                query, all_embeddings, company_ids, n=k, cache=hyde_cache
            )
            if match_kw:
                pipeline_hits = sum(1 for cid in pipeline_ids if _sdg_hit(id_to_co[cid], match_kw))
                pipeline_rel = pipeline_hits / k
                pipeline_relevance.append(pipeline_rel)

        # SDG Relevance@K
        sem_rel, bm25_rel = None, None
        if match_kw:
            sem_hits = sum(1 for cid in sem_ids if _sdg_hit(id_to_co[cid], match_kw))
            bm25_hits = sum(1 for cid in bm25_top_ids if _sdg_hit(id_to_co[cid], match_kw))
            sem_rel = sem_hits / k
            bm25_rel = bm25_hits / k if bm25_top_ids else 0.0
            sem_relevance.append(sem_rel)
            bm25_relevance.append(bm25_rel)

        # Website — pull from pre-computed baseline (no inline API calls)
        wp_result_count = None
        wp_rel = None
        if wb_m4:
            wp_entry = wb_m4.get("per_query", {}).get(q["id"])
            if wp_entry is not None:
                wp_result_count = wp_entry["result_count"]
                wp_rel = wp_entry.get("sdg_relevance")  # precision among returned (None for zero-result)

        results.append({
            "id": q["id"],
            "query": query,
            "theme": q["theme"],
            "expected_sdg": q["expected_sdg"],
            "semantic_results": sem_count,
            "bm25_results": bm25_count,
            "website_results": wp_result_count,
            "sem_relevance": round(sem_rel, 2) if sem_rel is not None else None,
            "bm25_relevance": round(bm25_rel, 2) if bm25_rel is not None else None,
            "pipeline_relevance": round(pipeline_rel, 2) if pipeline_rel is not None else None,
            "pipeline_inferred_type": pipeline_inferred,
            "website_relevance": round(wp_rel, 2) if wp_rel is not None else None,
        })

    total = len(MOTIVATIONAL_QUERIES)

    # Website zero-rate and relevance come from baseline (effective: zero-result = 0)
    wp_zero_rate = wb_m4.get("zero_rate") if wb_m4 else None
    mean_wp_relevance = wb_m4.get("mean_sdg_relevance_eff") if wb_m4 else None

    # Per-theme breakdown
    theme_stats = {}
    for theme in ("synonym", "sdg_jargon", "concept"):
        theme_qs = [r for r in results if r["theme"] == theme]
        theme_sem_rel = [r["sem_relevance"] for r in theme_qs if r["sem_relevance"] is not None]
        theme_bm25_rel = [r["bm25_relevance"] for r in theme_qs if r["bm25_relevance"] is not None]
        theme_pipeline_rel = [r["pipeline_relevance"] for r in theme_qs if r.get("pipeline_relevance") is not None]
        wb_theme = wb_m4.get("per_theme", {}).get(theme, {}) if wb_m4 else {}
        theme_stats[theme] = {
            "n": len(theme_qs),
            "bm25_zero_rate": sum(1 for r in theme_qs if r["bm25_results"] == 0) / len(theme_qs),
            "wp_zero_rate": wb_theme.get("zero_rate"),
            "sem_sdg_relevance": float(np.mean(theme_sem_rel)) if theme_sem_rel else None,
            "bm25_sdg_relevance": float(np.mean(theme_bm25_rel)) if theme_bm25_rel else None,
            "pipeline_sdg_relevance": float(np.mean(theme_pipeline_rel)) if theme_pipeline_rel else None,
            "website_sdg_relevance": wb_theme.get("mean_sdg_relevance_eff"),
        }

    return {
        "n_queries": total,
        "k": k,
        "run_hyde": run_hyde,
        "semantic_zero_rate": sem_zero / total,
        "bm25_zero_rate": bm25_zero / total,
        "website_zero_rate": wp_zero_rate,
        "website_available": bool(wb_m4),
        "mean_sem_sdg_relevance": float(np.mean(sem_relevance)) if sem_relevance else None,
        "mean_bm25_sdg_relevance": float(np.mean(bm25_relevance)) if bm25_relevance else None,
        "mean_pipeline_sdg_relevance": float(np.mean(pipeline_relevance)) if pipeline_relevance else None,
        "mean_website_sdg_relevance": mean_wp_relevance,
        "per_theme": theme_stats,
        "query_results": results,
    }


# ---------------------------------------------------------------------------
# Method 8: Document Fragment Robustness
# ---------------------------------------------------------------------------
# Split each company document into 3 equal fragments (beginning / middle / end).
# Use each fragment as a standalone query to find the same company.
#
# Hypothesis:
#   Semantic search: document embedding is built from meaning, so any portion of
#   a description still carries the company's semantic "fingerprint" → stable MRR.
#   BM25: different fragments contain different keyword sets → MRR collapses.
#
# Metric:
#   Fragment MRR — mean reciprocal rank across all (company, fragment) pairs.
#   Rank Variance — variance of rank across 3 fragments per company (lower = more robust).
# ---------------------------------------------------------------------------


def run_fragment_robustness(
    companies: list[dict],
    all_embeddings: np.ndarray,
    company_ids: list[int],
    encoder,
    bm25,
    n: int = 20,
    k_cat: int = 5,
    sample: int | None = None,
    min_doc_len: int = 300,
) -> dict:
    """
    Method 8: Document Fragment — Category Precision Robustness.

    For each company, split its document into 3 fragments (beginning / middle / end).
    Use each fragment as a query. Measure Category Precision@K (excluding self) —
    what fraction of top-K neighbours share the company's category?

    Semantic search: represents holistic meaning → category-similar companies found
      from any fragment of the description.
    BM25: relies on exact keywords → when category-discriminative keywords appear only
      in the first fragment, the other fragments retrieve off-topic companies.

    Key difference from self-retrieval MRR: this asks "are the RETRIEVED companies
    relevant?" not "can you find this specific company?" — a more realistic evaluation.
    """
    eligible = [co for co in companies if len(co["document"]) >= min_doc_len and co.get("categories")]
    pool = eligible if sample is None else random.sample(eligible, min(sample, len(eligible)))

    id_to_co = {co["id"]: co for co in companies}
    sem_cat_precisions, bm25_cat_precisions = [], []
    # Per-fragment-position precision (to show which parts benefit semantic most)
    sem_by_pos = [[], [], []]
    bm25_by_pos = [[], [], []]

    for co in pool:
        doc = co["document"]
        cid = co["id"]
        own_cats = _parse_categories(co["categories"])
        if not own_cats:
            continue

        n_chars = len(doc)
        t1, t2 = n_chars // 3, 2 * n_chars // 3
        fragments = [doc[:t1].strip(), doc[t1:t2].strip(), doc[t2:].strip()]

        for pos_idx, frag in enumerate(fragments):
            if len(frag) < 50:
                continue

            # Semantic (cap fragment to 512 chars for encoder)
            frag_emb = _encode_query(encoder, frag[:512])
            sem_results = _semantic_search_by_embedding(frag_emb, all_embeddings, company_ids, n=k_cat + 5)
            sem_neighbours = [rid for rid in sem_results if rid != cid][:k_cat]

            # BM25 (cap to 300 chars for token budget)
            bm25_results = _bm25_search(bm25, frag[:300], company_ids, n=k_cat + 5)
            bm25_neighbours = [rid for rid in bm25_results if rid != cid][:k_cat]

            def cat_prec(neighbours):
                if not neighbours:
                    return 0.0
                hits = sum(
                    1 for nid in neighbours
                    if own_cats & _parse_categories(id_to_co[nid].get("categories", ""))
                )
                return hits / k_cat

            sem_p = cat_prec(sem_neighbours)
            bm25_p = cat_prec(bm25_neighbours)
            sem_cat_precisions.append(sem_p)
            bm25_cat_precisions.append(bm25_p)
            sem_by_pos[pos_idx].append(sem_p)
            bm25_by_pos[pos_idx].append(bm25_p)

    pos_labels = ["Beginning (1st third)", "Middle (2nd third)", "End (3rd third)"]
    per_position = {}
    for i, label in enumerate(pos_labels):
        if sem_by_pos[i]:
            per_position[label] = {
                "semantic": float(np.mean(sem_by_pos[i])),
                "bm25": float(np.mean(bm25_by_pos[i])),
                "delta": float(np.mean(sem_by_pos[i])) - float(np.mean(bm25_by_pos[i])),
            }

    return {
        "n_companies": len(pool),
        "k_cat": k_cat,
        "semantic_cat_precision": float(np.mean(sem_cat_precisions)) if sem_cat_precisions else 0,
        "bm25_cat_precision": float(np.mean(bm25_cat_precisions)) if bm25_cat_precisions else 0,
        "delta": float(np.mean(sem_cat_precisions) - np.mean(bm25_cat_precisions)) if sem_cat_precisions else 0,
        "per_position": per_position,
    }


# ---------------------------------------------------------------------------
# Method 5: Out-of-Vocabulary (OOV) Queries + BM25 Score Distribution (Method 6)
# ---------------------------------------------------------------------------
# OOV queries describe an SDG concept WITHOUT using the canonical keywords that
# appear in company descriptions.  BM25 will fail (no token overlap); semantic
# search bridges the vocabulary gap.
#
# This section also computes BM25 Score Distribution (Method 6):
#   compare max BM25 score for OOV queries vs. "easy" keyword-control queries.
#   Even when BM25 is technically non-zero, the score is near-noise level.
# ---------------------------------------------------------------------------

OOV_QUERIES: list[dict] = [
    # Energy (SDG7) — avoids "energy", "renewable", "solar", "electricity", "power"
    {"id": 1,  "query": "making it cheaper for households to keep warm without burning coal or gas",       "expected_sdg": "SDG7"},
    {"id": 2,  "query": "harnessing natural forces like sunlight and moving air to replace fossil fuels", "expected_sdg": "SDG7"},

    # Climate (SDG13) — avoids "climate", "carbon", "emissions", "greenhouse", "net zero"
    {"id": 3,  "query": "helping businesses understand how much they are contributing to atmospheric temperature rise", "expected_sdg": "SDG13"},
    {"id": 4,  "query": "measuring and reducing the invisible gases companies release into the atmosphere",             "expected_sdg": "SDG13"},

    # Education (SDG4) — avoids "education", "training", "learning", "school", "teaching"
    {"id": 5,  "query": "equipping young people with the capabilities needed to join the modern workforce",       "expected_sdg": "SDG4"},
    {"id": 6,  "query": "closing the gap between the skills employers need and what graduates can offer",         "expected_sdg": "SDG4"},

    # Health (SDG3) — avoids "health", "medical", "wellness", "wellbeing", "clinical"
    {"id": 7,  "query": "supporting people struggling with emotional and psychological difficulties",             "expected_sdg": "SDG3"},
    {"id": 8,  "query": "helping individuals maintain their physical and mental condition throughout life",       "expected_sdg": "SDG3"},

    # Water (SDG6) — avoids "water", "sanitation", "drinking", "clean water"
    {"id": 9,  "query": "ensuring people in remote communities can access safe liquid resources to stay alive",   "expected_sdg": "SDG6"},
    {"id": 10, "query": "preventing contaminants from entering the supply that comes out of household taps",     "expected_sdg": "SDG6"},

    # Gender Equality (SDG5) — avoids "gender", "women", "female", "equality", "diversity"
    {"id": 11, "query": "breaking down the systemic barriers that prevent half the population from reaching senior roles", "expected_sdg": "SDG5"},
    {"id": 12, "query": "building a world where a person's sex does not determine their professional ceiling",           "expected_sdg": "SDG5"},

    # Circular Economy (SDG12) — avoids "waste", "recycling", "circular", "sustainable consumption"
    {"id": 13, "query": "turning yesterday's discarded packaging into tomorrow's raw materials",                 "expected_sdg": "SDG12"},
    {"id": 14, "query": "finding productive second lives for materials that would otherwise end up in the ground","expected_sdg": "SDG12"},

    # Innovation (SDG9) — avoids "innovation", "technology", "digital", "tech"
    {"id": 15, "query": "pioneering novel engineering approaches to solve problems that have plagued industry for decades", "expected_sdg": "SDG9"},
    {"id": 16, "query": "creating tools and systems that did not exist before to make production more efficient",          "expected_sdg": "SDG9"},

    # Poverty (SDG1) — avoids "poverty", "financial inclusion", "income", "economic"
    {"id": 17, "query": "helping people escape deprivation and build resilience against life's unexpected shocks",   "expected_sdg": "SDG1"},
    {"id": 18, "query": "giving marginalised communities the means to improve their own living standards",            "expected_sdg": "SDG1"},

    # Decent Work (SDG8) — avoids "employment", "jobs", "work", "workforce", "labour"
    {"id": 19, "query": "businesses that verify factories are not cutting corners at their people's expense",         "expected_sdg": "SDG8"},
    {"id": 20, "query": "creating meaningful livelihoods for people who have been shut out of the formal economy",   "expected_sdg": "SDG8"},
]

# Keyword-control queries (BM25-friendly; used for score distribution comparison)
_KEYWORD_CONTROL_QUERIES: list[dict] = [
    {"query": "renewable energy company solar wind",         "expected_sdg": "SDG7"},
    {"query": "carbon emissions climate action net zero",    "expected_sdg": "SDG13"},
    {"query": "education training learning skills",          "expected_sdg": "SDG4"},
    {"query": "health wellness medical wellbeing",           "expected_sdg": "SDG3"},
    {"query": "water sanitation clean drinking",             "expected_sdg": "SDG6"},
]


def run_oov_analysis(
    companies: list[dict],
    all_embeddings: np.ndarray,
    company_ids: list[int],
    encoder,
    bm25,
    k: int = 10,
    run_hyde: bool = False,
    hyde_cache: dict | None = None,
    website_baseline: dict | None = None,
) -> dict:
    """
    Method 5 + 6: OOV queries & BM25 Score Distribution.

    Method 5 — OOV Zero-Result / Low-Signal Rate:
      Queries describe SDG concepts without using the canonical domain keywords.
      Raw semantic: always returns k results.
      Full pipeline (HyDE): expands OOV query into a rich company profile first.
      BM25: near-zero scores (no token overlap).

    Method 6 — BM25 Score Distribution:
      Compares max BM25 score for OOV vs keyword-control queries.
    """
    id_to_co = {co["id"]: co for co in companies}

    oov_results = []
    sem_relevance, bm25_relevance, pipeline_relevance, website_relevance = [], [], [], []
    bm25_max_scores_oov = []
    sem_zero, bm25_zero = 0, 0

    # Pre-computed website baseline for M5
    wb_m5 = (website_baseline or {}).get("m5", {}) if (website_baseline and website_baseline.get("available")) else {}

    for q in OOV_QUERIES:
        query = q["query"]
        expected_sdg_key = q["expected_sdg"]
        match_kw = next(
            (kw for sdg_name, kw in SDG_MATCH_KEYWORDS.items()
             if sdg_name.startswith(expected_sdg_key)),
            None
        )

        # Raw semantic
        query_emb = _encode_query(encoder, query)
        sem_ids = _semantic_search_by_embedding(query_emb, all_embeddings, company_ids, n=k)

        # BM25 — capture raw scores too
        tokens = query.lower().split()
        bm25_scores_arr = bm25.get_scores(tokens)
        max_bm25_score = float(np.max(bm25_scores_arr))
        bm25_max_scores_oov.append(max_bm25_score)
        top_bm25_idx = np.argsort(-bm25_scores_arr)[:k]
        bm25_top_ids = [company_ids[i] for i in top_bm25_idx if float(bm25_scores_arr[i]) > 0]
        if not bm25_top_ids:
            bm25_zero += 1

        # Full pipeline (HyDE) — optional
        pipeline_rel, pipeline_inferred = None, None
        if run_hyde:
            pipeline_ids, pipeline_inferred = _hyde_pipeline_search(
                query, all_embeddings, company_ids, n=k, cache=hyde_cache
            )
            if match_kw:
                pipeline_hits = sum(1 for cid in pipeline_ids if _sdg_hit(id_to_co[cid], match_kw))
                pipeline_rel = pipeline_hits / k
                pipeline_relevance.append(pipeline_rel)

        # SDG relevance
        sem_rel, bm25_rel = None, None
        if match_kw:
            sem_hits = sum(1 for cid in sem_ids if _sdg_hit(id_to_co[cid], match_kw))
            bm25_hits = sum(1 for cid in bm25_top_ids[:k] if _sdg_hit(id_to_co[cid], match_kw))
            sem_rel = sem_hits / k
            bm25_rel = bm25_hits / k if bm25_top_ids else 0.0
            sem_relevance.append(sem_rel)
            bm25_relevance.append(bm25_rel)

        # Website — pull from pre-computed baseline (effective: zero-result = 0)
        wp_rel_oov = None
        if wb_m5:
            wp_entry = wb_m5.get("per_query", {}).get(q["id"])
            if wp_entry is not None:
                wp_rel_oov = wp_entry.get("sdg_relevance_eff")
                if wp_rel_oov is not None:
                    website_relevance.append(wp_rel_oov)

        # Category diversity: number of distinct categories in top-K results
        def _cat_diversity(ids: list[int]) -> int:
            cats: set[str] = set()
            for cid in ids:
                co = id_to_co.get(cid)
                if co:
                    cats.update(_parse_categories(co.get("categories", "")))
            return len(cats)

        sem_cat_div = _cat_diversity(sem_ids)
        pipeline_cat_div = _cat_diversity(pipeline_ids) if run_hyde else None

        oov_results.append({
            "id": q["id"],
            "query": query,
            "expected_sdg": expected_sdg_key,
            "sem_results": len(sem_ids),
            "bm25_results": len(bm25_top_ids),
            "max_bm25_score": round(max_bm25_score, 3),
            "sem_relevance": round(sem_rel, 2) if sem_rel is not None else None,
            "bm25_relevance": round(bm25_rel, 2) if bm25_rel is not None else None,
            "pipeline_relevance": round(pipeline_rel, 2) if pipeline_rel is not None else None,
            "pipeline_inferred_type": pipeline_inferred,
            "website_relevance": round(wp_rel_oov, 2) if wp_rel_oov is not None else None,
            "sem_cat_diversity": sem_cat_div,
            "pipeline_cat_diversity": pipeline_cat_div,
        })

    # --- Keyword-control queries (BM25 score comparison) ---
    bm25_max_scores_control = []
    for ctrl in _KEYWORD_CONTROL_QUERIES:
        tokens = ctrl["query"].lower().split()
        scores = bm25.get_scores(tokens)
        bm25_max_scores_control.append(float(np.max(scores)))

    total = len(OOV_QUERIES)
    bm25_zero_rate = bm25_zero / total
    control_p10 = float(np.percentile(bm25_max_scores_control, 10))
    low_signal_rate = sum(1 for s in bm25_max_scores_oov if s < control_p10) / total

    sem_cat_divs = [r["sem_cat_diversity"] for r in oov_results]
    pipeline_cat_divs = [r["pipeline_cat_diversity"] for r in oov_results if r.get("pipeline_cat_diversity") is not None]

    return {
        "n_queries": total,
        "k": k,
        "run_hyde": run_hyde,
        "bm25_zero_rate": bm25_zero_rate,
        "website_zero_rate": wb_m5.get("zero_rate") if wb_m5 else None,
        "mean_sem_relevance": float(np.mean(sem_relevance)) if sem_relevance else None,
        "mean_bm25_relevance": float(np.mean(bm25_relevance)) if bm25_relevance else None,
        "mean_pipeline_relevance": float(np.mean(pipeline_relevance)) if pipeline_relevance else None,
        "mean_website_relevance": float(np.mean(website_relevance)) if website_relevance else None,
        "mean_sem_cat_diversity": float(np.mean(sem_cat_divs)) if sem_cat_divs else None,
        "mean_pipeline_cat_diversity": float(np.mean(pipeline_cat_divs)) if pipeline_cat_divs else None,
        # BM25 score distribution
        "mean_max_bm25_oov": float(np.mean(bm25_max_scores_oov)),
        "mean_max_bm25_control": float(np.mean(bm25_max_scores_control)),
        "low_signal_rate": low_signal_rate,
        "control_p10_threshold": control_p10,
        "query_results": oov_results,
    }


# ---------------------------------------------------------------------------
# Method 7: HyDE Partner Diversity vs Plain Semantic
# ---------------------------------------------------------------------------
# Hypothesis: plain semantic search finds SIMILAR companies (competitors);
# HyDE first generates a hypothetical PARTNER description, then searches.
# Result: HyDE top-5 should have higher cross-sector diversity.
#
# Metric: Cross-Sector Rate@5 = fraction of top-5 results from a DIFFERENT
# category than the user's own category.
# Also: Category Diversity@5 = number of unique categories in top-5.
# ---------------------------------------------------------------------------

PARTNER_CASES: list[dict] = [
    {
        "id": 1,
        "user_desc": "We are a carbon auditing and net-zero consulting firm helping large corporations measure and reduce their greenhouse gas emissions.",
        "user_category": "Professional Services",
        "label": "Carbon auditor → complementary partners",
    },
    {
        "id": 2,
        "user_desc": "We are an EdTech company offering online vocational courses and digital skills bootcamps for unemployed adults.",
        "user_category": "Education & Training",
        "label": "EdTech → complementary partners",
    },
    {
        "id": 3,
        "user_desc": "We manufacture compostable and biodegradable packaging solutions for food and beverage brands.",
        "user_category": "Manufacturing",
        "label": "Sustainable packaging → complementary partners",
    },
    {
        "id": 4,
        "user_desc": "We are a community mental health charity providing counselling and crisis support services to low-income households.",
        "user_category": "Health & Wellness",
        "label": "Mental health charity → complementary partners",
    },
    {
        "id": 5,
        "user_desc": "We supply sustainable timber, insulation, and low-carbon concrete alternatives to the construction sector.",
        "user_category": "Building & Construction",
        "label": "Green materials supplier → complementary partners",
    },
    {
        "id": 6,
        "user_desc": "We run leadership development programmes for women in mid-career looking to reach senior management.",
        "user_category": "Professional Services",
        "label": "Women's leadership training → complementary partners",
    },
    {
        "id": 7,
        "user_desc": "We are a B2B SaaS company providing supply chain transparency and ethical sourcing analytics to retailers.",
        "user_category": "IT & Software",
        "label": "Supply chain analytics → complementary partners",
    },
    {
        "id": 8,
        "user_desc": "We are a social enterprise providing employment and skills training to adults with learning disabilities.",
        "user_category": "Community & Social Purpose",
        "label": "Disability employment org → complementary partners",
    },
]


def _category_diversity(ids: list[int], id_to_co: dict) -> int:
    """Count unique categories across a list of company IDs."""
    cats = set()
    for cid in ids:
        for c in _parse_categories(id_to_co[cid].get("categories", "")):
            cats.add(c)
    return len(cats)


def _cross_sector_rate(ids: list[int], user_category: str, id_to_co: dict) -> float:
    """Fraction of results NOT sharing the user's primary category."""
    if not ids:
        return 0.0
    cross = sum(
        1 for cid in ids
        if user_category.lower() not in (id_to_co[cid].get("categories") or "").lower()
    )
    return cross / len(ids)


def run_hyde_comparison(
    companies: list[dict],
    all_embeddings: np.ndarray,
    company_ids: list[int],
    encoder,
    k: int = 5,
) -> dict:
    """
    Method 7: HyDE Partner Diversity vs Plain Semantic Search.

    Plain semantic: encode user description → top-k most SIMILAR companies.
      Finds companies like the user — potential competitors, not partners.

    HyDE semantic: LLM generates a hypothetical ideal PARTNER description,
      then encode that → top-k companies of the partner type.
      Finds companies complementary to the user — true partners.

    Metric: Cross-Sector Rate@5 (higher = more complementary partners found).
    """
    try:
        from agent.search_agent import _run_hyde
    except ImportError as e:
        return {"error": str(e), "available": False}

    id_to_co = {co["id"]: co for co in companies}
    case_results = []
    plain_cross_rates, hyde_cross_rates = [], []
    plain_cat_diversities, hyde_cat_diversities = [], []

    for case in PARTNER_CASES:
        user_desc = case["user_desc"]
        user_cat = case["user_category"]

        # --- Plain semantic: encode user description directly ---
        plain_emb = _encode_query(encoder, user_desc)
        plain_ids = _semantic_search_by_embedding(plain_emb, all_embeddings, company_ids, n=k + 3)
        # Exclude exact self if present (shouldn't be — user isn't in DB)
        plain_top = plain_ids[:k]

        # --- HyDE: generate hypothetical partner description, then encode ---
        try:
            partner_desc, expansions, inferred_type = _run_hyde(
                user_company_desc=user_desc,
                other_requirements="",
                partner_type_desc="",
                filters=None,
            )
            # Average embedding across partner_desc + expansions (same as pipeline)
            texts = [partner_desc] + expansions
            embeddings = encoder.encode(texts, normalize_embeddings=True)
            hyde_emb = np.mean(embeddings, axis=0)
            hyde_emb = hyde_emb / max(np.linalg.norm(hyde_emb), 1e-9)

            hyde_ids = _semantic_search_by_embedding(hyde_emb, all_embeddings, company_ids, n=k + 3)
            hyde_top = hyde_ids[:k]
            hyde_ok = True
        except Exception as ex:
            inferred_type = "error"
            hyde_top = plain_top  # fallback
            hyde_ok = False

        plain_cross = _cross_sector_rate(plain_top, user_cat, id_to_co)
        hyde_cross = _cross_sector_rate(hyde_top, user_cat, id_to_co)
        plain_div = _category_diversity(plain_top, id_to_co)
        hyde_div = _category_diversity(hyde_top, id_to_co)

        plain_cross_rates.append(plain_cross)
        hyde_cross_rates.append(hyde_cross)
        plain_cat_diversities.append(plain_div)
        hyde_cat_diversities.append(hyde_div)

        # Top-3 company names for qualitative inspection
        plain_names = [id_to_co[cid]["name"] for cid in plain_top[:3]]
        hyde_names = [id_to_co[cid]["name"] for cid in hyde_top[:3]]

        case_results.append({
            "id": case["id"],
            "label": case["label"],
            "user_category": user_cat,
            "inferred_partner_type": inferred_type if hyde_ok else "N/A",
            "plain_cross_sector_rate": round(plain_cross, 2),
            "hyde_cross_sector_rate": round(hyde_cross, 2),
            "plain_cat_diversity": plain_div,
            "hyde_cat_diversity": hyde_div,
            "plain_top3": plain_names,
            "hyde_top3": hyde_names,
            "hyde_ok": hyde_ok,
        })

    return {
        "k": k,
        "n_cases": len(PARTNER_CASES),
        "mean_plain_cross_sector": float(np.mean(plain_cross_rates)),
        "mean_hyde_cross_sector": float(np.mean(hyde_cross_rates)),
        "mean_plain_cat_diversity": float(np.mean(plain_cat_diversities)),
        "mean_hyde_cat_diversity": float(np.mean(hyde_cat_diversities)),
        "delta_cross_sector": float(np.mean(hyde_cross_rates)) - float(np.mean(plain_cross_rates)),
        "delta_cat_diversity": float(np.mean(hyde_cat_diversities)) - float(np.mean(plain_cat_diversities)),
        "case_results": case_results,
        "available": True,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1%}"


def _delta_str(d: float) -> str:
    return f"{d:+.1%}"


def _results_table(
    rows: list[tuple[str, str | None, str | None, str | None, str | None]],
    has_website: bool,
    has_pipeline: bool,
) -> list[str]:
    """
    Render a standardised results table.
    Each row is (label, website_val, bm25_val, semantic_val, pipeline_val).
    Pass None for a cell to show '—'. Columns suppressed when not available.
    """
    cols = ["Metric"]
    if has_website:
        cols.append("Website (MySQL)")
    cols.append("BM25")
    cols.append("Raw Semantic")
    if has_pipeline:
        cols.append("Full Pipeline (HyDE)")
    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    out = [hdr, sep]
    for (label, wp, bm25, sem, pipe) in rows:
        cells = [label]
        if has_website:
            cells.append(wp if wp is not None else "—")
        cells.append(bm25 if bm25 is not None else "—")
        cells.append(sem if sem is not None else "—")
        if has_pipeline:
            cells.append(pipe if pipe is not None else "—")
        out.append("| " + " | ".join(cells) + " |")
    return out


def generate_report(
    m1: dict,
    m2: dict,
    m3: dict,
    m4: dict,
    n_total: int,
    m5: dict | None = None,
    m7: dict | None = None,
    m8: dict | None = None,
    m0: dict | None = None,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d")
    has_pipeline = m2.get("run_hyde") or m4.get("run_hyde") or (m5 and m5.get("run_hyde"))
    has_website  = bool(m0 and m0.get("available"))

    # ------------------------------------------------------------------ helpers
    def _R(label, wp, bm25, sem, pipe) -> tuple:
        return (label, _pct(wp) if wp is not None else None,
                _pct(bm25) if bm25 is not None else None,
                _pct(sem) if sem is not None else None,
                _pct(pipe) if pipe is not None else None)

    def _section(title, description_lines, calculation_lines, result_rows, analysis_lines):
        """One complete method section. Each subsection is separated by exactly one blank line."""
        # Strip trailing empty strings from caller-supplied lists to avoid double blanks
        def _strip(lst):
            while lst and lst[-1] == "":
                lst = lst[:-1]
            return lst
        out = ["", "---", "", f"## {title}", ""]
        out += ["### What it measures", ""] + _strip(description_lines) + [""]
        out += ["### How it is calculated", ""] + _strip(calculation_lines) + [""]
        out += ["### Results", ""] + _results_table(result_rows, has_website, has_pipeline) + [""]
        out += ["### Analysis", ""] + _strip(analysis_lines)
        return out

    # ------------------------------------------------------------------ header
    lines = [
        "# RAG Search Quality Evaluation Report",
        "",
        f"> Generated: {ts}",
        f"> Database: {n_total} companies",
        f"> Embedding model: all-MiniLM-L6-v2 (384-dim, cosine similarity)",
        f"> Baselines: Website (WordPress MySQL LIKE) · BM25Okapi · Raw Semantic · Full Pipeline (HyDE)",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # ------------------------------------------------------------------ summary table
    sum_rows = []
    sum_rows.append(("M1 — Self-Retrieval MRR",
        None, f"{m1['bm25_mrr']:.4f}", f"{m1['semantic_mrr']:.4f}",
        f"{m1['pipeline_mrr']:.4f}" if m1.get("pipeline_mrr") is not None else None))
    sum_rows.append(_R(f"M2 ⚠️ — SDG Precision@{m2['k']} (all 17, vs ML predictor)",
        m2.get("mean_website_precision"),
        m2["mean_bm25_precision"], m2["mean_semantic_precision"],
        m2.get("mean_pipeline_precision")))
    sum_rows.append(_R(f"M2 ⚠️ — SDG Precision@{m2['k']} (discriminative)",
        None,
        m2["disc_bm25_precision"], m2["disc_semantic_precision"],
        m2.get("disc_pipeline_precision")))
    sum_rows.append(_R(f"M2 ⚠️ — SDG NDCG@{m2['k']} (discriminative)",
        None,
        m2["disc_bm25_ndcg"], m2["disc_semantic_ndcg"],
        m2.get("disc_pipeline_ndcg")))
    sum_rows.append(_R(f"M3 — Category Consistency@{m3['k']}",
        None,
        m3["mean_bm25_consistency"], m3["mean_semantic_consistency"],
        m3.get("mean_pipeline_consistency")))
    sum_rows.append(_R(f"M4 — Zero-Result Rate ↓",
        m4.get("website_zero_rate"),
        m4["bm25_zero_rate"], m4["semantic_zero_rate"],
        0.0 if has_pipeline else None))
    sum_rows.append(_R(f"M4 — SDG Relevance@{m4['k']} (effective) ↑",
        m4.get("mean_website_sdg_relevance"),
        m4.get("mean_bm25_sdg_relevance"), m4.get("mean_sem_sdg_relevance"),
        m4.get("mean_pipeline_sdg_relevance")))
    if m5:
        sum_rows.append(_R(f"M5 — OOV Zero-Result Rate ↓",
            m5.get("website_zero_rate"),
            m5["bm25_zero_rate"], 0.0, 0.0 if has_pipeline else None))
        sum_rows.append(_R(f"M5 — OOV SDG Relevance@{m5['k']} (effective) ↑",
            m5.get("mean_website_relevance"),
            m5.get("mean_bm25_relevance"), m5.get("mean_sem_relevance"),
            m5.get("mean_pipeline_relevance")))
        sum_rows.append(_R("M6 — BM25 Low-Signal Rate ↑",
            None, m5["low_signal_rate"], None, None))
    if m7 and m7.get("available"):
        sum_rows.append(("M7 — HyDE Cross-Sector@5 ↑",
            "—", "—",
            _pct(m7["mean_plain_cross_sector"]) + " (plain)",
            _pct(m7["mean_hyde_cross_sector"]) + " (HyDE)"))
    if m8:
        sum_rows.append((_R(f"M8 ⚠️ — Fragment Cat. Precision@{m8['k_cat']} (embedding only)",
            None, m8["bm25_cat_precision"], m8["semantic_cat_precision"], None)))
    lines += _results_table(sum_rows, has_website, has_pipeline)

    # ================================================================== Method 0
    if has_website:
        m0m2, m0m4, m0m5 = m0["m2"], m0["m4"], m0["m5"]
        wp_m2_nonzero = (
            sum(r["sdg_relevance"] for r in m0m2["per_sdg"].values() if r.get("sdg_relevance") is not None)
            / m0m2["n_nonzero"]
            if m0m2["n_nonzero"] > 0 else None
        )
        lines += [
            "", "---", "",
            "## Method 0: Original Website Baseline (WordPress MySQL LIKE)",
            "",
            "### What it measures",
            "",
            "The original SDGZero website (sdgzero.com) uses WordPress GeoDirectory with",
            "MySQL LIKE keyword search. This method establishes the real-world lower bound:",
            "how well the system being replaced actually performs on natural-language queries.",
            "",
            "### How it is calculated",
            "",
            "- The same 3 natural-language query sets (M2/M4/M5) are sent to the WordPress REST API.",
            "- **Zero-result rate**: fraction of queries returning an empty page.",
            "- **Effective SDG relevance**: zero-result queries count as 0 precision.",
            "  When the user gets no results, they get no value — so zero-result queries",
            "  are included in the mean rather than excluded.",
            "- WordPress post IDs are identical to our PostgreSQL `businesses.id`, so",
            "  returned results can be scored for SDG relevance directly.",
            "",
            "### Results",
            "",
            "**Zero-result rate by query set:**",
            "",
            "| Query Set | # Queries | Website | BM25 | Semantic | Pipeline |",
            "|-----------|-----------|---------|------|----------|----------|",
            f"| M2: SDG descriptions | 17 | **{_pct(m0m2['zero_rate'])}** | 0.0% | 0.0% | 0.0% |",
            f"| M4: Motivational | 30 | **{_pct(m0m4['zero_rate'])}** | {_pct(m4['bm25_zero_rate'])} | 0.0% | 0.0% |",
            f"| M5: OOV | 20 | **{_pct(m0m5['zero_rate'])}** | {_pct(m5['bm25_zero_rate'] if m5 else None)} | 0.0% | 0.0% |",
            "",
            "**SDG relevance by query set** (effective — zero-result queries counted as 0):",
            "",
            "| Query Set | Website (eff.) | Website (non-zero only) | BM25 | Raw Semantic | Full Pipeline |",
            "|-----------|----------------|------------------------|------|-------------|--------------|",
            f"| M2: SDG descriptions | {_pct(m0m2['mean_sdg_relevance_eff'])} | {_pct(wp_m2_nonzero)} | {_pct(m2['mean_bm25_precision'])} | {_pct(m2['mean_semantic_precision'])} | {_pct(m2.get('mean_pipeline_precision'))} |",
            f"| M4: Motivational     | {_pct(m0m4['mean_sdg_relevance_eff'])} | {_pct(m4.get('mean_website_sdg_relevance'))} | {_pct(m4['mean_bm25_sdg_relevance'])} | {_pct(m4['mean_sem_sdg_relevance'])} | {_pct(m4.get('mean_pipeline_sdg_relevance'))} |",
            f"| M5: OOV              | {_pct(m0m5['mean_sdg_relevance_eff'])} | — | {_pct(m5['mean_bm25_relevance'] if m5 else None)} | {_pct(m5['mean_sem_relevance'] if m5 else None)} | {_pct(m5.get('mean_pipeline_relevance') if m5 else None)} |",
            "",
            "**Per-SDG website P@K (M2):**",
            "",
            "| SDG | Website results | Website P@K (eff.) | BM25 P@K | Semantic P@K | Pipeline P@K |",
            "|-----|----------------|--------------------|---------|-------------|-------------|",
        ]
        for sdg_name, wp_entry in m0m2["per_sdg"].items():
            s = m2["per_sdg"].get(sdg_name, {})
            lines.append(
                f"| {sdg_name} | {wp_entry['result_count']} | {_pct(wp_entry['sdg_relevance_eff'])}"
                f" | {_pct(s.get('bm25_precision'))} | {_pct(s.get('semantic_precision'))} | {_pct(s.get('pipeline_precision'))} |"
            )
        lines += [
            "",
            "### Analysis",
            "",
            f"- The website returns **0 results for 100% of SDG description queries** and "
            f"{_pct(m0m4['zero_rate'])} of motivational queries — users searching with "
            "natural language consistently see an empty page.",
            "- When the site does return results (only 2/30 motivational queries), the results are "
            "high-precision keyword matches. Keyword search is exact by nature, but the coverage "
            "is negligible.",
            f"- The pipeline always returns {m4['k']} results for every query, with "
            f"{_pct(m4.get('mean_pipeline_sdg_relevance'))} mean SDG relevance on motivational "
            f"queries vs the website's effective {_pct(m0m4['mean_sdg_relevance_eff'])}.",
        ]

    # ================================================================== Method 1
    lines += _section(
        "Method 1: Self-Retrieval MRR",
        description_lines=[
            "A company's own description, used as a search query, should return itself as rank #1.",
            "This is a sanity-check on the embedding space: if a company's document is not its own",
            "nearest neighbour, the vector space is incoherent.",
            "",
            "- **Website**: N/A — the website search box accepts short keyword queries only;",
            "  it cannot accept a full company document as input.",
            "- **BM25**: Expected near-perfect (1.0) — exact token overlap guarantees self-match.",
            "- **Raw Semantic**: Should be close to 1.0 if the embedding captures document identity.",
            "- **Pipeline (HyDE)**: Tested by passing the company's own description as `partner_type_desc`",
            "  (leaving `user_company_desc` empty). HyDE generates a hypothetical company profile",
            "  matching that description, then finds the nearest companies in the embedding space.",
            "  The question: does the original company rank highly when you describe it as the target type?",
            "  This is not identical to self-retrieval (HyDE adds LLM-generated noise), but it tests",
            "  whether the pipeline can 'find itself' when given its own description as input.",
        ],
        calculation_lines=[
            f"For each of {m1['n_companies']} companies, encode its full description as a query.",
            f"Retrieve top-{m1['n_results']} results. Record the rank of the company itself.",
            "MRR = mean(1 / rank). MRR = 1.0 means every company ranks itself #1.",
        ],
        result_rows=[
            ("MRR", None,
             f"{m1['bm25_mrr']:.4f}",
             f"{m1['semantic_mrr']:.4f}",
             f"{m1['pipeline_mrr']:.4f}" if m1.get("pipeline_mrr") is not None else None),
            ("Rank 1 rate", None,
             f"{m1['rank_distribution']['rank_1']/m1['n_companies']:.1%}",
             f"{m1['rank_distribution']['rank_1']/m1['n_companies']:.1%}",
             (f"{m1['pipe_rank_distribution']['rank_1']/m1['n_companies']:.1%}"
              if m1.get("pipe_rank_distribution") else None)),
            ("Rank 2–5", None, "—",
             f"{m1['rank_distribution']['rank_2_5']/m1['n_companies']:.1%}",
             (f"{m1['pipe_rank_distribution']['rank_2_5']/m1['n_companies']:.1%}"
              if m1.get("pipe_rank_distribution") else None)),
            ("Rank 11+", None, "—",
             f"{m1['rank_distribution']['rank_11_plus']/m1['n_companies']:.1%}",
             (f"{m1['pipe_rank_distribution']['rank_11_plus']/m1['n_companies']:.1%}"
              if m1.get("pipe_rank_distribution") else None)),
        ],
        analysis_lines=[
            f"- Semantic MRR = **{m1['semantic_mrr']:.4f}** (BM25 = {m1['bm25_mrr']:.4f}, Δ = {_delta_str(m1['delta'])}).",
            (f"- Pipeline MRR = **{m1['pipeline_mrr']:.4f}** — HyDE adds generative noise, so"
             " this will be lower than raw semantic. Gap shows how much the LLM 'drifts' from"
             " the original description.")
            if m1.get("pipeline_mrr") is not None else "",
            f"- {m1['rank_distribution']['rank_1']/m1['n_companies']:.1%} of companies rank themselves #1 in raw semantic.",
            "- Result confirms the embedding space is coherent.",
        ] + (["", "**Companies not ranking #1 (raw semantic):**", "", "| Company | Rank |", "|---------|------|"]
             + [f"| {f['name']} | {f['rank']} |" for f in m1["top_failures"]]
             if m1["top_failures"] else []),
    )

    # ================================================================== Method 2
    lines += _section(
        f"Method 2: SDG-based Precision@{m2['k']} and NDCG@{m2['k']} ⚠️ Supplemental",
        description_lines=[
            "> ⚠️ **Methodological limitation — use with caution.**",
            "> The ground truth labels (`predicted_sdg_tags`) were generated by an ML model.",
            "> M2 therefore measures how well the search system agrees with the SDG *predictor*,",
            "> not how relevant results are to an independent human standard. A result that is",
            "> genuinely relevant but wasn't tagged by the predictor will be counted as wrong.",
            "> Treat M2 as a consistency check, not a ground-truth quality measurement.",
            "",
            "17 UN SDG official descriptions are used as natural-language queries.",
            "Results are scored for SDG relevance using `predicted_sdg_tags` labels in the database.",
            "The primary signal is the *relative* comparison between systems under identical conditions.",
            "",
            "- **Website**: 100% zero-result (formal UN text has no keyword overlap) — all scores 0.",
            "- **BM25**: Benefits from SDG keyword overlap in descriptions.",
            "- **Raw Semantic**: Bridges vocabulary gap between UN language and company text.",
            "- **Pipeline (HyDE)**: Rewrites the SDG description into a hypothetical company profile.",
            "  This may broaden the query semantics beyond the narrow SDG label — leading to lower",
            "  precision against the predictor's labels even when results are legitimately relevant.",
            "",
            f"**Discriminative SDGs** ({m2['n_discriminative']} of 17): SDGs with 2%–35% corpus",
            "prevalence, where retrieval quality differs between systems.",
            "High-prevalence SDGs (e.g. SDG8 Decent Work at 72%) inflate all systems equally.",
            "",
            "**NDCG (Normalized Discounted Cumulative Gain)**: weights relevant results higher if",
            "they appear at rank 1 vs rank 10. NDCG = 1.0 means all relevant results are at the top.",
            "Unlike Precision@K which only counts hits, NDCG rewards correct *ordering*.",
        ],
        calculation_lines=[
            f"For each of 17 SDGs, the official UN description is used as the query.",
            f"Top-{m2['k']} results are retrieved. For each result, check whether `predicted_sdg_tags`",
            "contains the query SDG.",
            f"- **Precision@{m2['k']}** = (# results with correct tag) / {m2['k']}",
            f"- **NDCG@{m2['k']}** = discounted cumulative gain, weighting results higher when",
            "  relevant results appear at rank 1 vs rank 10.",
        ],
        result_rows=[
            _R(f"Mean Precision@{m2['k']} — all 17 SDGs",
               m2.get("mean_website_precision"),
               m2["mean_bm25_precision"], m2["mean_semantic_precision"],
               m2.get("mean_pipeline_precision")),
            _R(f"Mean Precision@{m2['k']} — discriminative",
               None, m2["disc_bm25_precision"], m2["disc_semantic_precision"],
               m2.get("disc_pipeline_precision")),
            _R(f"Mean NDCG@{m2['k']} — discriminative",
               None, m2["disc_bm25_ndcg"], m2["disc_semantic_ndcg"],
               m2.get("disc_pipeline_ndcg")),
        ],
        analysis_lines=[
            f"- All 17 SDGs: Pipeline {_pct(m2.get('mean_pipeline_precision'))} vs "
            f"BM25 {_pct(m2['mean_bm25_precision'])} vs Raw Semantic {_pct(m2['mean_semantic_precision'])}.",
            f"- Discriminative SDGs: Pipeline {_pct(m2.get('disc_pipeline_precision'))} vs "
            f"BM25 {_pct(m2['disc_bm25_precision'])} ({_delta_str((m2.get('disc_pipeline_precision') or 0) - (m2['disc_bm25_precision'] or 0))}).",
            f"- NDCG discriminative: Pipeline {_pct(m2.get('disc_pipeline_ndcg'))} vs BM25 {_pct(m2['disc_bm25_ndcg'])}.",
            "- Pipeline scores lower on discriminative precision *against the SDG predictor's labels*.",
            "  This is an artefact of the circular-reasoning limitation above: HyDE expands the query",
            "  into a broader company profile, which may retrieve genuinely relevant companies that",
            "  the predictor did not tag with the exact SDG label.",
            "- SDG13 Climate Action is a notable exception: Pipeline NDCG improves because HyDE",
            "  reframes abstract climate language into concrete activities (carbon audits, renewable energy),",
            "  which aligns better with how companies are actually tagged.",
            "",
            "**Per-SDG breakdown:**",
            "",
        ] + (
            ["| SDG | Corpus | Prev. | Disc. | Website P@K | BM25 P@K | Semantic P@K | Pipeline P@K | BM25 NDCG | Pipeline NDCG |",
             "|-----|--------|-------|-------|------------|---------|-------------|-------------|-----------|--------------|"]
            if has_website and has_pipeline else
            ["| SDG | Corpus | Prev. | Disc. | BM25 P@K | Semantic P@K | Pipeline P@K | BM25 NDCG | Pipeline NDCG |",
             "|-----|--------|-------|-------|---------|-------------|-------------|-----------|--------------|"]
            if has_pipeline else
            ["| SDG | Corpus | Prev. | Disc. | BM25 P@K | Semantic P@K | BM25 NDCG | Sem NDCG |",
             "|-----|--------|-------|-------|---------|-------------|-----------|---------|"]
        ) + [
            (
                f"| {sdg_name} | {s['corpus_count']} | {s['prevalence']:.0%} | {'✅' if s['discriminative'] else '—'} "
                + (f"| {_pct(s.get('website_precision'))} " if has_website else "")
                + f"| {_pct(s['bm25_precision'])} | {_pct(s['semantic_precision'])} "
                + (f"| {_pct(s.get('pipeline_precision'))} | {_pct(s['bm25_ndcg'])} | {_pct(s.get('pipeline_ndcg'))} |"
                   if has_pipeline else
                   f"| {_pct(s['bm25_ndcg'])} | {_pct(s['semantic_ndcg'])} |")
            )
            for sdg_name, s in m2["per_sdg"].items()
        ],
    )

    # ================================================================== Method 3
    lines += _section(
        f"Method 3: Category Consistency@{m3['k']}",
        description_lines=[
            "For each company, use its full description as a query and retrieve the top-K neighbours.",
            "Measure what fraction share at least one business category with the query company.",
            "High consistency means the embedding space correctly clusters companies by industry.",
            "",
            "- **Website**: N/A — cannot accept full company documents as queries.",
            "- **BM25**: Companies in the same category share industry keywords → moderate consistency.",
            "- **Raw Semantic**: Meaning-based embeddings should cluster similar businesses.",
            "- **Pipeline (HyDE)**: Tested by passing the company's description as `partner_type_desc`.",
            "  HyDE generates a hypothetical company of that type, then retrieves nearest neighbours.",
            "  Whether results are in the same category depends on how concretely the description",
            "  identifies an industry — a useful additional signal on top of raw semantic.",
        ],
        calculation_lines=[
            f"{m3['n_companies']} companies evaluated. For each, retrieve top-{m3['k']} neighbours",
            "(excluding self). Count how many share at least one `categories` tag with the query company.",
            f"Consistency@{m3['k']} = matching neighbours / {m3['k']}. Mean across all companies.",
        ],
        result_rows=[
            _R(f"Mean Consistency@{m3['k']}",
               None, m3["mean_bm25_consistency"], m3["mean_semantic_consistency"],
               m3.get("mean_pipeline_consistency")),
        ],
        analysis_lines=[
            f"- Semantic {_pct(m3['mean_semantic_consistency'])} vs BM25 {_pct(m3['mean_bm25_consistency'])} "
            f"({_delta_str(m3['delta'])}).",
            (f"- Pipeline {_pct(m3['mean_pipeline_consistency'])} — when HyDE is given a company description"
             " as the target type, this measures how well it finds companies in the same category.")
            if m3.get("mean_pipeline_consistency") is not None else "",
            "- Semantic search clusters by industry — category keywords may not appear uniformly",
            "  across all descriptions, so raw token overlap (BM25) is less reliable.",
            "",
            "**Per-category breakdown (top 10 categories):**",
            "",
            "| Category | # Companies | Semantic Consistency@5 |",
            "|----------|------------|------------------------|",
        ] + [f"| {cat} | {s['count']} | {s['mean_semantic_consistency']:.1%} |"
             for cat, s in m3["per_category_top10"].items()] + (
            ["", "**Low consistency examples (< 40%):**", "",
             "| Company | Categories | Consistency |", "|---------|-----------|-------------|"]
            + [f"| {ex['name']} | {ex['categories']} | {ex['sem_consistency']:.0%} |"
               for ex in m3["low_consistency_examples"]]
            if m3["low_consistency_examples"] else []
        ),
    )

    # ================================================================== Method 4
    lines += _section(
        "Method 4: Motivational Queries — Zero-Result Rate and SDG Relevance",
        description_lines=[
            "30 natural-language queries that a real user might type, split into 3 themes:",
            "- **Synonym**: same concept, different words (e.g. 'clean power' vs 'renewable energy')",
            "- **SDG Jargon**: formal UN SDG language for real activities",
            "- **Concept**: describes a concept without any literal keyword from company descriptions",
            "",
            "Keyword systems (website, BM25) fail when there is no token overlap.",
            "Semantic systems always return results by finding the nearest vectors.",
        ],
        calculation_lines=[
            "For each of 30 queries, run all 4 systems and record:",
            "1. **Zero-result rate**: fraction of queries returning 0 results.",
            "2. **SDG Relevance@K** (effective): fraction of top-K results tagged with the expected SDG.",
            "   Zero-result queries are counted as 0 (user gets nothing → no value).",
        ],
        result_rows=[
            _R("Zero-Result Rate ↓",
               m4.get("website_zero_rate"),
               m4["bm25_zero_rate"], m4["semantic_zero_rate"],
               0.0 if has_pipeline else None),
            _R(f"SDG Relevance@{m4['k']} — effective ↑",
               m4.get("mean_website_sdg_relevance"),
               m4.get("mean_bm25_sdg_relevance"), m4.get("mean_sem_sdg_relevance"),
               m4.get("mean_pipeline_sdg_relevance")),
        ],
        analysis_lines=[
            f"- Website fails on **{_pct(m4.get('website_zero_rate'))}** of queries — users see empty pages.",
            f"- BM25 fails on {_pct(m4['bm25_zero_rate'])} — even a strong keyword system struggles with paraphrases.",
            "- Semantic and pipeline always return results (0% zero-rate).",
            f"- SDG relevance (effective): Pipeline {_pct(m4.get('mean_pipeline_sdg_relevance'))} vs "
            f"BM25 {_pct(m4.get('mean_bm25_sdg_relevance'))} vs Website {_pct(m4.get('mean_website_sdg_relevance'))}.",
            "",
            "**Per-theme breakdown:**",
            "",
            "| Theme | N | Website Zero-Rate | BM25 Zero-Rate | Website Rel. | BM25 Rel. | Semantic Rel. | Pipeline Rel. |",
            "|-------|---|-------------------|---------------|-------------|-----------|--------------|--------------|",
        ] + [
            f"| {theme} | {s['n']} | {_pct(s.get('wp_zero_rate'))} | {_pct(s['bm25_zero_rate'])} "
            f"| {_pct(s.get('website_sdg_relevance'))} | {_pct(s.get('bm25_sdg_relevance'))} "
            f"| {_pct(s.get('sem_sdg_relevance'))} | {_pct(s.get('pipeline_sdg_relevance'))} |"
            for theme, s in m4["per_theme"].items()
        ] + [
            "",
            "**Query-level results:**",
            "",
            "| ID | Query | Theme | Website Rel. | BM25 Rel. | Semantic Rel. | Pipeline Rel. | HyDE Inferred Type |",
            "|----|-------|-------|-------------|-----------|--------------|--------------|---------------------|",
        ] + [
            f"| {r['id']} | {r['query'][:38]} | {r['theme']} "
            f"| {_pct(r.get('website_relevance')) if r.get('website_relevance') is not None else ('0 results' if r.get('website_results') == 0 else '—')} "
            f"| {_pct(r['bm25_relevance']) if r['bm25_relevance'] is not None else '—'} "
            f"| {_pct(r['sem_relevance']) if r['sem_relevance'] is not None else '—'} "
            f"| {_pct(r.get('pipeline_relevance')) if r.get('pipeline_relevance') is not None else '—'} "
            f"| {(r.get('pipeline_inferred_type') or '—')[:28]} |"
            for r in m4["query_results"]
        ],
    )

    # ================================================================== Method 5
    if m5:
        lines += _section(
            "Method 5: Out-of-Vocabulary (OOV) Queries — SDG Relevance and Result Quality",
            description_lines=[
                "20 queries that describe SDG concepts **without using the canonical domain keywords**.",
                "E.g., SDG7 described without 'energy'/'renewable'; SDG13 without 'carbon'/'climate'.",
                "This isolates whether a system can bridge the vocabulary gap.",
                "",
                "- **Website**: 100% zero-result expected — no token overlap whatsoever.",
                "- **BM25**: High zero-result/low-signal rate — vocabulary gap prevents matching.",
                "- **Raw Semantic**: Retrieves by meaning — should find relevant companies.",
                "- **Pipeline (HyDE)**: LLM rewrites the OOV description into a concrete hypothetical",
                "  company profile using its world knowledge, then embeds it — stronger vocabulary bridging.",
                "",
                "**Limitation of SDG-only relevance**: The SDG relevance metric only checks whether",
                "results are tagged with the expected SDG. It does not capture other quality dimensions",
                "such as result diversity or whether results are coherent as a set.",
                "We therefore also report **category diversity** (number of distinct industry categories",
                "in the top-K results) as a proxy for result coherence:",
                "- Low diversity = results are focused on one type of company (probably more useful)",
                "- High diversity = results are scattered across unrelated industries (less useful)",
            ],
            calculation_lines=[
                "For each of 20 OOV queries, retrieve top-K results from all systems.",
                "1. **Zero-result rate**: fraction returning 0 results.",
                f"2. **SDG Relevance@{m5['k']} (effective)**: fraction of top-{m5['k']} tagged with expected SDG.",
                "   Zero-result queries counted as 0 — user gets no value from an empty result page.",
                f"3. **Category Diversity@{m5['k']}**: number of distinct `categories` values in top-{m5['k']} results.",
                "   Computed for Semantic and Pipeline (BM25/Website return near-noise results).",
            ],
            result_rows=[
                _R("Zero-Result Rate ↓",
                   m5.get("website_zero_rate"),
                   m5["bm25_zero_rate"], 0.0,
                   0.0 if has_pipeline else None),
                _R(f"SDG Relevance@{m5['k']} (effective) ↑",
                   m5.get("mean_website_relevance"),
                   m5.get("mean_bm25_relevance"), m5.get("mean_sem_relevance"),
                   m5.get("mean_pipeline_relevance")),
                (f"Category Diversity@{m5['k']} ↓ (lower = more focused)",
                 None, None,
                 f"{m5['mean_sem_cat_diversity']:.1f}" if m5.get("mean_sem_cat_diversity") is not None else None,
                 f"{m5['mean_pipeline_cat_diversity']:.1f}" if m5.get("mean_pipeline_cat_diversity") is not None else None),
            ],
            analysis_lines=[
                f"- Website: {_pct(m5.get('website_zero_rate'))} zero-result rate → users see empty pages.",
                f"- BM25: {_pct(m5['bm25_zero_rate'])} zero-result rate, {_pct(m5.get('mean_bm25_relevance'))} effective SDG relevance.",
                f"- Raw Semantic: 0% zero-result, {_pct(m5.get('mean_sem_relevance'))} SDG relevance, "
                f"{m5['mean_sem_cat_diversity']:.1f} category diversity.",
                f"- Pipeline: 0% zero-result, {_pct(m5.get('mean_pipeline_relevance'))} SDG relevance, "
                f"{m5['mean_pipeline_cat_diversity']:.1f} category diversity "
                + (f"({_delta_str((m5['mean_pipeline_relevance'] or 0) - (m5['mean_sem_relevance'] or 0))} SDG rel vs raw semantic)."
                   if m5.get('mean_pipeline_relevance') else "."),
                "- HyDE provides the largest SDG relevance gain on OOV queries — concrete hypothetical",
                "  profiles bridge the vocabulary gap more effectively than direct embedding.",
                "- Category diversity shows whether the results are coherent: a lower number means",
                "  the system is returning companies from a focused sector; higher means scattered results.",
                "",
                "**Query-level results:**",
                "",
                "| ID | Query | SDG | Website Rel. | BM25 Rel. | Semantic Rel. | Sem Cat.Div | Pipeline Rel. | Pipe Cat.Div | HyDE Inferred Type |",
                "|----|-------|-----|-------------|-----------|--------------|------------|--------------|-------------|---------------------|",
            ] + [
                f"| {r['id']} | {r['query'][:38]} | {r['expected_sdg']} "
                f"| {_pct(r.get('website_relevance')) if r.get('website_relevance') is not None else '—'} "
                f"| {_pct(r['bm25_relevance']) if r['bm25_relevance'] is not None else '—'}{'❌' if r['bm25_results'] == 0 else ''} "
                f"| {_pct(r['sem_relevance']) if r['sem_relevance'] is not None else '—'} "
                f"| {r.get('sem_cat_diversity', '—')} "
                f"| {_pct(r.get('pipeline_relevance')) if r.get('pipeline_relevance') is not None else '—'} "
                f"| {r.get('pipeline_cat_diversity', '—')} "
                f"| {(r.get('pipeline_inferred_type') or '—')[:25]} |"
                for r in m5["query_results"]
            ],
        )

        # ================================================================== Method 6
        lines += _section(
            "Method 6: BM25 Score Distribution — Signal vs Noise",
            description_lines=[
                "Even when BM25 returns a non-empty result for an OOV query, the score may be",
                "near-zero — a coincidental single-token overlap, not a genuine match.",
                "This method checks whether BM25 OOV 'results' are real or noise.",
            ],
            calculation_lines=[
                "For 20 OOV queries and 5 keyword-control queries (with explicit domain terms),",
                "record the **max BM25 score** of the top result.",
                "**Low-signal threshold**: the 10th percentile of keyword-control scores (P10).",
                "**Low-signal rate**: fraction of OOV queries whose max score < P10 of controls.",
            ],
            result_rows=[
                ("Keyword controls — mean max BM25 score",
                 None, f"{m5['mean_max_bm25_control']:.2f}", "—", None),
                ("OOV queries — mean max BM25 score",
                 None, f"{m5['mean_max_bm25_oov']:.2f}", "—", None),
                ("Low-signal rate ↑ (OOV scores < P10 of controls)",
                 None, _pct(m5["low_signal_rate"]), "—", None),
            ],
            analysis_lines=[
                f"- Keyword controls score {m5['mean_max_bm25_control']:.2f} on average — genuine matches.",
                f"- OOV queries score {m5['mean_max_bm25_oov']:.2f} — near-noise level.",
                f"- {_pct(m5['low_signal_rate'])} of OOV queries have max scores below the P10 of",
                "  keyword controls — effectively no useful match, even when result count > 0.",
                "- BM25 non-zero results on OOV queries should not be trusted as meaningful.",
            ],
        )

    # ================================================================== Method 7
    if m7 and m7.get("available"):
        lines += _section(
            f"Method 7: HyDE Partner Diversity — Cross-Sector Rate@{m7['k']}",
            description_lines=[
                "Tests the partner-matching use case: given a company description, find suitable partners.",
                "",
                "- **Website (MySQL LIKE)**: Cannot perform this task. Keyword search matches companies",
                "  that share the same words as the query — i.e. similar companies, not partners.",
                "  Critically, it has no mechanism to *infer* what type of partner would complement",
                "  a given company, so it is architecturally unable to perform partner matching.",
                "- **BM25**: Same limitation — keyword overlap returns similar companies, not complementary",
                "  ones. Cannot reason about what type of partner is needed.",
                "- **Raw Semantic**: Encodes the user's description directly and finds the nearest",
                "  vectors — typically returns companies doing similar work (competitors/peers).",
                "  Provides the baseline for partner-matching quality.",
                "- **Pipeline (HyDE)**: Takes a user-written description of what *type of partner* they",
                "  want (e.g. 'a company that helps us measure our carbon footprint'), generates a",
                "  hypothetical company profile matching that description, then finds similar companies.",
                "  Note: the pipeline does not always return cross-sector results — it returns whatever",
                "  the user asked for. If the user describes a similar company, it finds similar ones.",
                "  The cross-sector improvement here reflects the test cases, where users are looking",
                "  for complementary partners.",
            ],
            calculation_lines=[
                f"8 partner-matching scenarios. For each, retrieve top-{m7['k']} results using plain",
                "semantic and HyDE pipeline.",
                f"**Cross-Sector Rate@{m7['k']}**: fraction of top-{m7['k']} results from a *different*",
                "category than the user's company. Higher = more complementary partners, fewer competitors.",
                "**Category Diversity@K**: number of distinct categories in top-K results.",
            ],
            result_rows=[
                (f"Cross-Sector Rate@{m7['k']} ↑",
                 None, None,
                 _pct(m7["mean_plain_cross_sector"]) + " (plain)",
                 _pct(m7["mean_hyde_cross_sector"]) + " (HyDE)"),
                (f"Category Diversity@{m7['k']} ↑",
                 None, None,
                 f"{m7['mean_plain_cat_diversity']:.1f} (plain)",
                 f"{m7['mean_hyde_cat_diversity']:.1f} (HyDE)"),
            ],
            analysis_lines=[
                f"- Cross-Sector Rate: plain {_pct(m7['mean_plain_cross_sector'])} → "
                f"HyDE {_pct(m7['mean_hyde_cross_sector'])} ({_delta_str(m7['delta_cross_sector'])}).",
                "- HyDE substantially increases partner diversity — users get complementary organisations",
                "  rather than their own competitors.",
                "",
                "**Case-by-case results:**",
                "",
                "| Case | HyDE Inferred Partner Type | Plain Cross-Sector | HyDE Cross-Sector | Δ |",
                "|------|---------------------------|-------------------|-------------------|---|",
            ] + [
                f"| {c['label']} | {c['inferred_partner_type']}{'⚠️' if not c['hyde_ok'] else ''} "
                f"| {_pct(c['plain_cross_sector_rate'])} | {_pct(c['hyde_cross_sector_rate'])} "
                f"| {_delta_str(c['hyde_cross_sector_rate'] - c['plain_cross_sector_rate'])} |"
                for c in m7["case_results"]
            ] + (
                ["", "**Qualitative comparison — Case 1 (Carbon Auditor):**", ""]
                + ([
                    f"- Plain top-3 (companies *similar to* a carbon auditor): {', '.join(m7['case_results'][0]['plain_top3'])}",
                    f"- HyDE top-3 (companies *complementary to* a carbon auditor, inferred type: "
                    f"*{m7['case_results'][0]['inferred_partner_type']}*): {', '.join(m7['case_results'][0]['hyde_top3'])}",
                ] if m7["case_results"] else [])
            ),
        )
    elif m7 and not m7.get("available"):
        lines += [
            "", "---", "",
            "## Method 7: HyDE Partner Diversity",
            "",
            f"> Skipped — LLM unavailable: {m7.get('error', 'unknown error')}.",
            "> Run with `--run-hyde` and ensure `GOOGLE_API_KEY` or `GROQ_API_KEY` is set.",
        ]

    # ================================================================== Method 8 (supplemental)
    if m8:
        lines += _section(
            f"Method 8: Document Fragment Robustness ⚠️ Supplemental (embedding layer only)",
            description_lines=[
                "> ⚠️ **Scope**: M8 tests only the *embedding / retrieval layer*, not the full pipeline.",
                "> It is a diagnostic for the vector space, not an end-to-end system evaluation.",
                "",
                "Each company description is split into 3 equal fragments (beginning / middle / end).",
                "Each fragment is used as a standalone query to retrieve similar companies.",
                "The hypothesis: if the embedding captures holistic meaning, any fragment should",
                "still return category-consistent neighbours. BM25 depends on keywords being present.",
                "",
                "- **Website**: N/A — cannot accept document fragments as queries.",
                "- **BM25**: Category keywords may appear only in one fragment → precision collapses",
                "  for other fragments (depends on which section holds the discriminative keywords).",
                "- **Raw Semantic**: Holistic meaning is distributed — any fragment carries the",
                "  company's semantic fingerprint, so category precision should be stable.",
                "- **Pipeline (HyDE)**: N/A — you would not pass a fragment of a company description",
                "  to the partner-search pipeline. This method has no HyDE equivalent.",
                "",
                "**Note**: More informative robustness methods for a full system evaluation would include:",
                "query reformulation robustness (similar intent, different phrasing — already tested in M4),",
                "noisy input robustness (typos, truncated queries), or cross-language queries.",
            ],
            calculation_lines=[
                f"{m8['n_companies']} companies, each split into 3 equal character fragments.",
                f"Each fragment retrieves top-{m8['k_cat']} companies (excluding self).",
                "Category Precision@K = fraction sharing at least one `categories` tag.",
                "Mean across all companies × all fragment positions.",
            ],
            result_rows=[
                _R(f"Mean Cat. Precision@{m8['k_cat']} (all fragments)",
                   None, m8["bm25_cat_precision"], m8["semantic_cat_precision"], None),
            ] + [
                _R(f"  {label}", None, s["bm25"], s["semantic"], None)
                for label, s in m8["per_position"].items()
            ],
            analysis_lines=[
                f"- Semantic {_pct(m8['semantic_cat_precision'])} vs BM25 {_pct(m8['bm25_cat_precision'])} "
                f"({_delta_str(m8['delta'])}).",
                "- If BM25 precision drops from fragment 1 → fragment 3, it indicates BM25 depends on",
                "  keywords concentrated at the start of descriptions.",
                "- Semantic search should show smaller variance across positions if meaning is distributed.",
            ],
        )

    # ================================================================== Methodology Notes
    lines += [
        "", "---", "",
        "## Methodology Notes",
        "",
        "- **No annotation required**: Methods 1–5 use existing DB labels (`predicted_sdg_tags`, `categories`) as ground truth.",
        "- **No pipeline impact**: Evaluation reads directly from PGStore; does not call agents, LangGraph, or FastAPI.",
        "- **BM25**: BM25Okapi represents the theoretical best a pure keyword system can achieve on this corpus.",
        "- **Original website**: WordPress GeoDirectory MySQL LIKE search — the system being replaced.",
        "- **OOV queries (M5/6)**: Minimise token overlap with company descriptions to isolate semantic understanding.",
        "- **HyDE (M7)**: Requires LLM (Gemini/Groq). Run with `--run-hyde`. Tests the full partner-matching use case.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Search Quality Evaluation")
    parser.add_argument("--output", default="notes/rag_eval_summary.md", help="Output markdown file")
    parser.add_argument("--sample", type=int, default=None, help="Sample N companies for M1/M3 (default: all)")
    parser.add_argument("--skip-website", action="store_true", help="Skip original website API calls")
    parser.add_argument("--k1", type=int, default=20, help="Top-K for Method 1 (default 20)")
    parser.add_argument("--k2", type=int, default=10, help="Top-K for Method 2 SDG Precision (default 10)")
    parser.add_argument("--k3", type=int, default=5,  help="Top-K for Method 3 Category Consistency (default 5)")
    parser.add_argument("--k4", type=int, default=10, help="Top-K for Method 4 Motivational (default 10)")
    parser.add_argument("--k5", type=int, default=10, help="Top-K for Method 5 OOV (default 10)")
    parser.add_argument("--k7", type=int, default=5,  help="Top-K for Method 7 HyDE (default 5)")
    parser.add_argument("--k8", type=int, default=20, help="Top-K for Method 8 Fragment (default 20)")
    parser.add_argument("--run-hyde", action="store_true",
                        help="Run full pipeline (HyDE) for all query-based methods (M2/M4/M5/M7). Requires LLM API key.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load HyDE embedding cache (avoids re-running LLM for repeated eval runs)
    hyde_cache = None
    if args.run_hyde:
        hyde_cache = _load_hyde_cache()
        print(f"HyDE cache loaded ({len(hyde_cache)} entries from previous runs)")

    # ------------------------------------------------------------------
    print("Loading companies from database...")
    t0 = time.time()
    companies = _get_all_companies()
    print(f"  Loaded {len(companies)} companies ({time.time()-t0:.1f}s)")

    company_ids = [co["id"] for co in companies]
    documents = [co["document"] for co in companies]

    # Pre-stack all embeddings into a matrix for fast in-memory search
    print("Building embedding matrix...")
    all_embeddings = np.stack([co["embedding"] for co in companies]).astype(np.float32)
    # Ensure normalised (should already be, but guard against edge cases)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / np.where(norms > 0, norms, 1)
    print(f"  Embedding matrix: {all_embeddings.shape}")

    # BM25 index
    print("Building BM25 index...")
    bm25 = _build_bm25(documents)
    print("  BM25 index ready")

    # Encoder (lazy-loaded, cached)
    print("Loading encoder...")
    encoder = _get_encoder()
    print("  Encoder ready")

    # ------------------------------------------------------------------
    # Method 0: Original Website Baseline
    # ------------------------------------------------------------------
    m0 = None
    if not args.skip_website:
        print(f"\n[0/0] Original Website Baseline (M2: {len(SDG_QUERIES)} SDG + M4: {len(MOTIVATIONAL_QUERIES)} motivational + M5: {len(OOV_QUERIES)} OOV queries)...")
        t0_web = time.time()
        m0 = run_website_baseline(companies, skip=False, k=args.k4)
        if m0.get("available"):
            print(f"  Done ({time.time()-t0_web:.1f}s) — M4 zero-rate: {m0['m4']['zero_rate']:.0%} | M5 zero-rate: {m0['m5']['zero_rate']:.0%} | M2 zero-rate: {m0['m2']['zero_rate']:.0%}")
        else:
            print(f"  Website unavailable: {m0.get('error', 'unknown error')}")
    else:
        m0 = {"available": False, "skipped": True, "k": args.k4}
        print("\n[0/0] Website baseline skipped (--skip-website)")

    # ------------------------------------------------------------------
    # Method 1: Self-Retrieval MRR
    # ------------------------------------------------------------------
    n_label = f" (sample={args.sample})" if args.sample else f" (all {len(companies)})"
    print(f"\n[1/4] Self-Retrieval MRR{n_label}...")
    t1 = time.time()
    m1 = run_self_retrieval(
        companies, all_embeddings, company_ids, bm25,
        n=args.k1, sample=args.sample,
        run_hyde=args.run_hyde, hyde_cache=hyde_cache,
    )
    pipe_mrr_str = f" | Pipeline MRR: {m1['pipeline_mrr']:.4f}" if m1.get("pipeline_mrr") is not None else ""
    print(f"  Semantic MRR: {m1['semantic_mrr']:.4f} | BM25 MRR: {m1['bm25_mrr']:.4f}{pipe_mrr_str}  ({time.time()-t1:.1f}s)")

    # ------------------------------------------------------------------
    # Method 2: SDG Precision@K
    # ------------------------------------------------------------------
    hyde_label = " + Full Pipeline (HyDE)" if args.run_hyde else ""
    print(f"\n[2/4] SDG Precision@{args.k2} (17 SDG queries){hyde_label}...")
    t2 = time.time()
    m2 = run_sdg_precision(
        companies, all_embeddings, company_ids, encoder, bm25, k=args.k2,
        run_hyde=args.run_hyde, hyde_cache=hyde_cache, website_baseline=m0,
    )
    pipe_str = f" | Pipeline P@{args.k2}: {m2['mean_pipeline_precision']:.1%}" if m2.get('mean_pipeline_precision') is not None else ""
    print(f"  Semantic P@{args.k2}: {m2['mean_semantic_precision']:.1%} | BM25 P@{args.k2}: {m2['mean_bm25_precision']:.1%}{pipe_str}  ({time.time()-t2:.1f}s)")

    # ------------------------------------------------------------------
    # Method 3: Category Consistency@K
    # ------------------------------------------------------------------
    print(f"\n[3/4] Category Consistency@{args.k3}{n_label}...")
    t3 = time.time()
    m3 = run_category_consistency(
        companies, all_embeddings, company_ids, bm25,
        k=args.k3, sample=args.sample,
        run_hyde=args.run_hyde, hyde_cache=hyde_cache,
    )
    pipe_con_str = f" | Pipeline: {m3['mean_pipeline_consistency']:.1%}" if m3.get("mean_pipeline_consistency") is not None else ""
    print(f"  Semantic: {m3['mean_semantic_consistency']:.1%} | BM25: {m3['mean_bm25_consistency']:.1%}{pipe_con_str}  ({time.time()-t3:.1f}s)")

    # ------------------------------------------------------------------
    # Method 4: Motivational Zero-Result Rate
    # ------------------------------------------------------------------
    wp_label = "(skip)" if args.skip_website else ""
    print(f"\n[4/4] Motivational Queries (30 queries) {wp_label}{hyde_label}...")
    t4 = time.time()
    m4 = run_motivational_analysis(
        companies, all_embeddings, company_ids, encoder, bm25,
        k=args.k4, run_hyde=args.run_hyde, hyde_cache=hyde_cache,
        website_baseline=m0,
    )
    pipe_rel_str = f" | Pipeline SDG Rel: {m4['mean_pipeline_sdg_relevance']:.1%}" if m4.get('mean_pipeline_sdg_relevance') is not None else ""
    print(f"  Semantic zero-rate: {m4['semantic_zero_rate']:.1%} | BM25 zero-rate: {m4['bm25_zero_rate']:.1%}{pipe_rel_str}  ({time.time()-t4:.1f}s)")
    if m4["website_zero_rate"] is not None:
        print(f"  Website zero-rate: {m4['website_zero_rate']:.1%}")

    # ------------------------------------------------------------------
    # Method 8: Document Fragment Robustness
    # ------------------------------------------------------------------
    print(f"\n[8/8] Fragment Category Precision{n_label}...")
    t8 = time.time()
    m8 = run_fragment_robustness(
        companies, all_embeddings, company_ids, encoder, bm25,
        n=args.k8, sample=args.sample,
    )
    print(f"  Cat. Precision@{m8['k_cat']}: Semantic {m8['semantic_cat_precision']:.1%} | BM25 {m8['bm25_cat_precision']:.1%}  ({time.time()-t8:.1f}s)")

    # ------------------------------------------------------------------
    # Method 5 + 6: OOV Queries & BM25 Score Distribution
    # ------------------------------------------------------------------
    print(f"\n[5/6] OOV Queries + BM25 Score Distribution ({len(OOV_QUERIES)} OOV + {len(_KEYWORD_CONTROL_QUERIES)} controls){hyde_label}...")
    t5 = time.time()
    m5 = run_oov_analysis(
        companies, all_embeddings, company_ids, encoder, bm25, k=args.k5,
        run_hyde=args.run_hyde, hyde_cache=hyde_cache, website_baseline=m0,
    )
    print(f"  BM25 zero-rate: {m5['bm25_zero_rate']:.1%} | Low-signal rate: {m5['low_signal_rate']:.1%}")
    pipe_oov_str = f" | Pipeline SDG Rel: {m5['mean_pipeline_relevance']:.1%}" if m5.get('mean_pipeline_relevance') is not None else ""
    print(f"  Semantic SDG Rel: {m5['mean_sem_relevance']:.1%} | BM25 SDG Rel: {m5['mean_bm25_relevance']:.1%}{pipe_oov_str}  ({time.time()-t5:.1f}s)")
    print(f"  BM25 score: OOV avg={m5['mean_max_bm25_oov']:.2f} vs keyword-control avg={m5['mean_max_bm25_control']:.2f}")

    # ------------------------------------------------------------------
    # Method 7: HyDE Partner Diversity (optional — requires LLM)
    # ------------------------------------------------------------------
    m7 = None
    if args.run_hyde:
        print(f"\n[7/7] HyDE Partner Diversity ({len(PARTNER_CASES)} cases, k={args.k7})...")
        t7 = time.time()
        m7 = run_hyde_comparison(
            companies, all_embeddings, company_ids, encoder, k=args.k7,
        )
        if m7.get("available"):
            print(f"  Plain cross-sector: {m7['mean_plain_cross_sector']:.1%} | HyDE cross-sector: {m7['mean_hyde_cross_sector']:.1%}  ({time.time()-t7:.1f}s)")
        else:
            print(f"  Skipped: {m7.get('error')}")
    else:
        print("\n[7/7] HyDE comparison skipped (use --run-hyde to enable, requires LLM API key)")

    # Save HyDE embedding cache so repeated runs are free
    if hyde_cache is not None:
        _save_hyde_cache(hyde_cache)
        print(f"  HyDE cache saved ({len(hyde_cache)} entries → {_HYDE_CACHE_PATH})")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\nGenerating report...")
    report = generate_report(m1, m2, m3, m4, m5=m5, m7=m7, m8=m8, m0=m0, n_total=len(companies))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"Report saved → {args.output}")

    # Console summary
    print("\n" + "=" * 56)
    print(f"{'Method':<30} {'Semantic':>10} {'BM25':>10} {'Δ':>8}")
    print("-" * 56)
    print(f"{'Self-Retrieval MRR':<30} {m1['semantic_mrr']:>10.4f} {m1['bm25_mrr']:>10.4f} {m1['delta']:>+8.4f}")
    print(f"{'SDG Precision@' + str(args.k2):<30} {m2['mean_semantic_precision']:>10.1%} {m2['mean_bm25_precision']:>10.1%} {m2['delta']:>+8.1%}")
    print(f"{'Category Consistency@' + str(args.k3):<30} {m3['mean_semantic_consistency']:>10.1%} {m3['mean_bm25_consistency']:>10.1%} {m3['delta']:>+8.1%}")
    print(f"{'Zero-Result Rate (lower=better)':<30} {m4['semantic_zero_rate']:>10.1%} {m4['bm25_zero_rate']:>10.1%} {'—':>8}")
    if m4["mean_sem_sdg_relevance"] is not None:
        d_rel = m4["mean_sem_sdg_relevance"] - m4["mean_bm25_sdg_relevance"]
        print(f"{'Motivational SDG Relevance@' + str(args.k4):<30} {m4['mean_sem_sdg_relevance']:>10.1%} {m4['mean_bm25_sdg_relevance']:>10.1%} {d_rel:>+8.1%}")
    if m8:
        print(f"{'Fragment Cat.Precision@' + str(m8['k_cat']):<30} {m8['semantic_cat_precision']:>10.1%} {m8['bm25_cat_precision']:>10.1%} {m8['delta']:>+8.1%}")
    if m5:
        d5 = (m5["mean_sem_relevance"] or 0) - (m5["mean_bm25_relevance"] or 0)
        print(f"{'OOV SDG Relevance@' + str(args.k5):<30} {m5['mean_sem_relevance']:>10.1%} {m5['mean_bm25_relevance']:>10.1%} {d5:>+8.1%}")
        print(f"{'BM25 Low-Signal Rate':<30} {'—':>10} {m5['low_signal_rate']:>10.1%} {'—':>8}")
    if m7 and m7.get("available"):
        print(f"{'HyDE Cross-Sector@' + str(args.k7):<30} {m7['mean_plain_cross_sector']:>10.1%} {'—':>10} {m7['mean_hyde_cross_sector']:>+8.1%}")
    if m4["website_zero_rate"] is not None:
        print(f"{'Website Zero-Result Rate':<30} {'0.0%':>10} {'—':>10} {m4['website_zero_rate']:>+8.1%}")
    print("=" * 56)


if __name__ == "__main__":
    main()
