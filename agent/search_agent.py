"""
SearchAgent — Step 1 implementation.

Responsibilities:
  1. HyDE: LLM generates a hypothetical ideal partner description (50-120 words)
           + Query Expansion (3-5 equivalent search phrases)
  2. Embed: encode all generated texts → average into a single query vector
  3. Search: semantic_search_from_embedding → Top-10 candidate companies
  4. Write: hypothetical_partner_desc, query_expansions, candidate_companies → AgentState

Step 2 upgrade path:
  - Add sql_filter and hybrid_search tools with @tool decorators
  - Replace direct function calls with LLM-based tool routing
  - Add three-level fallback logic

References:
  HyDE: Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels"
"""

import json
import logging
import os
import re
from typing import Optional

import numpy as np

from agent.state import AgentState
from agent.llm import get_llm
from agent.tools import (
    semantic_search_from_embedding,
    sql_filter,
    hybrid_search,
    _get_encoder,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HyDE prompt
# ---------------------------------------------------------------------------

_HYDE_SYSTEM = """You are an expert at finding ideal business partners from a sustainability-focused directory (SDGZero).

Your job is to help identify the perfect partner for a given company by:
1. Writing a vivid description of an ideal partner company (as if it's a real company profile)
2. Generating multiple search phrases to find such companies

Write the partner description in the same style as a real company directory listing:
concrete, specific, mentioning what they DO, their sector, SDG focus, and business model.
Do NOT be generic. Imagine a real company that would be a great fit.

Respond ONLY with valid JSON. No explanation, no markdown.
"""

_HYDE_HUMAN = """My company:
{user_company_desc}
{extra}
{filter_context}
{partner_type_instruction}Generate a JSON response with exactly these two keys:
{{
  "partner_description": "<50-120 word description of an ideal partner, written like a real company profile>",
  "query_expansions": ["<phrase 1>", "<phrase 2>", "<phrase 3>", "<phrase 4>", "<phrase 5>"]
}}

The partner_description should describe ONE specific ideal partner company.
The query_expansions should be 3-5 different search angles (e.g. by sector, by SDG, by business model).
"""


# ---------------------------------------------------------------------------
# HyDE generation
# ---------------------------------------------------------------------------

def _run_hyde(
    user_company_desc: str,
    other_requirements: str = "",
    partner_type_desc: str = "",
    filters: Optional[dict] = None,
) -> tuple[str, list[str]]:
    """
    Call LLM to generate a hypothetical partner description + query expansions.

    Args:
        user_company_desc:   User's own company description.
        other_requirements:  Optional free-text additional criteria.
        partner_type_desc:   Optional explicit description of the target company type.
                             When provided, strongly steers the generated profile.
        filters:             Optional SearchFilters dict — SDG tags and categories
                             are injected as context so HyDE stays on-topic.

    Returns:
        (partner_description, query_expansions)
    """
    llm = get_llm()

    if partner_type_desc.strip():
        partner_type_instruction = (
            f"IMPORTANT: The ideal partner MUST BE a {partner_type_desc.strip()}. "
            f"Write the partner_description as a profile of that type of company — "
            f"NOT another company like mine.\n\n"
        )
    else:
        partner_type_instruction = ""

    # Inject filter context so HyDE stays aligned with the user's explicit criteria.
    # E.g. sdg_tags=["Quality Education"] → tell HyDE the partner should focus on
    # education, preventing it from drifting to an unrelated sector.
    filter_hints = []
    if filters:
        if filters.get("sdg_tags"):
            filter_hints.append(f"SDG focus: {', '.join(filters['sdg_tags'])}")
        if filters.get("categories"):
            cats = filters["categories"]
            filter_hints.append(f"Sector: {cats if isinstance(cats, str) else ', '.join(cats)}")
        if filters.get("city"):
            filter_hints.append(f"Location: {filters['city']}")
        if filters.get("job_sector"):
            filter_hints.append(f"Job sector: {filters['job_sector']}")
    filter_context = (
        f"\nSearch filters (the ideal partner should match these):\n"
        + "\n".join(f"  - {h}" for h in filter_hints) + "\n"
        if filter_hints else ""
    )

    # other_requirements intentionally excluded from HyDE:
    # it is a soft preference ("cares about nature") that should not steer the
    # vector search direction — only partner_type_desc and filters do that.
    # It is used in reasoning (scoring_agent) to inform the recommendation text.
    extra = ""
    prompt = _HYDE_HUMAN.format(
        user_company_desc=user_company_desc,
        partner_type_instruction=partner_type_instruction,
        filter_context=filter_context,
        extra=extra,
    )

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=_HYDE_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code blocks if LLM wraps in ```json ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        partner_desc = data.get("partner_description", "").strip()
        expansions = data.get("query_expansions", [])

        if not partner_desc:
            raise ValueError("Empty partner_description in LLM response")
        if not expansions:
            raise ValueError("Empty query_expansions in LLM response")

        # Ensure list of strings
        expansions = [str(e).strip() for e in expansions if str(e).strip()][:5]
        return partner_desc, expansions

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"HyDE JSON parse failed ({e}). Falling back to raw text.")
        # Graceful fallback: use raw response as the description, no expansions
        fallback_desc = raw[:500] if raw else user_company_desc
        return fallback_desc, []


# ---------------------------------------------------------------------------
# Embedding averaging
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _meaningful_filters(filters: dict) -> dict:
    """Return only the filter keys that have non-empty values."""
    return {k: v for k, v in filters.items() if v not in (None, "", [], False)}


def _relax_filters(filters: dict, level: int) -> dict:
    """
    Return a relaxed copy of the filters dict.

    Level 1: drop SDG tags and claimed (strictest conditions).
    Level 2: drop everything (handled by caller switching to pure semantic).
    """
    relaxed = dict(filters)
    if level >= 1:
        relaxed.pop("sdg_tags", None)
        relaxed.pop("claimed", None)
    return relaxed


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _averaged_embedding(texts: list[str]) -> list[float]:
    """
    Encode multiple texts and return their averaged, normalised embedding.

    Combining hypothetical_partner_desc + query_expansions into one vector
    captures multiple semantic angles simultaneously.

    Args:
        texts: List of strings to encode and average.

    Returns:
        384-dim list[float] (L2-normalised).
    """
    encoder = _get_encoder()
    embeddings = encoder.encode(texts, normalize_embeddings=True)  # (N, 384)
    avg = embeddings.mean(axis=0)
    # Re-normalise after averaging
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg.tolist()


# ---------------------------------------------------------------------------
# SearchAgent LangGraph node
# ---------------------------------------------------------------------------

def search_agent_node(state: AgentState) -> dict:
    """
    SearchAgent node for the LangGraph pipeline.

    Reads:
        user_company_desc, other_requirements, filters (from state)

    Writes:
        hypothetical_partner_desc, query_expansions, candidate_companies,
        search_method, search_fallback_level

    Step 2 will add:
        - Tool routing: sql_filter if only filters, hybrid_search if both
        - Three-level fallback logic
        - Schema injection (city/category/SDG valid values from Redis cache)
    """
    user_desc        = state.get("user_company_desc", "")
    other_req        = state.get("other_requirements", "")
    partner_type     = state.get("partner_type_desc", "")
    filters          = state.get("filters", {})

    has_desc = bool(user_desc.strip())
    has_filters = bool(_meaningful_filters(filters))

    if not has_desc and not has_filters:
        logger.error("SearchAgent: both user_company_desc and filters are empty")
        return {
            "candidate_companies": [],
            "errors": (state.get("errors") or []) + ["SearchAgent: provide a description or at least one filter"],
        }

    # ------------------------------------------------------------------
    # Step 1a: HyDE — only when we have a description to work with
    # ------------------------------------------------------------------
    partner_desc = ""
    expansions = []
    avg_embedding = [0.0] * 384   # placeholder; overwritten if has_desc

    if has_desc:
        logger.info("SearchAgent: running HyDE...")
        partner_desc, expansions = _run_hyde(user_desc, other_req, partner_type, filters)
        logger.info(f"SearchAgent: HyDE done. expansions={len(expansions)}")
        logger.info(f"SearchAgent: HyDE content → {partner_desc[:200]}")

        # ------------------------------------------------------------------
        # Step 1b: Embed — build search vector
        #
        # When partner_type_desc is explicit, use partner_desc ONLY (no
        # expansion averaging). Expansions tend to inject sustainability/SDG
        # vocabulary that dilutes the signal and pulls toward wrong sectors.
        # When no partner_type is given, averaging expansions broadens recall
        # which is helpful for open-ended queries.
        # ------------------------------------------------------------------
        if partner_type:
            all_texts = [partner_desc]
            logger.info("SearchAgent: partner_type provided — using HyDE only (no expansion averaging)")
        else:
            all_texts = [partner_desc] + expansions
        avg_embedding = _averaged_embedding(all_texts)
        logger.info(f"SearchAgent: averaged {len(all_texts)} embeddings → 384-dim vector")
    else:
        logger.info("SearchAgent: no description — skipping HyDE, using sql_filter only")

    # ------------------------------------------------------------------
    # Step 1c: Tool routing — choose search method based on inputs
    #
    # Rule-based (deterministic, no LLM cost):
    #   has desc + has filters → hybrid_search  (vector + SQL conditions)
    #   has desc only          → semantic_search (pure vector)
    #   has filters only       → sql_filter      (pure metadata match)
    #
    # Three-level fallback for hybrid/sql when results are too few:
    #   Level 0: full conditions                    → return if enough results
    #   Level 1: drop SDG + claimed (most strict)   → retry
    #   Level 2: pure semantic, all filters dropped → always returns results
    # ------------------------------------------------------------------
    MIN_RESULTS = 5
    MIN_SIMILARITY = 0.35   # below this, filters are hurting more than helping

    candidates = []
    method = "semantic"
    fallback_level = 0

    if has_desc and has_filters:
        # hybrid: vector + metadata
        method = "hybrid"
        candidates = hybrid_search(avg_embedding, filters, n_results=10)
        logger.info(f"SearchAgent: hybrid_search → {len(candidates)} results (level 0)")

        # Check similarity quality — if filters returned enough candidates but
        # none are semantically relevant, they filtered INTO noise rather than
        # filtering OUT noise. Fall back to pure semantic immediately.
        best_sim = max((c.get("similarity", 0) for c in candidates), default=0)
        if candidates and best_sim < MIN_SIMILARITY:
            candidates = semantic_search_from_embedding(avg_embedding, n_results=10)
            fallback_level = 2
            method = "semantic"
            logger.info(
                f"SearchAgent: hybrid best_sim={best_sim:.3f} < {MIN_SIMILARITY} "
                f"→ quality fallback to pure semantic (level 2)"
            )
        elif len(candidates) < MIN_RESULTS:
            relaxed = _relax_filters(filters, level=1)
            candidates = hybrid_search(avg_embedding, relaxed, n_results=10)
            fallback_level = 1
            logger.info(f"SearchAgent: hybrid relaxed → {len(candidates)} results (level 1)")

            if len(candidates) < MIN_RESULTS:
                candidates = semantic_search_from_embedding(avg_embedding, n_results=10)
                fallback_level = 2
                method = "semantic"
                logger.info(f"SearchAgent: fallback to pure semantic → {len(candidates)} results (level 2)")

    elif has_filters and not has_desc:
        # pure filter: no vector search
        method = "sql"
        candidates = sql_filter(filters, n_results=10)
        logger.info(f"SearchAgent: sql_filter → {len(candidates)} results (level 0)")

        if len(candidates) < MIN_RESULTS:
            relaxed = _relax_filters(filters, level=1)
            candidates = sql_filter(relaxed, n_results=10)
            fallback_level = 1
            logger.info(f"SearchAgent: sql relaxed → {len(candidates)} results (level 1)")

        if len(candidates) < MIN_RESULTS:
            candidates = semantic_search_from_embedding(avg_embedding, n_results=10)
            fallback_level = 2
            method = "semantic"
            logger.info(f"SearchAgent: fallback to pure semantic → {len(candidates)} results (level 2)")

    else:
        # pure semantic: no filters
        candidates = semantic_search_from_embedding(avg_embedding, n_results=10)
        logger.info(f"SearchAgent: semantic_search → {len(candidates)} results")

    logger.info(f"SearchAgent: final → {len(candidates)} candidates via {method} (fallback={fallback_level})")

    return {
        "hypothetical_partner_desc": partner_desc,
        "query_expansions": expansions,
        "candidate_companies": candidates,
        "search_method": method,
        "search_fallback_level": fallback_level,
    }
