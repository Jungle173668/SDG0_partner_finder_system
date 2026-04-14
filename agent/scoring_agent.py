"""
ScoringAgent — Step 4 implementation.

Responsibilities:
  1. Cross-encoder rerank: score each of the Top-10 candidates against the
     user's original company description, produce a ranked list.
  2. Select Top-5 by cross-encoder score.
  3. LLM reasoning: for each Top-5, generate a recommendation paragraph using
     the full research summary (DB text + Tavily web content).
  4. Assign a match_quality label per company for Result Transparency.

Two-step design rationale:
  - Cross-encoder query   = user_company_desc  (original input, unbiased anchor)
    Cross-encoder document = company.document   (DB text only, ~360 tokens,
    fits within the 512-token cross-encoder limit cleanly)
  - LLM reasoning uses research_results[slug]["summary"] (up to 4 000 chars,
    DB + Tavily). Tavily's real value is in generating compelling recommendation
    text with recent context — not in ranking where it would be truncated anyway.

Query anchor choice:
  hypothetical_partner_desc (HyDE) is used as the cross-encoder query.
  HyDE generates a company-profile-style description that matches the vocabulary
  and format of DB documents, allowing ms-marco to score relevance properly.
  Using user_company_desc (casual text) gives scores near 0.001 — unusable.
  Using HyDE gives scores in the 0.80–0.96 range with clear discrimination.

References:
  Cross-encoder reranking: Reimers & Gurevych 2019 (SBERT);
  Nogueira & Cho 2019 (MonoBERT reranking).
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np

from agent.state import AgentState
from agent.llm import get_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_TOP_N = 5              # final Top-N to keep after reranking

# Sigmoid thresholds for match_quality labels (see _assign_match_quality)
_SCORE_STRONG  = 0.70   # >= 70% → "strong"
_SCORE_PARTIAL = 0.20   # >= 20% → "partial"; below → "fallback"


# ---------------------------------------------------------------------------
# Cross-encoder singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazy-load CrossEncoder singleton (loaded once, reused across requests)."""
    from sentence_transformers import CrossEncoder
    model_name = os.getenv("CROSS_ENCODER_MODEL", _CROSS_ENCODER_MODEL)
    logger.info(f"ScoringAgent: loading CrossEncoder '{model_name}'...")
    return CrossEncoder(model_name)


# ---------------------------------------------------------------------------
# Step 1: Cross-encoder reranking helpers
# ---------------------------------------------------------------------------

def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Score each candidate against the query using a Cross-encoder and return
    candidates sorted by score descending.

    The cross-encoder sees both texts jointly (not independently), enabling
    full token-level interaction — higher precision than bi-encoder similarity.

    Args:
        query:      User's original company description (unbiased ranking anchor).
        candidates: Candidate company dicts from SearchAgent.

    Returns:
        Same dicts, sorted descending, with two new fields added:
            cross_encoder_score  — sigmoid-normalised score in [0, 1]
            cross_encoder_raw    — raw logit (useful for debugging)

    Note on truncation:
        The tokenizer automatically truncates to max_length=512 tokens if the
        combined (query + document) is too long. No manual char-slicing needed.
    """
    cross_encoder = _get_cross_encoder()

    # Build (query, document) pairs — full document text, no manual truncation.
    # CrossEncoder tokenizer handles truncation automatically (max_length=512),
    # cutting from the tail if needed. The DB document is structured
    # "name → categories → city → description → SDGs" so the most important
    # fields are at the front and survive any truncation.
    pairs = [
        (query, c.get("document") or "")
        for c in candidates
    ]

    # Batch predict — one forward pass for all pairs (faster than looping)
    raw_scores = cross_encoder.predict(pairs)           # numpy array, shape (N,)
    sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores)) # sigmoid → [0, 1]

    scored = []
    for company, raw, sig in zip(candidates, raw_scores.tolist(), sigmoid_scores.tolist()):
        scored.append({
            **company,
            "cross_encoder_score": round(sig, 4),
            "cross_encoder_raw":   round(raw, 4),
        })

    scored.sort(key=lambda c: c["cross_encoder_score"], reverse=True)
    return scored


def _assign_match_quality(score: float, fallback_level: int, anchor: str = "HyDE") -> str:
    """
    Derive a Result Transparency label based on cross-encoder score only.

    Labels:
        strong   — score >= 70%
        partial  — score >= 20%
        fallback — score < 20%

    Args:
        score:          sigmoid cross_encoder_score in [0, 1].
        fallback_level: unused here; kept for signature compatibility.
        anchor:         unused here; kept for signature compatibility.

    Returns:
        "strong" | "partial" | "fallback"
    """
    if score >= _SCORE_STRONG:
        return "strong"
    if score >= _SCORE_PARTIAL:
        return "partial"
    return "fallback"


# ---------------------------------------------------------------------------
# Step 2: LLM reasoning helpers
# ---------------------------------------------------------------------------

_REASONING_SYSTEM = """\
You are a business partnership analyst for SDGZero, a sustainability-focused business directory.

Your task: write a concise recommendation explaining WHY a specific candidate company is a strong \
partner match for the USER'S COMPANY, and suggest concrete ways to start the collaboration.

Rules:
- The user is a BUSINESS OWNER — always write from their perspective, addressing them as "you"
- The reasoning explains why the CANDIDATE is a good fit FOR THE USER — not the other way around
- Ground every claim in details from the company profile or web research provided
- Highlight complementary services, shared SDG focus, or compatible business models
- Weave in recent news or projects from web research if available (makes it more compelling)
- 3-4 sentences for the reasoning — professional but approachable tone
- The "additional requirements" field describes what the user WANTS IN A PARTNER — do NOT use it to characterise the user's own company
- Do NOT open with "As a [description of user]..." — go straight to why the candidate is a good fit
- Respond ONLY with valid JSON. No explanation, no markdown wrapping.
"""

_REASONING_HUMAN = """\
My company (the user seeking a partner):
{user_company_desc}
{extra_requirements}
{soft_filter_context}
Candidate partner to evaluate:
{company_profile}

Respond with exactly this JSON structure:
{{
  "reasoning": "<3-4 sentence recommendation explaining why this candidate is a good partner FOR MY COMPANY>"
}}
"""


def _build_company_profile(company: dict, research_summary: str) -> str:
    """
    Assemble the company profile string passed to the LLM reasoning prompt.

    Combines a structured one-line header (for quick orientation) with the full
    research_results summary (DB text + Tavily web content, up to 4 000 chars).
    The LLM uses this rich context to generate specific, well-grounded reasoning.
    """
    name       = company.get("name", "Unknown")
    city       = company.get("city", "")
    categories = company.get("categories", "")
    biz_type   = company.get("business_type", "")
    sdg_tags   = company.get("sdg_tags", "") or company.get("predicted_sdg_tags", "")

    parts = [f"Name: {name}"]
    if city:
        parts.append(f"Location: {city}")
    if categories:
        parts.append(f"Categories: {categories}")
    if biz_type:
        parts.append(f"Type: {biz_type}")
    if sdg_tags:
        parts.append(f"SDGs: {sdg_tags}")

    header = " | ".join(parts)

    if research_summary:
        return f"{header}\n\n{research_summary}"
    return header


def _check_soft_filters(company: dict, soft_filters: dict) -> list[str]:
    """
    Check which soft filter conditions a company satisfies.

    Returns a list of human-readable strings describing satisfied conditions.
    e.g. ["SDG 7: Affordable And Clean Energy", "City: London", "Type: B2B"]
    """
    hits = []
    if not soft_filters:
        return hits

    city = soft_filters.get("city")
    if city and company.get("city", "").lower() == city.lower():
        hits.append(f"City: {city}")

    biz_type = soft_filters.get("business_type")
    if biz_type and company.get("business_type", "").lower() == biz_type.lower():
        hits.append(f"Type: {biz_type}")

    job_sector = soft_filters.get("job_sector")
    if job_sector and company.get("job_sector", "").lower() == job_sector.lower():
        hits.append(f"Sector: {job_sector}")

    company_size = soft_filters.get("company_size")
    if company_size and company.get("company_size", "").lower() == company_size.lower():
        hits.append(f"Size: {company_size}")

    sdg_tags = soft_filters.get("sdg_tags", [])
    if sdg_tags:
        company_sdgs = (company.get("sdg_tags") or "") + " " + (company.get("predicted_sdg_tags") or "")
        for tag in sdg_tags:
            if tag.lower() in company_sdgs.lower():
                hits.append(f"SDG: {tag}")

    categories = soft_filters.get("categories", [])
    if categories:
        company_cats = company.get("categories") or ""
        for cat in categories:
            if cat.lower() in company_cats.lower():
                hits.append(f"Category: {cat}")

    return hits


def _run_reasoning(
    user_company_desc: str,
    other_requirements: str,
    soft_filters: dict,
    company: dict,
    research_summary: str,
) -> str:
    """
    Call LLM to generate a partnership recommendation for one company.

    Args:
        user_company_desc:  User's original company description.
        other_requirements: Free-text additional criteria from the user.
        company:            Scored company dict (includes cross_encoder_score).
        research_summary:   Full summary from ResearchAgent (DB + Tavily, ≤4000 chars).

    Returns:
        reasoning_text — Falls back to a plain-text snippet on parse failure.
    """
    llm = get_llm()

    extra = f"\nAdditional requirements for the ideal partner (NOT about my company): {other_requirements}" if other_requirements.strip() else ""
    company_profile = _build_company_profile(company, research_summary)

    # Build soft filter context hint for LLM
    soft_hits = _check_soft_filters(company, soft_filters)
    if soft_hits and soft_filters:
        soft_ctx = "\nPreferred partner criteria (mention if relevant in your reasoning): " + "; ".join(
            f"{k}: {v}" for k, v in soft_filters.items() if v
        )
    else:
        soft_ctx = ""

    prompt = _REASONING_HUMAN.format(
        user_company_desc=user_company_desc,
        extra_requirements=extra,
        soft_filter_context=soft_ctx,
        company_profile=company_profile,
    )

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=_REASONING_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if LLM wraps output in ```json ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        reasoning = data.get("reasoning", "").strip()

        if not reasoning:
            raise ValueError("Empty reasoning field in LLM response")

        return reasoning

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"ScoringAgent: reasoning JSON parse failed ({e}). Using raw text.")
        return raw[:600] if raw else "No reasoning available."


# ---------------------------------------------------------------------------
# ScoringAgent LangGraph node
# ---------------------------------------------------------------------------

def scoring_agent_node(state: AgentState) -> dict:
    """
    ScoringAgent node for the LangGraph pipeline.

    Reads:
        user_company_desc         (user input)
        other_requirements        (user input)
        candidate_companies       (SearchAgent — Top-10)
        research_results          (ResearchAgent — DB + Tavily summaries)
        search_fallback_level     (SearchAgent — 0/1/2, used for match_quality)

    Writes:
        scored_companies — Top-5 dicts, each with:
            all candidate_companies fields
            cross_encoder_score  (float, sigmoid-normalised [0, 1])
            cross_encoder_raw    (float, raw logit, for debugging)
            match_quality        ("strong" | "partial" | "fallback")
            reasoning            (str, LLM-generated recommendation paragraph)
            entry_points         (list[str], concrete collaboration suggestions)
            soft_filter_hit      (list[str], Phase 3 — empty placeholder for now)

    Step 1 — Cross-encoder rerank (DB text only, fits 512-token limit):
        query    = user_company_desc  (original input, not HyDE — avoids drift)
        document = company.document[:1500]
        → sort 10 candidates → keep Top-5

    Step 2 — LLM reasoning (full context: DB + Tavily, ≤4 000 chars):
        5 sequential LLM calls (one per Top-5 company)
        → generates recommendation text + entry points
    """
    candidates    = state.get("candidate_companies", [])
    research      = state.get("research_results", {})
    user_desc     = state.get("user_company_desc", "")
    other_req     = state.get("other_requirements", "")
    soft_filters  = state.get("soft_filters") or {}
    fallback_lvl  = state.get("search_fallback_level", 0)
    errors        = list(state.get("errors") or [])

    if not candidates:
        logger.warning("ScoringAgent: no candidates received — returning empty scored_companies")
        return {"scored_companies": []}

    # ------------------------------------------------------------------
    # Step 1: Cross-encoder reranking
    #
    # Query anchor: hypothetical_partner_desc (HyDE).
    #
    # Why HyDE and not user_company_desc:
    #   HyDE generates a company-profile-style description ("An ideal partner is
    #   a London-based B2B energy consultancy..."), matching the vocabulary and
    #   format of the DB documents. ms-marco cross-encoder scores jump from
    #   ~0.001 (user raw text) to ~0.85-0.96 (HyDE text) because the model
    #   can properly assess relevance between two same-format documents.
    #   user_company_desc is casual/conversational — too far from DB document
    #   style for the cross-encoder to score meaningfully.
    #
    # Falls back to user_company_desc only when HyDE is absent (filter-only path).
    hyde_desc    = state.get("hypothetical_partner_desc", "")
    partner_type = state.get("partner_type_desc", "")

    # Cross-encoder query anchor:
    #   HyDE (passage-format description of the ideal partner) → ms-marco scores 0.80-0.96
    #   Falls back to user_desc only on filter-only path (no HyDE generated).
    if hyde_desc.strip():
        query  = hyde_desc.strip()
        anchor = "HyDE"
    else:
        query  = user_desc.strip()
        anchor = "user_desc"

    if query:
        logger.info(f"ScoringAgent: query anchor = {anchor} ({len(query)} chars)")
        logger.info(f"ScoringAgent: query preview → {query[:200]}")
    if not query:
        # No text anchor at all — skip reranking, use bi-encoder order
        logger.warning("ScoringAgent: no query text — skipping cross-encoder, using bi-encoder order")
        top5 = [
            {**c, "cross_encoder_score": c.get("similarity") or 0.0, "cross_encoder_raw": 0.0}
            for c in candidates[:_TOP_N]
        ]
    else:
        logger.info(f"ScoringAgent: cross-encoder scoring {len(candidates)} candidates...")
        try:
            ranked = _rerank(query, candidates)
            top5   = ranked[:_TOP_N]
            logger.info(
                f"ScoringAgent: top-{_TOP_N} scores = "
                f"{[c['cross_encoder_score'] for c in top5]}"
            )
        except Exception as e:
            logger.error(f"ScoringAgent: cross-encoder failed — {e}")
            errors.append(f"ScoringAgent: cross-encoder error — {e}")
            # Graceful fallback: keep bi-encoder order, zero cross-encoder score
            top5 = [
                {**c, "cross_encoder_score": c.get("similarity") or 0.0, "cross_encoder_raw": 0.0}
                for c in candidates[:_TOP_N]
            ]

    # ------------------------------------------------------------------
    # Step 1b: Assign match_quality labels (per company)
    # ------------------------------------------------------------------
    for company in top5:
        company["match_quality"]   = _assign_match_quality(
            company["cross_encoder_score"], fallback_lvl, anchor
        )
        company["soft_filter_hit"] = _check_soft_filters(company, soft_filters)

    # ------------------------------------------------------------------
    # Step 2: LLM reasoning — sequential, one call per Top-5 company
    #
    # Uses research_results[slug]["summary"] which contains:
    #   DB text (always) + Tavily web content (when available)
    # This gives LLM richer context for specific, well-grounded reasoning.
    # ------------------------------------------------------------------
    logger.info(f"ScoringAgent: generating reasoning for {len(top5)} companies (sequential)...")

    scored_companies = []
    for i, company in enumerate(top5):
        slug             = company.get("slug") or company.get("id", "unknown")
        research_entry   = research.get(slug, {})
        research_summary = research_entry.get("summary", "")
        source           = research_entry.get("source", "db")
        try:
            reasoning = _run_reasoning(
                user_company_desc=user_desc,
                other_requirements=other_req,
                soft_filters=soft_filters,
                company=company,
                research_summary=research_summary,
            )
            logger.info(
                f"ScoringAgent: #{i+1} {company.get('name', slug)!r} — "
                f"score={company['cross_encoder_score']:.3f} "
                f"quality={company['match_quality']} "
                f"source={source}"
            )
        except Exception as e:
            logger.error(f"ScoringAgent: reasoning failed for {slug!r}: {e}")
            errors.append(f"ScoringAgent: reasoning error for {slug} — {e}")
            reasoning = ""
        scored_companies.append({**company, "reasoning": reasoning})

    # Log quality distribution summary
    quality_dist = {q: sum(1 for c in scored_companies if c["match_quality"] == q)
                    for q in ("strong", "partial", "fallback")}
    logger.info(
        f"ScoringAgent: complete — {len(scored_companies)} scored. "
        f"Quality: {quality_dist}"
    )

    return {"scored_companies": scored_companies, "errors": errors}
