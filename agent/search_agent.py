"""
SearchAgent — Step 1 implementation.

Responsibilities:
  1. HyDE: LLM generates a hypothetical ideal partner description (50-120 words)
           + Query Expansion (3-5 equivalent search phrases)
  2. Embed: encode all generated texts → average into a single query vector
  3. Search: semantic_search_from_embedding → Top-10 candidate companies
  4. LLM-as-Judge + Reflection: evaluate candidate quality, reflect on failure,
     selectively retry (CRAG) — keeps good results, replaces bad ones
  5. Write: hypothetical_partner_desc, query_expansions, candidate_companies → AgentState

References:
  HyDE:       Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels"
  CRAG:       Yan et al. 2024, "Corrective Retrieval Augmented Generation"
  Reflexion:  Shinn et al. 2023, "Reflexion: Language Agents with Verbal Reinforcement"
  LLM-Judge:  Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench"
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

Your job is to write a profile of the PARTNER company and generate search phrases to find them.

CRITICAL RULE for partner_description:
  - Based on the user's company's desccription, describe the PARTNER company's own identity: what they DO, their services, sector, SDG focus, business model.
  - Do NOT repeat or paraphrase the user's company in the description.
  - The user's company is context — use it to decide WHICH type of partner to describe and WHAT they need,
    but the final description must read like the partner's own company profile, not a description of who they serve.

Two modes depending on whether a target partner type is specified:

MODE A — Target type specified (e.g. "media company", "logistics provider"):
  - The target type may be explicit ("media company") OR vague ("companies who may need our products").
  - If the target type is VAGUE or intent-based (e.g. "companies who need X", "potential customers"):
      → First INFER the actual industry/company type from the user's business context.
      → e.g. user = sports nutrition company, target = "companies who may need our products"
        → infer: gyms, fitness centres, sports clubs, sports retailers, athletic training facilities
        → describe ONE of those inferred types
      → IMPORTANT: infer narrowly and specifically. Do NOT infer broadly (e.g. "any company with employees").
        If the user's product/service is niche, the target should also be niche.
      → Ambiguous words in user_company_desc: interpret them in context of the primary business.
        e.g. "sports nutrition and caring company" → "caring" = health/wellness care, NOT car care or industrial care.
        e.g. "food and beverage company" → partners = restaurants, retailers, caterers — NOT generic "any business".
  - If the target type is EXPLICIT ("gym", "media company", "logistics provider"):
      → describe a company of that exact type directly
  - In both cases: describe what the PARTNER does — their services, sector, SDG alignment, business model.
  - Keep user's industry context only in query_expansions, NOT in partner_description.
  - Example: user = skin care brand, target = "media company"
    WRONG: "A media agency specialising in beauty and wellness brands..."
    RIGHT:  "A digital marketing and PR agency offering brand storytelling, social media management,
             influencer campaigns, and content creation. SDG focus on responsible consumption..."

MODE B — No target type specified:
  - Identify what kind of partner would best complement the user's company.
  - Describe that partner company's own profile: what they do, their sector, SDG focus, business model.

Write the partner_description in the same style as a real company directory listing: concrete and specific.
Do NOT be generic. Imagine one real company.

Respond ONLY with valid JSON. No explanation, no markdown.
"""

_HYDE_HUMAN = """My company:
{user_company_desc}
{extra}
{filter_context}
{partner_type_instruction}Generate a JSON response with exactly these three keys:
{{
  "inferred_partner_type": "<3-6 word label for the specific company type you decided to describe, e.g. 'fitness centre', 'sports equipment retailer', 'marketing agency'>",
  "partner_description": "<50-120 word description of an ideal partner, written like a real company profile>",
  "query_expansions": ["<phrase 1>", "<phrase 2>", "<phrase 3>", "<phrase 4>", "<phrase 5>"]
}}

The partner_description must describe ONE specific ideal partner company.
{expansion_instruction}
"""


# ---------------------------------------------------------------------------
# HyDE generation
# ---------------------------------------------------------------------------

def _run_hyde(
    user_company_desc: str,
    other_requirements: str = "",
    partner_type_desc: str = "",
    filters: Optional[dict] = None,
) -> tuple[str, list[str], str]:
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
        (partner_description, query_expansions, inferred_partner_type)
        inferred_partner_type: concise label for the specific company type LLM decided
        to describe (e.g. "fitness centre", "marketing agency"). Falls back to
        partner_type_desc if not present.
    """
    llm = get_llm()

    if partner_type_desc.strip():
        partner_type_instruction = (
            f"IMPORTANT — MODE A: The target partner is described as: '{partner_type_desc.strip()}'.\n"
            f"CRITICAL: First check whether the description STARTS WITH an explicit company type name "
            f"(e.g. 'Accountants', 'Law firms', 'Marketing agencies', 'Logistics providers', 'Banks', 'Consultants'). "
            f"If it does — even if followed by intent context like 'help us with X' or 'who can do Y' — "
            f"treat the named type as EXPLICIT and describe a company of THAT exact type. "
            f"Do NOT blend my industry with the intent to infer a hybrid third type. "
            f"Example: 'Accountants or law companies help us deal with risks' → describe an accountancy or law firm, "
            f"NOT a 'construction risk management firm'.\n"
            f"If the description does NOT start with an explicit company type name and is vague or intent-based "
            f"(e.g. 'companies who need our products', 'potential customers'), "
            f"first infer the most specific industry/company type this maps to given MY company, "
            f"then describe a company of that inferred type.\n"
            f"Write the partner_description as the PARTNER'S OWN company profile — "
            f"focus on what THEY do, their services, sector, and SDG focus.\n"
            f"Do NOT mention my industry or describe them as 'serving my type of company' — "
            f"that context belongs only in query_expansions, not in partner_description.\n\n"
        )
        expansion_instruction = (
            f"The query_expansions should find {partner_type_desc.strip()} companies "
            f"that would suit businesses like mine — use industry-specific angles here."
        )
    else:
        partner_type_instruction = (
            f"MODE B: No specific partner type was given. "
            f"Based on my company description, identify the most valuable type of partner I need "
            f"(e.g. a distributor, marketing agency, technology provider, investor, etc.) "
            f"and describe a company that fills that gap while sharing compatible sustainability values.\n"
            f"IMPORTANT: prefer a partner from a DIFFERENT industry/sector than my own company, "
            f"unless the additional requirements below explicitly ask for peers or collaborators in the same space.\n\n"
        )
        expansion_instruction = (
            "The query_expansions should cover different angles: "
            "by what the partner does, by the sector they serve, and by shared SDG/sustainability focus."
        )

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

    # Include other_requirements in HyDE context so exclusions and preferences
    # (e.g. "exclude skin-care", "prefer health sector") steer the generated
    # partner description and query expansions.
    extra = (
        f"\nAdditional requirements for the ideal partner: {other_requirements.strip()}\n"
        if other_requirements and other_requirements.strip()
        else ""
    )
    prompt = _HYDE_HUMAN.format(
        user_company_desc=user_company_desc,
        partner_type_instruction=partner_type_instruction,
        expansion_instruction=expansion_instruction,
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
        inferred_type = data.get("inferred_partner_type", "").strip() or partner_type_desc
        return partner_desc, expansions, inferred_type

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"HyDE JSON parse failed ({e}). Falling back to raw text.")
        fallback_desc = raw[:500] if raw else user_company_desc
        return fallback_desc, [], partner_type_desc


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

    Level 1: drop categories (softest constraint — broad industry label).
    Level 2: drop sdg_tags + claimed (mission tags and verification flag).
    Hard filters (city, business_type, job_sector, company_size) are never
    dropped here — Level 3 (pure semantic) is handled by the caller.
    """
    relaxed = dict(filters)
    if level >= 1:
        relaxed.pop("categories", None)
    if level >= 2:
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
# LLM-as-Judge + Reflection
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a B2B partner matching quality evaluator for SDGZero, a sustainability directory.

Your job is to evaluate a list of candidate companies and determine whether each one is a
good match for the user's requirements. You also reflect on WHY the search went wrong
(if it did), so the next search attempt can avoid the same mistake.

Respond ONLY with valid JSON. No explanation, no markdown.
"""

_JUDGE_HUMAN = """\
User's company:
{user_company_desc}

Target partner type: {partner_type_desc}
(If empty, evaluate purely on business complementarity with the user's company.)

Candidate companies (evaluate each):
{candidates_json}

For each candidate, score on TWO dimensions:
  type_score (0-2): Does it match the target partner type?
    0 = clearly wrong type
    1 = possibly relevant but uncertain
    2 = clearly the right type
  fit_score (0-2): Does it complement the user's company business?
    0 = no relevance
    1 = some relevance
    2 = highly complementary

Also write a one-sentence reflection explaining WHY the overall search results are
poor (if they are), focusing on what caused the mismatch — e.g. embedding drift,
wrong vocabulary, HyDE mixing two industries, etc.
If results are good overall, set reflection to an empty string.

Respond with exactly this JSON structure:
{{
  "judgments": [
    {{"id": "<company id>", "type_score": 0-2, "fit_score": 0-2}}
  ],
  "reflection": "<one sentence explaining the failure cause, or empty string>"
}}
"""

_MIN_GOOD = 5        # minimum good candidates before triggering retry
_GOOD_THRESHOLD = 3  # minimum total score (type*2 + fit) to be "good"


def _judge_and_reflect(
    candidates: list[dict],
    user_company_desc: str,
    partner_type_desc: str,
) -> tuple[list[str], list[str], str]:
    """
    LLM-as-Judge: evaluate each candidate and reflect on search quality.

    One LLM call evaluates all candidates simultaneously and produces:
      - good_ids:   company IDs that pass the quality threshold
      - bad_ids:    company IDs that fail
      - reflection: one-sentence explanation of why results are poor (or "")

    Scoring logic:
      With partner_type: good = type_score >= 1 AND (type_score*2 + fit_score) >= THRESHOLD
      Without partner_type: good = fit_score >= 1

    Args:
        candidates:        list of candidate company dicts from search
        user_company_desc: user's own company description
        partner_type_desc: target partner type (may be empty)

    Returns:
        (good_ids, bad_ids, reflection)
    """
    llm = get_llm()

    # Compact representation to stay within token limits
    candidates_summary = [
        {
            "id": c.get("id", c.get("slug", "")),
            "name": c.get("name", ""),
            "categories": c.get("categories", ""),
            "description": (c.get("document") or "")[:200],
        }
        for c in candidates
    ]

    from langchain_core.messages import SystemMessage, HumanMessage
    prompt = _JUDGE_HUMAN.format(
        user_company_desc=user_company_desc,
        partner_type_desc=partner_type_desc or "(not specified — focus on business fit)",
        candidates_json=json.dumps(candidates_summary, ensure_ascii=False, indent=2),
    )
    messages = [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"SearchAgent judge: LLM call failed ({e}) — skipping reflection")
        all_ids = [c.get("id", c.get("slug", "")) for c in candidates]
        return all_ids, [], ""

    judgments = {j["id"]: j for j in data.get("judgments", [])}
    reflection = data.get("reflection", "").strip()

    good_ids, bad_ids = [], []
    for c in candidates:
        cid = c.get("id", c.get("slug", ""))
        j = judgments.get(cid)
        if j is None:
            good_ids.append(cid)  # unknown → assume ok
            continue

        type_score = int(j.get("type_score", 0))
        fit_score = int(j.get("fit_score", 0))

        if partner_type_desc.strip():
            # Strict: must clearly match the target type (score=2), not just "possibly"
            # type_score=1 ("possibly relevant") allows wrong-industry companies through
            is_good = type_score >= 2
        else:
            is_good = fit_score >= 1

        (good_ids if is_good else bad_ids).append(cid)

    logger.info(
        f"SearchAgent judge: {len(good_ids)} good, {len(bad_ids)} bad. "
        f"reflection={'yes' if reflection else 'none'}"
    )
    if reflection:
        logger.info(f"SearchAgent reflection: {reflection}")

    return good_ids, bad_ids, reflection


def _selective_retry(
    candidates: list[dict],
    good_ids: set,
    reflection: str,
    partner_type_desc: str,
    user_company_desc: str,
    filters: dict,
    avg_embedding: list[float],
) -> tuple[list[dict], str]:
    """
    CRAG selective retry: keep good candidates, search for replacements for bad ones.

    Strategy:
      1. Keep all good candidates as-is
      2. Re-run HyDE with reflection injected → new embedding
      3. Search for (10 - len(good)) additional candidates
      4. Deduplicate against all previously seen IDs
      5. Return (good + new, retry_desc) so caller can update hypothetical_partner_desc

    Args:
        candidates:        original candidate list
        good_ids:          IDs of candidates that passed judge
        reflection:        one-sentence failure reason from judge
        partner_type_desc: target partner type
        user_company_desc: user's company description
        filters:           current search filters
        avg_embedding:     original search embedding (fallback)

    Returns:
        (new_candidates, retry_desc) — candidates with good originals + fresh replacements,
        and the updated HyDE description to use as hypothetical_partner_desc for scoring.
    """
    good_candidates = [c for c in candidates if c.get("id", c.get("slug")) in good_ids]
    seen_ids = {c.get("id", c.get("slug", "")) for c in candidates}
    need = max(10 - len(good_candidates), 3)

    logger.info(
        f"SearchAgent retry: keeping {len(good_candidates)} good, "
        f"searching {need} replacements. reflection='{reflection[:80]}...'"
    )

    # Re-run HyDE with reflection injected as memory
    reflection_hint = (
        f"\n\nIMPORTANT — previous search failed. Reason: {reflection}\n"
        f"Adjust your description to avoid repeating this mistake.\n"
        if reflection else ""
    )

    retry_desc = ""
    try:
        retry_desc, _, _retry_inferred = _run_hyde(
            user_company_desc=user_company_desc + reflection_hint,
            partner_type_desc=partner_type_desc,
            filters=filters,
        )
        logger.info(f"SearchAgent retry: HyDE → {retry_desc[:150]}")
        # Anchor embedding toward partner type (same strategy as primary search)
        if partner_type_desc.strip():
            retry_texts = [partner_type_desc, partner_type_desc, retry_desc]
        else:
            retry_texts = [retry_desc]
        retry_embedding = _averaged_embedding(retry_texts)
    except Exception as e:
        logger.warning(f"SearchAgent retry: HyDE failed ({e}), reusing original embedding")
        retry_embedding = avg_embedding

    # Search for replacements using the same 3-level fallback as the initial search.
    # Level 0: full filters + vector
    # Level 1: drop categories (skip if not present)
    # Level 2: drop sdg_tags + claimed
    # Level 3: pure semantic
    fetch = need + len(seen_ids)
    raw_new = hybrid_search(retry_embedding, filters, n_results=fetch) if filters else []
    logger.info(f"SearchAgent retry level 0: {len(raw_new)} raw")

    if len([c for c in raw_new if c.get("id", c.get("slug", "")) not in seen_ids]) < need:
        if "categories" in filters:
            relaxed1 = _relax_filters(filters, level=1)
            raw_new = hybrid_search(retry_embedding, relaxed1, n_results=fetch)
            logger.info(f"SearchAgent retry level 1 (drop categories): {len(raw_new)} raw")

    if len([c for c in raw_new if c.get("id", c.get("slug", "")) not in seen_ids]) < need:
        relaxed2 = _relax_filters(filters, level=2)
        if relaxed2:
            raw_new = hybrid_search(retry_embedding, relaxed2, n_results=fetch)
        else:
            raw_new = semantic_search_from_embedding(retry_embedding, n_results=fetch)
        logger.info(f"SearchAgent retry level 2 (drop sdg+claimed): {len(raw_new)} raw")

    if len([c for c in raw_new if c.get("id", c.get("slug", "")) not in seen_ids]) < need:
        raw_new = semantic_search_from_embedding(retry_embedding, n_results=fetch)
        logger.info(f"SearchAgent retry level 3 (pure semantic): {len(raw_new)} raw")

    new_candidates = [c for c in raw_new if c.get("id", c.get("slug", "")) not in seen_ids][:need]

    logger.info(f"SearchAgent retry: found {len(new_candidates)} fresh replacements")
    return good_candidates + new_candidates, retry_desc

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
        partner_desc, expansions, inferred_type = _run_hyde(user_desc, other_req, partner_type, filters)
        logger.info(f"SearchAgent: HyDE done. expansions={len(expansions)}")
        logger.info(f"SearchAgent: HyDE content → {partner_desc[:200]}")
        if inferred_type and inferred_type != partner_type:
            logger.info(f"SearchAgent: inferred partner type → '{inferred_type}'")

        # ------------------------------------------------------------------
        # Step 1b: Embed — build search vector
        #
        # Use inferred_type (the specific label HyDE resolved to) as the
        # anchor instead of the raw partner_type_desc, which may be vague
        # (e.g. "companies who may need our products"). inferred_type is
        # always a concrete label ("fitness centre", "marketing agency").
        # Repeat x2 to give it stronger weight vs HyDE.
        # ------------------------------------------------------------------
        if inferred_type:
            all_texts = [inferred_type, inferred_type, partner_desc]
            logger.info(
                f"SearchAgent: anchoring embedding with inferred_type='{inferred_type}' (x2) + HyDE"
            )
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
    MIN_RESULTS = 5   # pad with semantic results if filtered set is smaller

    candidates = []
    method = "semantic"
    fallback_level = 0
    _dead_zone_notice = ""

    def _pad_with_semantic(existing: list, target: int) -> list:
        """
        Supplement existing results with semantic search up to `target` total.
        Existing (filtered) results always come first and are never discarded.
        """
        if len(existing) >= target:
            return existing
        seen_ids = {c.get("id") for c in existing}
        semantic = semantic_search_from_embedding(avg_embedding, n_results=target)
        extras = [c for c in semantic if c.get("id") not in seen_ids]
        return existing + extras[: target - len(existing)]

    if has_desc and has_filters:
        # hybrid: SQL WHERE + vector ORDER BY (pgvector single query)
        method = "hybrid"
        candidates = hybrid_search(avg_embedding, filters, n_results=10)
        logger.info(f"SearchAgent: hybrid_search → {len(candidates)} results (level 0)")

        if len(candidates) == 0 and "categories" in filters:
            # Level 1: drop categories
            relaxed1 = _relax_filters(filters, level=1)
            candidates = hybrid_search(avg_embedding, relaxed1, n_results=10)
            fallback_level = 1
            logger.info(f"SearchAgent: hybrid level 1 (drop categories) → {len(candidates)} results")

        if len(candidates) == 0:
            # Level 2: drop sdg_tags + claimed
            relaxed2 = _relax_filters(filters, level=2)
            candidates = hybrid_search(avg_embedding, relaxed2, n_results=10)
            fallback_level = 2
            logger.info(f"SearchAgent: hybrid level 2 (drop sdg+claimed) → {len(candidates)} results")

        if len(candidates) == 0:
            # Level 3: pure semantic — drop all filters
            candidates = semantic_search_from_embedding(avg_embedding, n_results=10)
            fallback_level = 3
            method = "semantic"
            logger.info(f"SearchAgent: level 3 pure semantic → {len(candidates)} results")
        elif len(candidates) < MIN_RESULTS:
            candidates = _pad_with_semantic(candidates, MIN_RESULTS)
            logger.info(f"SearchAgent: padded to {len(candidates)} results")

    elif has_filters and not has_desc:
        # pure SQL filter (no description → no vector)
        method = "sql"
        candidates = sql_filter(filters, n_results=10)
        logger.info(f"SearchAgent: sql_filter → {len(candidates)} results (level 0)")

        if len(candidates) == 0 and "categories" in filters:
            # Level 1: drop categories
            relaxed1 = _relax_filters(filters, level=1)
            candidates = sql_filter(relaxed1, n_results=10)
            fallback_level = 1
            logger.info(f"SearchAgent: sql level 1 (drop categories) → {len(candidates)} results")

        if len(candidates) == 0:
            # Level 2: drop sdg_tags + claimed
            relaxed2 = _relax_filters(filters, level=2)
            candidates = sql_filter(relaxed2, n_results=10)
            fallback_level = 2
            logger.info(f"SearchAgent: sql level 2 (drop sdg+claimed) → {len(candidates)} results")

        if len(candidates) == 0:
            # Level 3: pure semantic
            candidates = semantic_search_from_embedding(avg_embedding, n_results=10)
            fallback_level = 3
            method = "semantic"
            logger.info(f"SearchAgent: level 3 pure semantic → {len(candidates)} results")
        elif len(candidates) < MIN_RESULTS:
            candidates = _pad_with_semantic(candidates, MIN_RESULTS)
            logger.info(f"SearchAgent: padded to {len(candidates)} results")

    else:
        # pure semantic: no filters
        candidates = semantic_search_from_embedding(avg_embedding, n_results=10)
        logger.info(f"SearchAgent: semantic_search → {len(candidates)} results")

    logger.info(f"SearchAgent: initial → {len(candidates)} candidates via {method} (fallback={fallback_level})")

    # ------------------------------------------------------------------
    # Step 2: LLM-as-Judge + Reflection + CRAG selective retry
    #
    # Only runs when we have candidates AND a user description to judge against.
    # Skipped when there are too few candidates to evaluate meaningfully.
    # ------------------------------------------------------------------
    if candidates and has_desc:
        # Use inferred_type for judge — it's a concrete label (e.g. "fitness centre")
        # even when the original partner_type_desc was vague ("companies who may need our products")
        judge_type = inferred_type if has_desc else partner_type
        good_ids, bad_ids, reflection = _judge_and_reflect(
            candidates=candidates,
            user_company_desc=user_desc,
            partner_type_desc=judge_type,
        )

        if len(good_ids) < _MIN_GOOD and bad_ids:
            # Not enough good candidates — selectively retry for replacements
            candidates, retry_partner_desc = _selective_retry(
                candidates=candidates,
                good_ids=set(good_ids),
                reflection=reflection,
                partner_type_desc=judge_type,
                user_company_desc=user_desc,
                filters=filters,
                avg_embedding=avg_embedding,
            )
            # Update hypothetical_partner_desc so ScoringAgent's cross-encoder
            # uses the corrected (retry) HyDE as its query anchor, not the
            # original beauty-biased description that caused the wrong ranking.
            if retry_partner_desc:
                partner_desc = retry_partner_desc
                logger.info("SearchAgent: hypothetical_partner_desc updated to retry HyDE")
            logger.info(f"SearchAgent: after retry → {len(candidates)} candidates")

            # ------------------------------------------------------------------
            # SQL quality dead zone escape
            #
            # Problem: hard filters (city/business_type/...) may return enough
            # results to skip the 3-level fallback, but all candidates are
            # semantically poor. After one CRAG retry (still within the same
            # filtered pool), quality may still be bad — we never see globally
            # better matches outside the filter.
            #
            # Fix: run Judge once more after retry. If still poor AND hard
            # filters were active, do a final unconstrained semantic search.
            # Always keep good candidates; replace bad ones with global results.
            # ------------------------------------------------------------------
            hard_filter_keys = {"city", "business_type", "job_sector", "company_size"}
            had_hard_filters = any(k in filters for k in hard_filter_keys)
            if had_hard_filters and state.get("allow_global_fallback", False):
                good_ids_2, bad_ids_2, _ = _judge_and_reflect(candidates, user_desc, judge_type)
                if len(good_ids_2) < _MIN_GOOD and bad_ids_2:
                    logger.info(
                        f"SearchAgent: quality still poor after retry "
                        f"({len(good_ids_2)}/{len(candidates)} good) — escaping hard filters"
                    )
                    good_candidates = [
                        c for c in candidates
                        if c.get("id", c.get("slug", "")) in set(good_ids_2)
                    ]
                    seen_ids = {c.get("id", c.get("slug", "")) for c in candidates}
                    need = max(10 - len(good_candidates), 0)
                    global_results = semantic_search_from_embedding(
                        avg_embedding, n_results=need + len(seen_ids)
                    )
                    fresh = [
                        c for c in global_results
                        if c.get("id", c.get("slug", "")) not in seen_ids
                    ][:need]
                    candidates = good_candidates + fresh
                    logger.info(
                        f"SearchAgent: dead zone escape — kept {len(good_candidates)} good, "
                        f"added {len(fresh)} unconstrained results"
                    )
                    _dead_zone_notice = (
                        "No strong matches found within your selected filters. "
                        "Search has been expanded globally to find the best available partners."
                    )
        else:
            logger.info(f"SearchAgent: judge passed ({len(good_ids)}/{len(candidates)} good) — no retry needed")

    logger.info(f"SearchAgent: final → {len(candidates)} candidates via {method} (fallback={fallback_level})")

    result: dict = {
        "hypothetical_partner_desc": partner_desc,
        "query_expansions": expansions,
        "candidate_companies": candidates,
        "search_method": method,
        "search_fallback_level": fallback_level,
    }
    if _dead_zone_notice:
        result["notices"] = (state.get("notices") or []) + [_dead_zone_notice]
    return result
