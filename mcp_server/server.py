"""
SDGZero Partner Finder — MCP Server

Exposes four tools to MCP clients (Claude Desktop, Cursor, etc.):

  1. search_companies     — Semantic search by natural language description,
                            optionally combined with structured filters.
                            Uses the same hybrid_search / semantic_search
                            logic as the SearchAgent internally.

  2. filter_companies     — Pure SQL metadata filter (city, SDG, category, etc.)
                            with no vector search. Fast and deterministic.

  3. get_company          — Fetch full profile of a single company by slug.

  4. find_partners        — Trigger the full Multi-Agent Pipeline
                            (SearchAgent → ResearchAgent → ScoringAgent → ReportAgent)
                            and return the structured report + top-5 matches.

All tools reuse existing system modules without modification:
  - agent/tools.py        for semantic_search, sql_filter, hybrid_search
  - agent/graph.py        for run_pipeline
  - db/pg_store.py        for direct DB access in get_company

Usage (Claude Desktop config):
    See mcp_server/README.md for setup instructions.

Run directly for testing:
    python -m mcp_server.server
"""

import os
import sys

# Ensure the project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "SDGZero Partner Finder",
    instructions=(
        "This server gives you access to the SDGZero sustainability business directory. "
        "Use list_filters to discover valid field values before filtering. "
        "Use search_companies for natural language queries, filter_companies for structured "
        "filtering, get_company for full company details, and find_partners to run the full "
        "AI partner-matching pipeline."
    ),
)


# ---------------------------------------------------------------------------
# Helpers — lazy singletons reused from agent/tools.py
# ---------------------------------------------------------------------------

def _encode(text: str) -> list[float]:
    """Encode a query string to a 384-dim normalised embedding."""
    from agent.tools import _get_encoder
    encoder = _get_encoder()
    return encoder.encode([text], normalize_embeddings=True)[0].tolist()


def _store():
    """Return the shared PGStore singleton."""
    from agent.tools import _get_store
    return _get_store()


def _format_company(company: dict, *, include_document: bool = False) -> dict:
    """
    Return a clean subset of company fields for MCP tool output.

    Strips internal fields (embedding, similarity) that are not useful
    to an MCP client, and truncates the full document to 300 chars.
    """
    sdg = company.get("sdg_tags") or company.get("predicted_sdg_tags") or ""
    description = company.get("document", "")
    if not include_document:
        description = description[:300] + ("..." if len(description) > 300 else "")

    return {
        "name":          company.get("name", ""),
        "slug":          company.get("slug", ""),
        "city":          company.get("city", ""),
        "country":       company.get("country", ""),
        "categories":    company.get("categories", ""),
        "sdg_tags":      sdg,
        "business_type": company.get("business_type", ""),
        "website":       company.get("website", ""),
        "linkedin":      company.get("linkedin", ""),
        "description":   description,
        "similarity":    round(company["similarity"], 3) if company.get("similarity") else None,
    }


# ---------------------------------------------------------------------------
# Tool 1: search_companies
# ---------------------------------------------------------------------------

@mcp.tool()
def search_companies(
    query: str,
    city: str = "",
    sdg: str = "",
    category: str = "",
    business_type: str = "",
    n_results: int = 10,
) -> list[dict]:
    """
    Search SDGZero companies by natural language description.

    Combines semantic similarity with optional structured filters.
    If no filters are provided, returns the most semantically similar companies.
    If filters are provided, applies SQL conditions first then ranks by similarity.

    IMPORTANT: sdg and category use fuzzy matching (substring), so partial values work
    (e.g. sdg="climate" matches "Climate Action"). city must be an exact match
    (case-sensitive). If unsure of valid values, call list_filters() first.

    Args:
        query:         Natural language description of the company or partner type you're looking for.
                       e.g. "renewable energy consultancy focused on heat pump installations"
        city:          Filter by city — must exactly match a DB value (e.g. "London", "Preston").
                       Call list_filters() to see all valid cities.
        sdg:           Filter by SDG goal — partial/fuzzy match works, e.g. "climate", "clean energy".
        category:      Filter by business category — partial match works, e.g. "Energy", "Health".
        business_type: Filter by business model — "B2B", "B2C", or "Both".
        n_results:     Number of results to return (default 10, max 50).

    Returns:
        List of matching companies sorted by relevance, each with name, location,
        SDG tags, categories, website, LinkedIn, and a short description.
    """
    from agent.tools import semantic_search_from_embedding, hybrid_search
    from agent.search_agent import _run_hyde, _averaged_embedding

    n_results = min(n_results, 50)

    # Build filters dict — only include non-empty values
    filters = {}
    if city:
        filters["city"] = city
    if sdg:
        filters["sdg_tags"] = [sdg]
    if category:
        filters["categories"] = category
    if business_type:
        filters["business_type"] = business_type

    # HyDE: treat the query as the target partner type description.
    # Passing query as partner_type_desc (not user_company_desc) tells HyDE
    # to generate a profile OF that type of company, not a partner FOR it.
    partner_desc, expansions, inferred_type = _run_hyde(
        user_company_desc="",
        partner_type_desc=query,
        filters=filters,
    )
    all_texts = [inferred_type, inferred_type, partner_desc] if inferred_type else [partner_desc] + expansions
    embedding = _averaged_embedding(all_texts)

    if filters:
        results = hybrid_search(embedding, filters, n_results=n_results)
    else:
        results = semantic_search_from_embedding(embedding, n_results=n_results)

    return [_format_company(c) for c in results]


# ---------------------------------------------------------------------------
# Tool 2: filter_companies
# ---------------------------------------------------------------------------

@mcp.tool()
def filter_companies(
    city: str = "",
    sdg: str = "",
    category: str = "",
    business_type: str = "",
    job_sector: str = "",
    claimed: bool = False,
    n_results: int = 20,
) -> list[dict]:
    """
    Filter SDGZero companies by structured metadata conditions.

    No vector search — results are sorted by membership tier (premium first).
    Use this when you want deterministic filtering without semantic ranking.

    IMPORTANT: If you are unsure of valid values for any field, call list_filters()
    first to retrieve all available options from the database.

    Args:
        city:          Exact city name — must match DB exactly (case-sensitive).
                       Call list_filters() to see all valid cities.
        sdg:           SDG goal — partial/fuzzy match works, e.g. "climate", "health".
        category:      Business category — partial match works, e.g. "Energy", "Marketing".
        business_type: "B2B", "B2C", or "Both".
        job_sector:    "Private", "Public", or "Agencies".
                       Call list_filters() to confirm available values.
        claimed:       If True, return only verified/claimed profiles.
        n_results:     Number of results to return (default 20, max 100).

    Returns:
        List of companies matching all specified conditions, sorted by membership tier.
    """
    from agent.tools import sql_filter

    n_results = min(n_results, 100)

    filters = {}
    if city:
        filters["city"] = city
    if sdg:
        filters["sdg_tags"] = [sdg]
    if category:
        filters["categories"] = category
    if business_type:
        filters["business_type"] = business_type
    if job_sector:
        filters["job_sector"] = job_sector
    if claimed:
        filters["claimed"] = True

    if not filters:
        return [{"error": "Please provide at least one filter condition."}]

    results = sql_filter(filters, n_results=n_results)
    return [_format_company(c) for c in results]


# ---------------------------------------------------------------------------
# Tool 3: list_filters
# ---------------------------------------------------------------------------

@mcp.tool()
def list_filters() -> dict:
    """
    Return all valid filter values available in the SDGZero database.

    Call this tool BEFORE using filter_companies, search_companies, or find_partners
    when you are unsure what values to pass for city, sdg, category, business_type,
    or job_sector. This prevents empty results caused by invalid field values.

    Returns:
        A dict with keys:
          - cities:         All distinct city names (exact match required)
          - sdg_goals:      All 17 SDG goal names (fuzzy match supported)
          - categories:     All business categories (fuzzy match supported)
          - business_types: Valid values — "B2B", "B2C", "Both"
          - job_sectors:    Valid values — "Private", "Public", "Agencies"
          - company_sizes:  Valid company size bands
    """
    options = _store().build_schema_data()
    return {
        "cities":         options.get("city", []),
        "sdg_goals":      options.get("sdg_tags", []),
        "categories":     options.get("categories", []),
        "business_types": options.get("business_type", []),
        "job_sectors":    options.get("job_sector", []),
        "company_sizes":  options.get("company_size", []),
    }


# ---------------------------------------------------------------------------
# Tool 4: get_company
# ---------------------------------------------------------------------------

@mcp.tool()
def get_company(slug: str) -> dict:
    """
    Get the full profile of a single SDGZero company by its slug.

    Args:
        slug: The company's URL slug, e.g. "greentech-london", "heat-engineer-software-ltd".
              You can find the slug from search_companies or filter_companies results.

    Returns:
        Full company profile including description, SDG tags, social links,
        location, business details, and AI-predicted SDG tags.
        Returns an error dict if the company is not found.
    """
    store = _store()
    with store._cursor() as cur:
        cur.execute(
            "SELECT * FROM businesses WHERE slug = %s LIMIT 1",
            (slug,),
        )
        row = cur.fetchone()

    if not row:
        return {"error": f"Company with slug '{slug}' not found."}

    company = dict(row)
    # Return full description for detail view
    return _format_company(company, include_document=True)


# ---------------------------------------------------------------------------
# Tool 5: find_partners
# ---------------------------------------------------------------------------

@mcp.tool()
def find_partners(
    my_company_desc: str,
    partner_type: str = "",
    city: str = "",
    sdg: str = "",
    category: str = "",
    other_requirements: str = "",
) -> dict:
    """
    Run the full AI partner-matching pipeline to find the top 5 partner companies.

    This triggers the complete Multi-Agent Pipeline:
      1. SearchAgent  — HyDE query expansion + hybrid search + CRAG reflection
      2. ResearchAgent — Tavily web enrichment for each candidate
      3. ScoringAgent  — Cross-encoder reranking + LLM reasoning
      4. ReportAgent   — Structured Markdown report with outreach drafts

    This tool takes 20-60 seconds. Use search_companies for faster lookups.

    Args:
        my_company_desc:    Description of YOUR company (the one looking for partners).
                            e.g. "We provide carbon audit and net-zero consulting to UK SMEs."
        partner_type:       Optional: explicit description of the kind of partner you want.
                            e.g. "media agency", "logistics provider", "impact investor"
        city:               Filter candidates by city, e.g. "London".
        sdg:                Filter candidates by SDG goal, e.g. "Climate Action".
        category:           Filter candidates by category, e.g. "Energy & Renewables".
        other_requirements: Free-text additional preferences, e.g. "prefer B2B, under 50 employees".

    Returns:
        A dict with:
          - report:            Full Markdown partner-matching report with top-5 recommendations
          - top_companies:     Structured list of top-5 companies with scores and reasoning
          - search_method:     Which search method was used (semantic / hybrid / sql)
          - fallback_level:    Whether search conditions were relaxed (0 = full, 1-3 = relaxed)
          - session_id:        Session ID for reference
    """
    from agent.graph import run_pipeline

    filters = {}
    if city:
        filters["city"] = city
    if sdg:
        filters["sdg_tags"] = [sdg]
    if category:
        filters["categories"] = category

    state = run_pipeline(
        user_company_desc=my_company_desc,
        partner_type_desc=partner_type,
        filters=filters,
        other_requirements=other_requirements,
    )

    # Format top companies cleanly
    top_companies = []
    for c in state.get("scored_companies", []):
        top_companies.append({
            "name":               c.get("name", ""),
            "slug":               c.get("slug", ""),
            "city":               c.get("city", ""),
            "categories":         c.get("categories", ""),
            "sdg_tags":           c.get("sdg_tags") or c.get("predicted_sdg_tags", ""),
            "website":            c.get("website", ""),
            "linkedin":           c.get("linkedin", ""),
            "match_score":        round(c.get("cross_encoder_score", 0.0), 3),
            "reasoning":          c.get("reasoning", ""),
        })

    return {
        "report":         state.get("report", ""),
        "top_companies":  top_companies,
        "search_method":  state.get("search_method", ""),
        "fallback_level": state.get("search_fallback_level", 0),
        "session_id":     state.get("session_id", ""),
        "errors":         state.get("errors", []),
        "notices":        state.get("notices", []),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
