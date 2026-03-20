"""
AgentState — shared state object passed through the entire LangGraph pipeline.

Every agent reads from this state and writes to its own output fields.
All fields are declared upfront so adding a new agent = filling in its slot.

Field ownership:
  User input     → set before pipeline starts, all agents can read
  SearchAgent    → hypothetical_partner_desc, query_expansions, candidate_companies, ...
  ResearchAgent  → research_results
  ScoringAgent   → scored_companies
  ReportAgent    → report
"""

from typing import TypedDict, Optional


class SearchFilters(TypedDict, total=False):
    """
    Structured filters from the user's search form.

    All fields are optional — user may leave any blank.
    Values must match what actually exists in the database (no free-text like "England").
    The frontend populates dropdowns from real DB values via a schema cache.

    Examples:
        {"city": "London", "business_type": "B2B", "claimed": True}
        {"sdg_tags": ["Climate Action", "Affordable And Clean Energy"]}
    """
    city: str                    # e.g. "London" — must match DB value exactly
    categories: list[str]        # e.g. ["Energy & Renewables", "Technology & Digital"]
    sdg_tags: list[str]          # e.g. ["Climate Action", "SDG 7"]
    business_type: str           # "B2B" | "B2C" | "Both"
    job_sector: str              # "Private" | "Public" | "Third Sector"
    claimed: bool                # True = verified profiles only
    company_size: str            # e.g. "SME"


class AgentState(TypedDict, total=False):
    """
    Shared state object for the entire Multi-Agent Pipeline.

    Passed sequentially through: SearchAgent → ResearchAgent → ScoringAgent → ReportAgent
    Each agent reads upstream fields and writes to its own fields.

    Usage in LangGraph:
        Each node function receives this state and returns a dict of updated fields.
        LangGraph merges the returned dict into the state automatically.

    Adding a new agent:
        1. Add its output fields here (with comments)
        2. Implement the node function
        3. Register it in graph.py
    """

    # ------------------------------------------------------------------
    # User input — set before pipeline starts, never modified by agents
    # ------------------------------------------------------------------

    session_id: str
    """Unique session ID for URL sharing and persistence. e.g. 'x7k2m9'"""

    user_company_desc: str
    """User's free-text description of their own company. Core anchor for all agents."""

    filters: SearchFilters
    """Structured filter conditions from the search form (Hard filters — go into WHERE clause)."""

    soft_filters: SearchFilters
    """
    Soft filter conditions from the search form (user marked as 'Preferred', not 'Must').
    NOT passed to the database WHERE clause — bypasses SQL entirely.
    Passed to ScoringAgent as bonus scoring criteria:
      - Companies satisfying soft conditions rank higher
      - Companies NOT satisfying them are still included (not excluded)
    """

    other_requirements: str
    """Free-text additional requirements. Appended to semantic search context."""

    partner_type_desc: str
    """
    User's explicit description of the kind of company they want to find.
    e.g. 'a media or marketing agency that can help promote our brand'
    When provided, used directly as the Cross-encoder query anchor (replaces HyDE inference).
    Also injected into HyDE prompt so the generated profile matches the target type.
    """

    # ------------------------------------------------------------------
    # SearchAgent output
    # Writes: hypothetical_partner_desc, query_expansions, candidate_companies,
    #         search_method, search_fallback_level
    # ------------------------------------------------------------------

    hypothetical_partner_desc: str
    """
    HyDE: LLM-generated description of an ideal partner company.
    50-120 words, written in the style of a real company profile.
    Passed to ScoringAgent as the Cross-encoder anchor.
    """

    query_expansions: list[str]
    """
    Query Expansion: 3-5 equivalent search phrases from different angles.
    Combined with hypothetical_partner_desc for richer vector search.
    """

    candidate_companies: list[dict]
    """
    Top-10 candidate companies returned by the search tool.
    Each dict has: id, slug, name, city, categories, sdg_tags,
                   website, linkedin, similarity, document (full embedding text).
    """

    search_method: str
    """Which tool was used: 'semantic' | 'sql' | 'hybrid'"""

    search_fallback_level: int
    """
    Degradation level used:
      0 = full conditions (SQL + vector)
      1 = relaxed SQL, kept vector
      2 = vector only, all SQL filters dropped
    """

    # ------------------------------------------------------------------
    # ResearchAgent output  (Step 3)
    # Writes: research_results
    # ------------------------------------------------------------------

    research_results: dict
    """
    External research per company, keyed by company slug.
    Each entry: {"summary": str, "source": "db" | "tavily_extract" | "tavily_search"}
    Example: {"greentech-london": {"summary": "...", "source": "tavily_extract"}}
    """

    # ------------------------------------------------------------------
    # ScoringAgent output  (Step 4)
    # Writes: scored_companies
    # ------------------------------------------------------------------

    scored_companies: list[dict]
    """
    Top-5 companies after Cross-encoder reranking, with LLM reasoning.
    Each dict has: all candidate_companies fields + cross_encoder_score + reasoning.
    """

    # ------------------------------------------------------------------
    # ReportAgent output  (Step 5)
    # Writes: report
    # ------------------------------------------------------------------

    report: str
    """Final Markdown report with Top-5 companies, match scores, and outreach drafts."""

    # ------------------------------------------------------------------
    # Pipeline metadata — written by any agent
    # ------------------------------------------------------------------

    errors: list[str]
    """Non-fatal errors encountered during the run (e.g. Tavily timeout for one company)."""
