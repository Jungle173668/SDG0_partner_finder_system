"""
ResearchAgent — Step 3 implementation.

Responsibilities:
  For each of the Top-10 candidate companies, gather information from
  three prioritised data layers:

    Layer 1 (always):         SDGZero database document field
                              (Company name, categories, description, summary,
                               achievements, SDG involvement — 100% coverage, free)
    Layer 2 (if Tavily key):  Tavily extract — crawls the company's website URL
                               handles JS rendering and anti-scraping transparently
    Layer 3 (fallback):       Tavily search — queries company name + "sustainability"
                               used when website URL is missing or returns no content

  All 20 companies are processed in parallel (ThreadPoolExecutor) to stay under 10s.

Output per company (keyed by slug):
  {
    "summary": "<~300-word enriched text, truncated>",
    "source":  "db" | "db+tavily_extract" | "db+tavily_search"
  }

Tavily setup:
  Add TAVILY_API_KEY=tvly_... to .env
  Free tier: 1000 calls/month → supports ~50 user searches/month

Graceful degradation:
  If TAVILY_API_KEY is absent or invalid, agent silently uses Layer 1 only.
  Pipeline continues — ScoringAgent can still work with DB text alone.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from agent.state import AgentState

logger = logging.getLogger(__name__)

_MAX_WORKERS = 10        # parallel Tavily calls (I/O-bound, threads work well)
_DB_SUMMARY_CHARS = 1200  # chars to keep from the DB document field (Layer 1)
_TAVILY_CHARS = 2500     # chars to keep from Tavily result
_COMBINED_CHARS = 4000   # max final summary length


# ---------------------------------------------------------------------------
# Tavily client — optional singleton
# ---------------------------------------------------------------------------

def _get_tavily_client() -> Optional[object]:
    """
    Return a TavilyClient if TAVILY_API_KEY is configured, else None.

    Failing silently here means the pipeline degrades to Layer 1 only
    rather than crashing when the key is missing.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key or api_key.startswith("tvly_your"):
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=api_key)
    except Exception as e:
        logger.warning(f"ResearchAgent: could not create TavilyClient: {e}")
        return None


# ---------------------------------------------------------------------------
# Layer implementations
# ---------------------------------------------------------------------------

def _build_db_summary(company: dict) -> str:
    """
    Layer 1: build a structured summary from the ChromaDB document field.

    The document contains the company's full embedding text:
      Company → Categories → Sector → City → Country →
      Description → Summary → Achievements → SDG involvement → SDGs
    """
    name = company.get("name", "Unknown")
    city = company.get("city", "")
    categories = company.get("categories", "")
    sdg_tags = company.get("sdg_tags", "") or company.get("predicted_sdg_tags", "")
    doc = company.get("document", "") or ""

    header_parts = [f"Company: {name}"]
    if city:
        header_parts.append(f"Location: {city}")
    if categories:
        header_parts.append(f"Categories: {categories}")
    if sdg_tags:
        header_parts.append(f"SDGs: {sdg_tags}")

    header = " | ".join(header_parts)
    truncated_doc = doc[:_DB_SUMMARY_CHARS].strip()
    if len(doc) > _DB_SUMMARY_CHARS:
        truncated_doc += "..."

    return f"{header}\n{truncated_doc}"


def _clean_web_content(raw: str) -> str:
    """
    Filter raw Tavily content to remove navigation noise before truncating.

    Websites always start with nav/header/logo, so a naive first-N-chars
    truncation captures nothing useful. This keeps only lines that look
    like real content (sentences, paragraphs) and discards:
      - Markdown image links  ![...](..)
      - Lines that are just a URL or markdown hyperlink
      - Lines shorter than 40 chars (nav items, phone numbers, headings)
      - Lines consisting only of punctuation / whitespace

    The remaining lines are joined and truncated to _TAVILY_CHARS.
    """
    import re
    lines = raw.splitlines()
    kept = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Drop markdown images
        if re.match(r"^!\[", line):
            continue
        # Drop lines that are just a markdown link (with or without list prefix)
        # Catches: [Text](url)  /  + [Text](url)  /  * [Text](url)
        if re.match(r"^[+\-*]?\s*\[.*\]\(.*\)$", line):
            continue
        # Drop bare URLs
        if re.match(r"^https?://\S+$", line):
            continue
        # Drop very short lines (nav items, phone, zip codes)
        if len(line) < 40:
            continue
        kept.append(line)

    cleaned = "\n".join(kept)
    # Fallback: if filtering removed everything, use raw (better than nothing)
    return cleaned if len(cleaned) > 100 else raw


def _try_tavily_extract(client, url: str) -> Optional[str]:
    """
    Layer 2: Tavily extract — fetch and parse the company's website.

    Tavily handles JS-rendered pages and common anti-scraping measures,
    returning the main textual content. The legal/compliance responsibility
    for the crawl rests with Tavily.

    Returns truncated raw_content string, or None on any failure.
    """
    if not url or not url.startswith("http"):
        return None
    try:
        result = client.extract(urls=[url])
        # Response: {"results": [{"url": ..., "raw_content": ...}], "failed_results": [...]}
        if result and result.get("results"):
            content = result["results"][0].get("raw_content", "")
            if content and len(content.strip()) > 100:
                return _clean_web_content(content)[:_TAVILY_CHARS].strip()
    except Exception as e:
        logger.debug(f"ResearchAgent: Tavily extract failed for {url}: {e}")
    return None


def _try_tavily_search(client, company_name: str) -> Optional[str]:
    """
    Layer 3: Tavily search — fallback when website URL is missing or fails.

    Searches "{company_name} sustainability" and combines top snippets.

    Returns combined snippet string, or None on any failure.
    """
    if not company_name:
        return None
    try:
        query = f"{company_name} sustainability SDG"
        result = client.search(query=query, max_results=3, search_depth="basic")
        # Response: {"results": [{"title": ..., "url": ..., "content": ...}]}
        if result and result.get("results"):
            snippets = []
            for r in result["results"][:3]:
                title = r.get("title", "")
                content = r.get("content", "")
                if content:
                    snippets.append(f"{title}: {content[:250]}")
            combined = "\n".join(snippets)
            if combined.strip():
                return combined[:_TAVILY_CHARS].strip()
    except Exception as e:
        logger.debug(f"ResearchAgent: Tavily search failed for '{company_name}': {e}")
    return None


# ---------------------------------------------------------------------------
# Per-company research (three-layer)
# ---------------------------------------------------------------------------

def _research_one_company(company: dict, tavily_client) -> tuple[str, dict]:
    """
    Research a single company using the three-layer strategy.

    Args:
        company:       Dict from candidate_companies (has slug, name, website, document).
        tavily_client: TavilyClient or None.

    Returns:
        (slug, {"summary": str, "source": str})
    """
    slug = company.get("slug") or company.get("id", "unknown")
    name = company.get("name", "Unknown")
    website = company.get("website", "") or ""

    # Layer 1 — always run
    db_summary = _build_db_summary(company)
    source = "db"
    tavily_content: Optional[str] = None

    if tavily_client is not None:
        # Layer 2: Tavily extract
        tavily_content = _try_tavily_extract(tavily_client, website)
        if tavily_content:
            source = "db+tavily_extract"
        else:
            # Layer 3: Tavily search fallback
            tavily_content = _try_tavily_search(tavily_client, name)
            if tavily_content:
                source = "db+tavily_search"

    if tavily_content:
        combined = f"{db_summary}\n\n--- Web Research ---\n{tavily_content}"
    else:
        combined = db_summary

    return slug, {
        "summary": combined[:_COMBINED_CHARS].strip(),
        "source": source,
    }


# ---------------------------------------------------------------------------
# ResearchAgent LangGraph node
# ---------------------------------------------------------------------------

def research_agent_node(state: AgentState) -> dict:
    """
    ResearchAgent node for the LangGraph pipeline.

    Reads:
        candidate_companies (list[dict], from SearchAgent)

    Writes:
        research_results — dict[slug → {summary, source}]
        errors           — appends any per-company failures (non-fatal)

    Parallel strategy:
        ThreadPoolExecutor with up to 10 workers.
        Tavily calls are I/O-bound so threads scale well.
        All 20 companies finish within ~5-8s (within 10s target).

    Graceful degradation:
        No TAVILY_API_KEY → Layer 1 only (still completes, just no web data).
        Individual company failure → logged + empty entry, pipeline continues.
    """
    candidates = state.get("candidate_companies", [])
    if not candidates:
        logger.info("ResearchAgent: no candidates to research")
        return {"research_results": {}}

    tavily_client = _get_tavily_client()
    if tavily_client:
        logger.info(
            f"ResearchAgent: Tavily active — researching {len(candidates)} companies "
            f"(Layer 1 + 2/3)"
        )
    else:
        logger.info(
            f"ResearchAgent: no TAVILY_API_KEY — using DB content only "
            f"for {len(candidates)} companies (Layer 1)"
        )

    research_results: dict = {}
    errors: list = list(state.get("errors") or [])

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_slug = {
            executor.submit(_research_one_company, company, tavily_client): (
                company.get("slug") or company.get("id", "unknown")
            )
            for company in candidates
        }
        for future in as_completed(future_to_slug):
            slug = future_to_slug[future]
            try:
                result_slug, result = future.result()
                research_results[result_slug] = result
            except Exception as e:
                logger.error(f"ResearchAgent: unexpected error for {slug}: {e}")
                errors.append(f"ResearchAgent: {slug} failed — {e}")
                research_results[slug] = {"summary": "", "source": "error"}

    # Log source distribution
    sources: dict = {}
    for v in research_results.values():
        src = v.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    logger.info(
        f"ResearchAgent: complete — {len(research_results)} companies. "
        f"Sources: {sources}"
    )

    return {"research_results": research_results, "errors": errors}
