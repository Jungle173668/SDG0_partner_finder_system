"""
Demo: run the Multi-Agent Pipeline (Steps 1-3).

Modes:
  python demo_agent.py                          # interactive mode (prompts for input)
  python demo_agent.py "Your company desc"      # description only, no filters
  python demo_agent.py --schema                 # print all valid filter values and exit
"""

import sys
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s [%(name)s] %(message)s",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import run_pipeline
from agent.schema_cache import get_schema, print_schema


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _pick(prompt: str, options: list[str], allow_empty: bool = True) -> str:
    """
    Print numbered options and let the user pick by number or type a value.
    Returns the selected string, or "" if skipped.
    """
    if not options:
        return input(f"{prompt} (free text, Enter to skip): ").strip()

    print(f"\n  {prompt}")
    print("  [0] Skip (leave blank)")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")

    while True:
        choice = input("  Enter number or value: ").strip()
        if not choice:
            return ""
        if choice == "0":
            return ""
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            print(f"  Please enter 0–{len(options)}")
        else:
            # Accept free-text if it roughly matches an option
            return choice


def _pick_multi(prompt: str, options: list[str]) -> list[str]:
    """
    Let the user pick multiple options (comma-separated numbers).
    Returns list of selected strings (may be empty).
    """
    if not options:
        return []

    print(f"\n  {prompt} (comma-separated numbers, or Enter to skip)")
    print("  [0] Skip")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")

    raw = input("  Enter numbers: ").strip()
    if not raw or raw == "0":
        return []

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(options):
                selected.append(options[idx - 1])
    return selected


def interactive_input() -> tuple[str, str, dict, str]:
    """
    Interactively collect user_company_desc, partner_type_desc, filters, other_requirements.
    Returns (desc, partner_type, filters, other_requirements).
    """
    schema = get_schema()

    print("\n" + "=" * 65)
    print("  SDGZero Partner Finder — Input")
    print("=" * 65)

    # --- Description (required) ---
    print("\n  Describe your company (what you do, your sector, SDG focus).")
    print("  Example: 'We provide carbon auditing and net-zero consulting")
    print("            for UK SMEs, focusing on Scope 1–3 emissions.'")
    desc = ""
    while not desc.strip():
        desc = input("\n  Your company description: ").strip()
        if not desc:
            print("  (Description is required)")

    # --- Partner type (optional but high-impact) ---
    print("\n  What kind of company are you looking to partner with?")
    print("  Example: 'a media or marketing agency to help promote our brand'")
    print("  (Leave blank to let the AI infer this from your description)")
    partner_type = input("\n  Target partner type (Enter to skip): ").strip()

    # --- Optional filters ---
    print("\n  --- Optional Filters (Enter or [0] to skip any) ---")

    filters = {}

    city = _pick("Filter by city:", schema["city"])
    if city:
        filters["city"] = city

    bt = _pick("Filter by business type:", schema["business_type"])
    if bt:
        filters["business_type"] = bt

    sector = _pick("Filter by job sector:", schema["job_sector"])
    if sector:
        filters["job_sector"] = sector

    sdgs = _pick_multi("Filter by SDG tags (keep companies matching ANY):", schema["sdg_tags"])
    if sdgs:
        filters["sdg_tags"] = sdgs

    claimed_input = input("\n  Only show claimed/verified profiles? [y/N]: ").strip().lower()
    if claimed_input == "y":
        filters["claimed"] = True

    # --- Other requirements ---
    print("\n  Any additional requirements? (free text, e.g. 'prefer under 50 employees')")
    other = input("  Other requirements (Enter to skip): ").strip()

    return desc, partner_type, filters, other


# ---------------------------------------------------------------------------
# Output printer
# ---------------------------------------------------------------------------

_QUALITY_ICON = {"strong": "✅", "partial": "◐ ", "fallback": "⚠️ "}
_FALLBACK_MSG  = {
    0: "",
    1: "⚠️  Filters partially relaxed (SDG / claimed conditions dropped)",
    2: "⚠️  No matches found within your filters — showing semantic matches only",
}


def print_results(state: dict, verbose: bool = False) -> None:
    W = 65

    # ── Header ──────────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  PIPELINE RESULTS")
    print("=" * W)

    fallback_lvl = state.get("search_fallback_level", 0)
    print(f"  Session  : {state.get('session_id')}")
    print(f"  Search   : {state.get('search_method', 'N/A')}  "
          f"(fallback level {fallback_lvl})")
    if _FALLBACK_MSG.get(fallback_lvl):
        print(f"  {_FALLBACK_MSG[fallback_lvl]}")

    # ── HyDE (verbose only) ─────────────────────────────────────────────────
    if verbose:
        hyde = state.get("hypothetical_partner_desc", "")
        if hyde:
            print(f"\n  HyDE — ideal partner profile:")
            for ln in _wrap(hyde, 60):
                print(f"    {ln}")
        expansions = state.get("query_expansions", [])
        if expansions:
            print(f"\n  Query expansions:")
            for i, q in enumerate(expansions, 1):
                print(f"    {i}. {q}")

    # ── Candidate pool (verbose only) ───────────────────────────────────────
    if verbose:
        candidates = state.get("candidate_companies", [])
        research   = state.get("research_results", {})
        print(f"\n  ── Bi-encoder candidates ({len(candidates)}) ──────────────")
        for i, c in enumerate(candidates, 1):
            slug   = c.get("slug", "")
            src    = research.get(slug, {}).get("source", "—")
            sim    = c.get("similarity")
            sim_s  = f"sim={sim:.3f}" if sim is not None else "sql"
            sdgs   = (c.get("sdg_tags") or c.get("predicted_sdg_tags") or "")[:50]
            print(f"  [{i:2d}] {c['name'][:40]:<40}  {sim_s}")
            print(f"        {c.get('city',''):<15} | {c.get('categories','')[:28]}")
            if sdgs:
                print(f"        SDGs: {sdgs}")
            print(f"        Research: {src}")

    # ── Scored Top-5 ────────────────────────────────────────────────────────
    scored = state.get("scored_companies", [])
    if not scored:
        print("\n  (ScoringAgent produced no results)")
    else:
        print(f"\n  ── Top-{len(scored)} Recommendations (Cross-encoder ranked) ──")
        for i, c in enumerate(scored, 1):
            score   = c.get("cross_encoder_score", 0.0)
            quality = c.get("match_quality", "fallback")
            icon    = _QUALITY_ICON.get(quality, "  ")
            pct     = int(score * 100)
            bar     = "█" * (pct // 5) + "░" * (20 - pct // 5)

            name    = c.get("name", "Unknown")
            city    = c.get("city", "")
            cats    = c.get("categories", "")[:30]
            sdgs    = (c.get("sdg_tags") or c.get("predicted_sdg_tags") or "")[:50]
            website = c.get("website", "")
            linkedin= c.get("linkedin", "")
            src     = state.get("research_results", {}).get(
                          c.get("slug",""), {}).get("source", "—")

            print(f"\n  {'─'*W}")
            print(f"  #{i}  {icon} {name}  —  {pct}% match  [{quality}]")
            print(f"       {bar}")
            print(f"       {city}  |  {cats}")
            if sdgs:
                print(f"       SDGs: {sdgs}")
            print(f"       Research source: {src}")

            reasoning = c.get("reasoning", "")
            if reasoning:
                print(f"\n       Recommendation:")
                for ln in _wrap(reasoning, 58):
                    print(f"         {ln}")

            entry_points = c.get("entry_points", [])
            if entry_points:
                print(f"\n       Collaboration entry points:")
                for ep in entry_points:
                    print(f"         • {ep}")

            if website or linkedin:
                contacts = "  |  ".join(filter(None, [website, linkedin]))
                print(f"\n       Contact: {contacts}")

    # ── Errors ──────────────────────────────────────────────────────────────
    errors = state.get("errors", [])
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    • {e}")

    print("\n" + "=" * W)


def _wrap(text: str, width: int = 60) -> list[str]:
    words = text.split()
    lines, current = [], []
    for word in words:
        if sum(len(w) + 1 for w in current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if "--schema" in sys.argv:
        print_schema()
        sys.exit(0)

    clean_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if clean_args:
        # Non-interactive: description passed as argument, no filters
        user_desc = " ".join(clean_args)
        partner_type = ""
        filters = {}
        other = ""
        print(f"\nRunning pipeline for: {user_desc[:80]}...")
    else:
        # Interactive mode
        user_desc, partner_type, filters, other = interactive_input()

    print(f"\n  Filters applied: {filters if filters else 'none'}")
    print("\nRunning pipeline...\n")

    state = run_pipeline(
        user_company_desc=user_desc,
        partner_type_desc=partner_type,
        filters=filters,
        other_requirements=other,
    )

    print_results(state, verbose=verbose)
