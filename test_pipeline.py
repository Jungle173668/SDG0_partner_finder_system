"""
Automated pipeline test runner.

Runs a set of pre-defined test cases through the full Multi-Agent Pipeline
and saves one HTML report per case to reports/.

Usage:
    python test_pipeline.py              # run all cases
    python test_pipeline.py --case 2     # run only case #2
    python test_pipeline.py --list       # print all cases and exit

Each test case exercises a different combination of inputs:
  - with / without partner_type_desc
  - with / without filters
  - different company sectors
"""

import sys
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING"),   # quiet during batch runs
    format="%(levelname)s [%(name)s] %(message)s",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import run_pipeline

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "id": 1,
        "name": "Construction firm — seeks waste management partner",
        "desc": "No filters, no partner_type — pure semantic. DB has 51 Waste & Recycling companies.",
        "user_company_desc": (
            "We are a commercial construction and demolition company in London. "
            "We generate significant volumes of building waste and rubble on every project "
            "and want to reduce landfill, improve recycling rates, and meet our sustainability targets."
        ),
        "partner_type_desc": "",
        "filters": {},
        "other_requirements": "",
    },
    {
        "id": 2,
        "name": "Construction firm — seeks waste partner in London (hybrid search)",
        "desc": "Same as Case 1 but with city=London filter to test hybrid_search path.",
        "user_company_desc": (
            "We are a commercial construction and demolition company in London. "
            "We generate significant volumes of building waste and rubble on every project "
            "and want to reduce landfill, improve recycling rates, and meet our sustainability targets."
        ),
        "partner_type_desc": "a waste collection or recycling company serving commercial clients",
        "filters": {
            "city": "London",
        },
        "other_requirements": "",
    },
    {
        "id": 3,
        "name": "Bridal boutique — seeks beauty & wellness partner",
        "desc": "Beauty & Personal Care has 60 companies in DB (Lancashire). No filters.",
        "user_company_desc": (
            "We are a premium bridal fashion boutique in Blackburn. "
            "We dress brides for their wedding day and want to offer complete bridal packages "
            "by partnering with beauty and wellness providers."
        ),
        "partner_type_desc": "a beauty salon, aesthetics clinic, or hair and makeup studio",
        "filters": {},
        "other_requirements": "",
    },
    {
        "id": 4,
        "name": "Corporate events firm — seeks London arts venue",
        "desc": "Arts has 61 companies in DB (London). Tests city filter + partner_type.",
        "user_company_desc": (
            "We organise corporate hospitality events, team-building days, and client entertainment "
            "for London-based businesses. We are looking for unique cultural venues to host our events."
        ),
        "partner_type_desc": "an art gallery, museum, or cultural venue available for private hire",
        "filters": {
            "city": "London",
        },
        "other_requirements": "",
    },
    {
        "id": 5,
        "name": "Sports nutrition brand — seeks London sports club",
        "desc": "Sports & Recreation has 52 companies. Tests London filter + SDG filter.",
        "user_company_desc": (
            "We produce plant-based sports nutrition products — protein powders, energy bars, "
            "and recovery drinks. We want to partner with sports clubs and fitness facilities "
            "to distribute our products and sponsor grassroots athletes."
        ),
        "partner_type_desc": "",
        "filters": {
            "city": "London",
            "sdg_tags": ["Decent Work And Economic Growth"],
        },
        "other_requirements": "",
    },
    {
        "id": 6,
        "name": "Social enterprise — seeks Inverness community org",
        "desc": "Community & Social Purpose has 49 companies (mostly Inverness). Tests city filter.",
        "user_company_desc": (
            "We are a social enterprise providing digital skills training and employment support "
            "for unemployed adults in the Scottish Highlands. "
            "We want to partner with local community organisations to reach more people."
        ),
        "partner_type_desc": "",
        "filters": {
            "city": "Inverness",
        },
        "other_requirements": "",
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _divider(char: str = "─", width: int = 65) -> str:
    return char * width


def run_case(case: dict) -> dict:
    """Run one test case and return result summary."""
    print(f"\n{_divider('═')}")
    print(f"  Case #{case['id']}: {case['name']}")
    print(f"  {case['desc']}")
    print(_divider())

    t0 = time.time()
    try:
        state = run_pipeline(
            user_company_desc=case["user_company_desc"],
            partner_type_desc=case.get("partner_type_desc", ""),
            filters=case.get("filters", {}),
            other_requirements=case.get("other_requirements", ""),
        )
        elapsed = round(time.time() - t0, 1)

        scored = state.get("scored_companies", [])
        errors = state.get("errors", [])
        report = state.get("report", "")
        fallback = state.get("search_fallback_level", 0)

        print(f"  ✓ Done in {elapsed}s  |  fallback_level={fallback}  |  errors={len(errors)}")
        print(f"  Top-{len(scored)} scores & quality:")
        for i, c in enumerate(scored, 1):
            score   = c.get("cross_encoder_score", 0)
            quality = c.get("match_quality", "?")
            name    = c.get("name", "?")[:40]
            icon    = {"strong": "✅", "partial": "◐ ", "fallback": "⚠️ "}.get(quality, "  ")
            print(f"    #{i} {icon} {score:.3f}  {name}")

        if errors:
            print(f"  Errors:")
            for e in errors:
                print(f"    • {e}")

        if report:
            print(f"  Report → {report}")

        return {"case_id": case["id"], "ok": True, "elapsed": elapsed,
                "n_scored": len(scored), "errors": errors, "report": report}

    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"  ✗ FAILED in {elapsed}s: {e}")
        return {"case_id": case["id"], "ok": False, "elapsed": elapsed, "error": str(e)}


def print_summary(results: list[dict]) -> None:
    print(f"\n{_divider('═')}")
    print("  SUMMARY")
    print(_divider())
    ok  = [r for r in results if r.get("ok")]
    err = [r for r in results if not r.get("ok")]
    print(f"  Passed: {len(ok)}/{len(results)}  |  Failed: {len(err)}")
    for r in results:
        status = "✓" if r.get("ok") else "✗"
        name   = next((c["name"] for c in CASES if c["id"] == r["case_id"]), "?")
        print(f"    {status}  Case #{r['case_id']}: {name}  ({r['elapsed']}s)")
    print(_divider('═'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\nAvailable test cases:")
        for c in CASES:
            pt = f"  partner_type: {c['partner_type_desc'][:50]}…" if c["partner_type_desc"] else ""
            fi = f"  filters: {c['filters']}" if c["filters"] else ""
            print(f"  #{c['id']}  {c['name']}")
            print(f"       {c['desc']}")
            if pt: print(f"      {pt}")
            if fi: print(f"      {fi}")
        sys.exit(0)

    # Select which cases to run
    case_arg = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--case" and i + 1 < len(sys.argv) - 1:
            case_arg = int(sys.argv[i + 2])
            break

    if case_arg is not None:
        cases_to_run = [c for c in CASES if c["id"] == case_arg]
        if not cases_to_run:
            print(f"Case #{case_arg} not found. Use --list to see available cases.")
            sys.exit(1)
    else:
        cases_to_run = CASES

    print(f"\nRunning {len(cases_to_run)} test case(s)...\n")

    results = []
    for case in cases_to_run:
        result = run_case(case)
        results.append(result)

    if len(results) > 1:
        print_summary(results)
