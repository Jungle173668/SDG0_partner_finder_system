"""
HITL End-to-End Test Script
============================
Tests: initial search → 1x refine → compare results

Usage (backend must be running on :8000):
    python tests/test_hitl_flow.py
    python tests/test_hitl_flow.py --scenario 2   # run one scenario
    python tests/test_hitl_flow.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BASE = "http://localhost:8000"
POLL_INTERVAL = 4   # seconds between status polls
POLL_TIMEOUT  = 180 # seconds before giving up

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "Change partner type via text",
        "initial": {
            "user_company_desc": "We are a sustainable skincare company focused on natural ingredients.",
            "partner_type_desc": "marketing agency",
            "other_requirements": "",
        },
        "refine": {
            "liked":    [],
            "disliked": [],
            "user_text": "Actually I want travel companies, not marketing agencies",
        },
    },
    {
        "name": "Add city filter (London — likely fallback)",
        "initial": {
            "user_company_desc": "We are a sports nutrition brand targeting amateur athletes.",
            "partner_type_desc": "gyms and fitness centres",
            "other_requirements": "",
        },
        "refine": {
            "liked":    [],
            "disliked": [],
            "user_text": "Only show companies in London",
        },
    },
    {
        "name": "Like / dislike companies",
        "initial": {
            "user_company_desc": "We run a zero-waste food packaging startup.",
            "partner_type_desc": "retail or distribution partner",
            "other_requirements": "",
        },
        "refine": {
            "liked":    [],     # filled dynamically from first result
            "disliked": [],     # filled dynamically from last result
            "user_text": "",
        },
        "dynamic_feedback": True,  # populate liked/disliked from actual results
    },
    {
        "name": "Add SDG filter + exclude sector",
        "initial": {
            "user_company_desc": "We provide corporate sustainability consulting.",
            "partner_type_desc": "technology or data companies",
            "other_requirements": "",
        },
        "refine": {
            "liked":    [],
            "disliked": [],
            "user_text": "Add SDG 13 Climate Action as a filter, and exclude public sector companies",
        },
    },
    {
        "name": "Verified only + city change",
        "initial": {
            "user_company_desc": "We make biodegradable packaging for e-commerce.",
            "partner_type_desc": "logistics or delivery companies",
            "other_requirements": "",
            "city": {"value": "Manchester", "mode": "hard"},
        },
        "refine": {
            "liked":    [],
            "disliked": [],
            "user_text": "Switch to Edinburgh, verified profiles only",
        },
    },
]

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def start_search(base: str, params: dict) -> str:
    """POST /api/search → return session_id."""
    r = requests.post(f"{base}/api/search", json=params, timeout=30)
    r.raise_for_status()
    return r.json()["session_id"]


def poll_until_done(base: str, session_id: str) -> dict:
    """Poll GET /api/search/{id} until status == done or error."""
    deadline = time.time() + POLL_TIMEOUT
    dots = 0
    while time.time() < deadline:
        r = requests.get(f"{base}/api/search/{session_id}", timeout=15)
        r.raise_for_status()
        data = r.json()
        if data["status"] == "done":
            print()  # newline after dots
            return data
        if data["status"] == "error":
            print()
            raise RuntimeError(f"Pipeline error for session {session_id}")
        print(".", end="", flush=True)
        dots += 1
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Timed out waiting for session {session_id}")


def refine(base: str, session_id: str, liked: list, disliked: list, user_text: str) -> dict:
    """POST /api/refine/{id} → return RefineResponse."""
    payload = {"liked": liked, "disliked": disliked, "user_text": user_text}
    r = requests.post(f"{base}/api/refine/{session_id}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fmt_company(c: dict, rank: int) -> str:
    name   = c.get("name", "Unknown")
    city   = c.get("city", "?")
    cats   = c.get("categories", "")[:40]
    pct    = round(c.get("cross_encoder_score", 0) * 100)
    qual   = c.get("match_quality", "")
    sdgs   = c.get("sdg_tags", "")[:40]
    qual_icon = {"strong": "✅", "partial": "🟡", "fallback": "🔴"}.get(qual, "❓")
    return f"  {rank}. {qual_icon} {pct:>3}%  {name} ({city})  [{cats}]  {sdgs}"


def fallback_warnings(result: dict) -> list[str]:
    """
    Detect mismatches between requested hard filters and actual results.
    Warns when 0 matches (hard mismatch) or < 50% match (partial mismatch).
    Checks sdg_tags + predicted_sdg_tags for SDG coverage.
    """
    warnings = []
    filters   = result.get("filters", {})
    companies = result.get("scored_companies", [])
    n = len(companies)

    if not companies:
        return warnings

    # If no filters were applied at all, note it (helps diagnose refine_agent failures)
    if not filters:
        # Check if partner_type_desc mentions SDG — sign that refine_agent put SDG in wrong field
        ptd = result.get("partner_type_desc", "").lower()
        sdg_keywords = ["sdg", "climate action", "clean energy", "zero hunger", "no poverty",
                        "good health", "quality education", "gender equality", "clean water",
                        "decent work", "innovation", "reduced inequalit", "sustainable cities",
                        "responsible consumption", "life below water", "life on land",
                        "peace, justice", "partnerships for"]
        if any(kw in ptd for kw in sdg_keywords):
            warnings.append("ℹ  No filters applied — SDG keyword detected in partner_type_desc. "
                            "Refine agent may have put SDG in the wrong field (should be sdg_tags filter).")
        return warnings

    def _pct(matched: int) -> str:
        return f"{matched}/{n}"

    # ── City ──────────────────────────────────────────────────────────────
    req_city = filters.get("city")
    if req_city:
        city_str = req_city if isinstance(req_city, str) else ", ".join(req_city)
        matched = sum(
            1 for c in companies
            if str(c.get("city", "")).strip().lower() == city_str.lower()
        )
        if matched == 0:
            warnings.append(f"⚠  City='{city_str}': 0/{n} results in that city — filter was dropped, showing alternatives")
        elif matched < n // 2 + 1:
            warnings.append(f"⚠  City='{city_str}': only {_pct(matched)} results actually in that city")

    # ── Claimed / verified ────────────────────────────────────────────────
    if filters.get("claimed"):
        matched = sum(
            1 for c in companies
            if str(c.get("claimed", "")).lower() in ("yes", "true", "1")
        )
        if matched == 0:
            warnings.append(f"⚠  Verified-only: 0/{n} results are verified — filter was dropped")
        elif matched < n // 2 + 1:
            warnings.append(f"⚠  Verified-only: only {_pct(matched)} results are actually verified")

    # ── SDG tags (checks both sdg_tags AND predicted_sdg_tags) ───────────
    req_sdgs = filters.get("sdg_tags")
    if req_sdgs:
        req_set = {s.lower().strip() for s in (req_sdgs if isinstance(req_sdgs, list) else [req_sdgs])}
        def _sdg_text(c):
            raw = f"{c.get('sdg_tags', '')} {c.get('predicted_sdg_tags', '')}".lower()
            return raw
        matched = sum(1 for c in companies if any(s in _sdg_text(c) for s in req_set))
        if matched == 0:
            warnings.append(f"⚠  SDG='{req_sdgs}': 0/{n} results carry that SDG — filter was relaxed (checked sdg_tags + predicted_sdg_tags)")
        elif matched < n // 2 + 1:
            warnings.append(f"⚠  SDG='{req_sdgs}': only {_pct(matched)} results carry that SDG (checked sdg_tags + predicted_sdg_tags)")

    # ── Categories ────────────────────────────────────────────────────────
    req_cats = filters.get("categories")
    if req_cats:
        req_set = {s.lower().strip() for s in (req_cats if isinstance(req_cats, list) else [req_cats])}
        matched = sum(
            1 for c in companies
            if any(s in str(c.get("categories", "")).lower() for s in req_set)
        )
        if matched == 0:
            warnings.append(f"⚠  Categories='{req_cats}': 0/{n} results match — filter was relaxed")
        elif matched < n // 2 + 1:
            warnings.append(f"⚠  Categories='{req_cats}': only {_pct(matched)} results match that category")

    # ── Business type ─────────────────────────────────────────────────────
    req_btype = filters.get("business_type")
    if req_btype:
        matched = sum(
            1 for c in companies
            if str(c.get("business_type", "")).lower() == req_btype.lower()
        )
        if matched == 0:
            warnings.append(f"⚠  Business type='{req_btype}': 0/{n} results match — filter was relaxed")
        elif matched < n // 2 + 1:
            warnings.append(f"⚠  Business type='{req_btype}': only {_pct(matched)} results match")

    # ── Job sector ────────────────────────────────────────────────────────
    req_sector = filters.get("job_sector")
    if req_sector:
        matched = sum(
            1 for c in companies
            if str(c.get("job_sector", "")).lower() == req_sector.lower()
        )
        if matched == 0:
            warnings.append(f"⚠  Job sector='{req_sector}': 0/{n} results match — filter was relaxed")
        elif matched < n // 2 + 1:
            warnings.append(f"⚠  Job sector='{req_sector}': only {_pct(matched)} results match")

    # ── Company size ──────────────────────────────────────────────────────
    req_size = filters.get("company_size")
    if req_size:
        matched = sum(
            1 for c in companies
            if str(c.get("company_size", "")).lower() == req_size.lower()
        )
        if matched == 0:
            warnings.append(f"⚠  Company size='{req_size}': 0/{n} results match — filter was relaxed")
        elif matched < n // 2 + 1:
            warnings.append(f"⚠  Company size='{req_size}': only {_pct(matched)} results match")

    return warnings


def print_results(label: str, result: dict):
    companies = result.get("scored_companies", [])
    fbl = result.get("search_fallback_level", 0)
    filters = result.get("filters", {})

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Session : {result.get('session_id', '?')}")
    print(f"  Searched: {result.get('partner_type_desc', '(none)')}")
    print(f"  Filters : {json.dumps(filters) if filters else '(none)'}")
    print(f"  Fallback: level {fbl}")
    print()

    if not companies:
        print("  (no results)")
    else:
        for i, c in enumerate(companies, 1):
            print(fmt_company(c, i))

    warns = fallback_warnings(result)
    if warns:
        print()
        for w in warns:
            print(f"  {w}")


def print_refine_action(action: str, summary: str, changes: dict):
    print(f"\n{'═'*60}")
    print(f"  REFINE → action={action}")
    if summary:
        print(f"  Summary: {summary}")
    if changes:
        print(f"  Changes:")
        for k, v in changes.items():
            print(f"    {k}: {v}")
    print(f"{'═'*60}")


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------

def run_scenario(base: str, idx: int, scenario: dict):
    name = scenario["name"]
    print(f"\n{'#'*60}")
    print(f"  SCENARIO {idx+1}: {name}")
    print(f"{'#'*60}")

    # ── Step 1: initial search ────────────────────────────────────────────
    print(f"\n[1/4] Starting initial search...")
    sid1 = start_search(base, scenario["initial"])
    print(f"      session_id = {sid1}  (polling", end="")
    result1 = poll_until_done(base, sid1)
    print_results("BEFORE (initial search)", result1)

    # ── Step 2: build refine payload ──────────────────────────────────────
    companies = result1.get("scored_companies", [])
    refine_cfg = dict(scenario["refine"])

    if scenario.get("dynamic_feedback") and companies:
        best  = companies[0]
        worst = companies[-1]
        refine_cfg["liked"]    = [{"name": best["name"],  "categories": best.get("categories"),  "city": best.get("city")}]
        refine_cfg["disliked"] = [{"name": worst["name"], "categories": worst.get("categories"), "city": worst.get("city")}]
        print(f"\n[2/4] Dynamic feedback:")
        print(f"      ✓ Liked:    {best['name']}")
        print(f"      ✗ Disliked: {worst['name']}")
    else:
        print(f"\n[2/4] Refine input: \"{refine_cfg['user_text']}\"")

    # ── Step 3: call refine agent ─────────────────────────────────────────
    print(f"\n[3/4] Calling refine agent...")
    refine_resp = refine(base, sid1,
                         liked=refine_cfg["liked"],
                         disliked=refine_cfg["disliked"],
                         user_text=refine_cfg["user_text"])

    action  = refine_resp.get("action", "unknown")
    summary = refine_resp.get("summary", "")
    changes = refine_resp.get("new_search_params", {})
    print_refine_action(action, summary, {k: v for k, v in changes.items()
                                          if k not in ("user_company_desc", "parent_id")})

    if action != "refine":
        print(f"\n  ⚠ Refine returned '{action}' — skipping second search")
        return

    # ── Step 4: run refined search ────────────────────────────────────────
    print(f"\n[4/4] Starting refined search...")
    sid2 = start_search(base, changes)
    print(f"      session_id = {sid2}  (polling", end="")
    result2 = poll_until_done(base, sid2)
    print_results("AFTER (refined search)", result2)

    # ── Summary diff ──────────────────────────────────────────────────────
    print(f"\n  📊 DIFF SUMMARY")
    names1 = [c["name"] for c in (result1.get("scored_companies") or [])]
    names2 = [c["name"] for c in (result2.get("scored_companies") or [])]
    new_in  = [n for n in names2 if n not in names1]
    dropped = [n for n in names1 if n not in names2]
    if new_in:
        print(f"  + New in results  : {', '.join(new_in)}")
    if dropped:
        print(f"  - Dropped from results: {', '.join(dropped)}")
    if not new_in and not dropped:
        print(f"  = Same companies returned (only ordering/scores may differ)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HITL end-to-end test")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--scenario", type=int, default=None,
                        help="Run only scenario N (1-indexed)")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # Health check
    try:
        r = requests.get(f"{base}/api/health", timeout=5)
        r.raise_for_status()
        print(f"✓ Backend healthy at {base}")
    except Exception as e:
        print(f"✗ Cannot reach backend at {base}: {e}")
        sys.exit(1)

    scenarios_to_run = (
        [SCENARIOS[args.scenario - 1]] if args.scenario
        else SCENARIOS
    )
    indices = (
        [args.scenario - 1] if args.scenario
        else list(range(len(SCENARIOS)))
    )

    passed = failed = 0
    for idx, scenario in zip(indices, scenarios_to_run):
        try:
            run_scenario(base, idx, scenario)
            passed += 1
        except Exception as e:
            print(f"\n  ✗ SCENARIO FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed / {passed+failed} total")
    print(f"{'='*60}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
