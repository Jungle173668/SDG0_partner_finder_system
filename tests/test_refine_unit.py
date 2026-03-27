"""
Unit tests for refine pipeline — no LLM calls, no DB, no server needed.

Tests cover:
  - _validate_b_class   (schema validation + rejected messages)
  - _fix_misplaced_fields (code-level LLM mistake correction)
  - explicit_removals logic (None/[] removal detection)
  - new_search_params merging (filter inheritance + override)
  - Edge cases: empty changes, multi-field remove, claimed boolean

Run:
    python -m pytest tests/test_refine_unit.py -v
    python tests/test_refine_unit.py          (no pytest needed)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.refine import _validate_b_class, _fix_misplaced_fields

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

SCHEMA = {
    "city":          ["London", "Manchester", "Birmingham", "Blackburn", "Preston"],
    "sdg_tags":      ["Climate Action", "Good Health And Well-Being", "Affordable And Clean Energy",
                      "Quality Education", "Zero Hunger"],
    "categories":    ["Tourism & Travel", "Marketing & Advertising", "Health & Wellbeing",
                      "Technology & Innovation", "Arts & Culture"],
    "business_type": ["B2B", "B2C", "Both"],
    "job_sector":    ["Private", "Public", "Third Sector"],
    "company_size":  ["Micro", "SME", "Large"],
}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
        results.append((name, True, ""))
    else:
        print(f"  {FAIL} {name}  ← {detail}")
        results.append((name, False, detail))


# ===========================================================================
# 1. _validate_b_class
# ===========================================================================

print("\n── 1. _validate_b_class ──────────────────────────────────────────────")

# 1-1 Exact city match
out, rej = _validate_b_class({"city": "London"}, SCHEMA)
check("city exact match", out.get("city") == "London" and not rej)

# 1-2 Lowercase city → normalised
out, rej = _validate_b_class({"city": "london"}, SCHEMA)
check("city lowercase → normalised", out.get("city") == "London", f"got {out.get('city')}")

# 1-3 Substring city match ("Manch" → "Manchester")
out, rej = _validate_b_class({"city": "Manch"}, SCHEMA)
check("city substring match", out.get("city") == "Manchester", f"got {out.get('city')}")

# 1-4 Invalid city → rejected, removed from changes
out, rej = _validate_b_class({"city": "New York"}, SCHEMA)
check("invalid city → rejected", "city" not in out and len(rej) == 1,
      f"city={out.get('city')} rejected={rej}")

# 1-5 Valid SDG tag
out, rej = _validate_b_class({"sdg_tags": ["Climate Action"]}, SCHEMA)
check("sdg_tags valid", out.get("sdg_tags") == ["Climate Action"] and not rej)

# 1-6 Mixed valid + invalid SDG tags
out, rej = _validate_b_class({"sdg_tags": ["Climate Action", "FAKE_SDG"]}, SCHEMA)
check("sdg_tags partial valid", out.get("sdg_tags") == ["Climate Action"] and len(rej) == 1,
      f"sdg={out.get('sdg_tags')} rej={rej}")

# 1-7 All invalid SDG → field removed
out, rej = _validate_b_class({"sdg_tags": ["FAKE1", "FAKE2"]}, SCHEMA)
check("sdg_tags all invalid → removed", "sdg_tags" not in out and len(rej) == 1,
      f"sdg={out.get('sdg_tags')}")

# 1-8 SDG as string (not list) → still validated
out, rej = _validate_b_class({"sdg_tags": "Climate Action"}, SCHEMA)
check("sdg_tags as string → accepted", out.get("sdg_tags") == ["Climate Action"])

# 1-9 business_type B2B
out, rej = _validate_b_class({"business_type": "B2B"}, SCHEMA)
check("business_type B2B", out.get("business_type") == "B2B")

# 1-10 business_type lowercase
out, rej = _validate_b_class({"business_type": "b2b"}, SCHEMA)
check("business_type lowercase b2b → B2B", out.get("business_type") == "B2B",
      f"got {out.get('business_type')}")

# 1-11 business_type invalid
out, rej = _validate_b_class({"business_type": "enterprise"}, SCHEMA)
check("business_type invalid → rejected", "business_type" not in out and rej)

# 1-12 categories partial match
out, rej = _validate_b_class({"categories": ["Tourism & Travel", "FAKE"]}, SCHEMA)
check("categories partial valid", out.get("categories") == ["Tourism & Travel"] and rej)

# 1-13 Multiple fields in one call
out, rej = _validate_b_class({"city": "London", "business_type": "B2B", "sdg_tags": ["Climate Action"]}, SCHEMA)
check("multi-field valid", out == {"city": "London", "business_type": "B2B", "sdg_tags": ["Climate Action"]} and not rej)

# 1-14 Empty changes dict → no-op
out, rej = _validate_b_class({}, SCHEMA)
check("empty changes → no-op", out == {} and not rej)

# 1-15 No schema available → pass through
out, rej = _validate_b_class({"city": "Anywhere"}, {})
check("empty schema → pass through", out.get("city") == "Anywhere" and not rej)


# ===========================================================================
# 2. _fix_misplaced_fields
# ===========================================================================

print("\n── 2. _fix_misplaced_fields ──────────────────────────────────────────")

# 2-1 City in partner_type_desc → extracted
out = _fix_misplaced_fields({"partner_type_desc": "companies in London"}, SCHEMA)
check("city extracted from partner_type_desc", out.get("city") == "London",
      f"got {out.get('city')}")

# 2-2 City already set → not overwritten
out = _fix_misplaced_fields({"partner_type_desc": "companies in London", "city": "Manchester"}, SCHEMA)
check("city already set → not overwritten", out.get("city") == "Manchester")

# 2-3 SDG name in partner_type_desc → extracted
out = _fix_misplaced_fields({"partner_type_desc": "climate action focused companies"}, SCHEMA)
check("SDG extracted from partner_type_desc", "sdg_tags" in out and "Climate Action" in out["sdg_tags"],
      f"got {out.get('sdg_tags')}")

# 2-4 SDG number pattern "sdg 13" → extracted
out = _fix_misplaced_fields({"partner_type_desc": "companies that focus on sdg 13"}, SCHEMA)
check("SDG number pattern extracted", "sdg_tags" in out,
      f"got {out.get('sdg_tags')}")

# 2-5 Claimed keyword in other_requirements → extracted
out = _fix_misplaced_fields({"other_requirements": "verified only companies"}, SCHEMA)
check("claimed extracted from other_requirements", out.get("claimed") is True)

# 2-6 Claimed already set → not overwritten
out = _fix_misplaced_fields({"other_requirements": "verified only", "claimed": False}, SCHEMA)
check("claimed already set → not overwritten", out.get("claimed") is False)

# 2-7 No misplaced fields → no-op
changes = {"partner_type_desc": "software companies", "other_requirements": "prefer B2B"}
out = _fix_misplaced_fields(changes, SCHEMA)
check("no misplaced fields → no-op", "city" not in out and "sdg_tags" not in out and "claimed" not in out)

# 2-8 Original dict not mutated
original = {"partner_type_desc": "companies in Manchester"}
_ = _fix_misplaced_fields(original, SCHEMA)
check("original dict not mutated", "city" not in original)


# ===========================================================================
# 3. explicit_removals logic (inlined — mirrors refine.py route logic)
# ===========================================================================

print("\n── 3. explicit_removals ──────────────────────────────────────────────")

_B_CLASS = ("city", "sdg_tags", "categories", "business_type", "job_sector", "company_size", "claimed")

def detect_and_strip_removals(changes: dict) -> tuple[dict, set]:
    """Mirror of the explicit_removals block in refine_search()."""
    changes = dict(changes)
    removals = set()
    for field in _B_CLASS:
        if field in changes:
            val = changes[field]
            if val is None or val == [] or val == "":
                removals.add(field)
    for field in removals:
        changes.pop(field, None)
    return changes, removals


# 3-1 sdg_tags: null → detected and stripped
out, rem = detect_and_strip_removals({"sdg_tags": None, "city": "London"})
check("sdg_tags: null → removed from changes", "sdg_tags" not in out and "sdg_tags" in rem)
check("other fields preserved", out.get("city") == "London")

# 3-2 sdg_tags: [] → detected
out, rem = detect_and_strip_removals({"sdg_tags": []})
check("sdg_tags: [] → removal detected", "sdg_tags" in rem)

# 3-3 city: null → detected
out, rem = detect_and_strip_removals({"city": None})
check("city: null → removal detected", "city" in rem and "city" not in out)

# 3-4 Multiple removals at once
out, rem = detect_and_strip_removals({"city": None, "sdg_tags": None, "business_type": "B2B"})
check("multi-field removal", rem == {"city", "sdg_tags"} and out.get("business_type") == "B2B",
      f"rem={rem} out={out}")

# 3-5 claimed: None (boolean field)
out, rem = detect_and_strip_removals({"claimed": None})
check("claimed: null → removal detected", "claimed" in rem)

# 3-6 No removals → changes unchanged
out, rem = detect_and_strip_removals({"city": "London", "business_type": "B2B"})
check("no removals → changes unchanged", rem == set() and out == {"city": "London", "business_type": "B2B"})

# 3-7 Empty changes → no crash
out, rem = detect_and_strip_removals({})
check("empty changes → no crash", out == {} and rem == set())


# ===========================================================================
# 4. new_search_params merging logic
# ===========================================================================

print("\n── 4. new_search_params merging ──────────────────────────────────────")

def build_new_params(changes, explicit_removals, hard_filters, soft_filters, current_params, session_id="test"):
    """Mirror of the merging logic in refine_search()."""
    new_params = {
        "user_company_desc":  changes.get("user_company_desc",  current_params.get("user_company_desc", "")),
        "partner_type_desc":  changes.get("partner_type_desc",  current_params.get("partner_type_desc", "")),
        "other_requirements": changes.get("other_requirements", current_params.get("other_requirements", "")),
        "parent_id": session_id,
    }
    b_class_fields = ("city", "sdg_tags", "categories", "business_type", "job_sector", "company_size", "claimed")
    for field in b_class_fields:
        if field in explicit_removals:
            pass  # explicitly removed
        elif field in changes:
            new_params[field] = {"value": changes[field], "mode": "hard"}
        elif field in hard_filters:
            new_params[field] = {"value": hard_filters[field], "mode": "hard"}
        elif field in soft_filters:
            new_params[field] = {"value": soft_filters[field], "mode": "soft"}
    return new_params


# 4-1 New city overrides old city
params = build_new_params(
    changes={"city": "Manchester"},
    explicit_removals=set(),
    hard_filters={"city": "London"},
    soft_filters={},
    current_params={"user_company_desc": "test co", "partner_type_desc": "", "other_requirements": ""},
)
check("new city overrides old hard city", params.get("city", {}).get("value") == "Manchester")

# 4-2 Old hard filter inherited when not changed
params = build_new_params(
    changes={},
    explicit_removals=set(),
    hard_filters={"city": "London"},
    soft_filters={},
    current_params={"user_company_desc": "test co", "partner_type_desc": "", "other_requirements": ""},
)
check("old hard filter inherited", params.get("city", {}).get("value") == "London")

# 4-3 Explicit removal trumps old filter
params = build_new_params(
    changes={},
    explicit_removals={"sdg_tags"},
    hard_filters={"sdg_tags": ["Climate Action"]},
    soft_filters={},
    current_params={"user_company_desc": "test co", "partner_type_desc": "", "other_requirements": ""},
)
check("explicit removal trumps old hard filter", "sdg_tags" not in params)

# 4-4 Soft filter inherited when not changed
params = build_new_params(
    changes={},
    explicit_removals=set(),
    hard_filters={},
    soft_filters={"sdg_tags": ["Climate Action"]},
    current_params={"user_company_desc": "test co", "partner_type_desc": "", "other_requirements": ""},
)
check("soft filter inherited", params.get("sdg_tags", {}).get("mode") == "soft")

# 4-5 partner_type_desc inherited
params = build_new_params(
    changes={},
    explicit_removals=set(),
    hard_filters={},
    soft_filters={},
    current_params={"user_company_desc": "test co", "partner_type_desc": "skin-care brands", "other_requirements": ""},
)
check("partner_type_desc inherited", params.get("partner_type_desc") == "skin-care brands")

# 4-6 partner_type_desc cleared to ""
params = build_new_params(
    changes={"partner_type_desc": ""},
    explicit_removals=set(),
    hard_filters={},
    soft_filters={},
    current_params={"user_company_desc": "test co", "partner_type_desc": "skin-care brands", "other_requirements": ""},
)
check("partner_type_desc cleared to empty string", params.get("partner_type_desc") == "")

# 4-7 Remove SDG + add city simultaneously
params = build_new_params(
    changes={"city": "London"},
    explicit_removals={"sdg_tags"},
    hard_filters={"sdg_tags": ["Climate Action"]},
    soft_filters={},
    current_params={"user_company_desc": "test", "partner_type_desc": "", "other_requirements": ""},
)
check("remove SDG + add city simultaneously",
      "sdg_tags" not in params and params.get("city", {}).get("value") == "London")

# 4-8 parent_id always set
params = build_new_params({}, set(), {}, {}, {"user_company_desc": "", "partner_type_desc": "", "other_requirements": ""}, session_id="abc123")
check("parent_id set to session_id", params.get("parent_id") == "abc123")


# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 60)
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} failed)")
    print("\nFailed tests:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {FAIL} {name}: {detail}")
else:
    print("  — all green ✓")
print()
