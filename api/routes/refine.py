"""
Refine route:
  POST /api/refine/{session_id}

Interprets user feedback (liked/disliked companies + free text) and returns
updated search parameters ready to pass to POST /api/search.

Does NOT start a new pipeline — the frontend calls POST /api/search with the
returned `new_search_params` after the user confirms.
"""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.session_store import load_session

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CompanyFeedback(BaseModel):
    name: str
    categories: Optional[str] = None
    sdg_tags: Optional[str] = None
    business_type: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


class RefineRequest(BaseModel):
    liked: List[CompanyFeedback] = []
    disliked: List[CompanyFeedback] = []
    user_text: str = ""
    allow_global_fallback: bool = False


class RefineResponse(BaseModel):
    action: str                  # "refine" | "unclear"
    summary: str = ""
    # Ready-to-use body for POST /api/search (only present when action == "refine")
    new_search_params: dict = {}
    # Fields that were requested but failed validation (invalid values)
    rejected: list = []


# ---------------------------------------------------------------------------
# Schema helper — reuse the cached schema from the schema route
# ---------------------------------------------------------------------------

def _get_schema_safe() -> dict:
    try:
        from agent.schema_cache import get_schema
        return get_schema()
    except Exception as e:
        logger.warning(f"refine: could not load schema — {e}")
        return {}


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

def _fix_misplaced_fields(changes: dict, schema: dict) -> dict:
    """
    Code-level safety net: detect common LLM mistakes where structured values
    were placed in free-text fields, and move them to the correct B-class field.

    Covers:
      - City name in partner_type_desc / other_requirements → city
      - SDG name in partner_type_desc → sdg_tags
      - claimed keywords in other_requirements → claimed
    """
    changes = dict(changes)  # shallow copy, don't mutate original

    valid_cities  = {v.lower(): v for v in schema.get("city", [])}
    valid_sdgs    = {v.lower(): v for v in schema.get("sdg_tags", [])}

    ptd = changes.get("partner_type_desc", "").lower()
    oreq = changes.get("other_requirements", "").lower()

    # ── City misplaced in partner_type_desc ──────────────────────────────────
    if "city" not in changes:
        for lc, original in valid_cities.items():
            if lc in ptd or lc in oreq:
                changes["city"] = original
                logger.info(f"refine fix: extracted city='{original}' from text fields")
                break

    # ── SDG misplaced in partner_type_desc ───────────────────────────────────
    if "sdg_tags" not in changes:
        found_sdgs = [original for lc, original in valid_sdgs.items() if lc in ptd]
        # Also match "sdg 13", "sdg13" patterns
        import re
        sdg_nums = re.findall(r"sdg\s*(\d{1,2})", ptd)
        num_to_name = {
            "1":"No Poverty","2":"Zero Hunger","3":"Good Health And Well-Being",
            "4":"Quality Education","5":"Gender Equality","6":"Clean Water And Sanitation",
            "7":"Affordable And Clean Energy","8":"Decent Work And Economic Growth",
            "9":"Industry Innovation And Infrastructure","10":"Reduced Inequalities",
            "11":"Sustainable Cities And Communities","12":"Responsible Consumption And Production",
            "13":"Climate Action","14":"Life Below Water","15":"Life On Land",
            "16":"Peace Justice And Strong Institutions","17":"Partnerships For The Goals",
        }
        for num in sdg_nums:
            name = num_to_name.get(num)
            if name:
                # find closest match in valid_sdgs
                for lc, original in valid_sdgs.items():
                    if name.lower() in lc or lc in name.lower():
                        if original not in found_sdgs:
                            found_sdgs.append(original)
                        break
        if found_sdgs:
            changes["sdg_tags"] = found_sdgs
            logger.info(f"refine fix: extracted sdg_tags={found_sdgs} from partner_type_desc")

    # ── Claimed misplaced in other_requirements ──────────────────────────────
    if "claimed" not in changes:
        claimed_kws = ["verified only", "certified only", "claimed only", "must be verified",
                       "verified profiles", "certified profiles"]
        if any(kw in oreq for kw in claimed_kws):
            changes["claimed"] = True
            logger.info("refine fix: extracted claimed=True from other_requirements")

    return changes


def _validate_b_class(changes: dict, schema: dict) -> tuple[dict, list[str]]:
    """
    Validate B-class fields in changes against the schema.

    For each B-class field:
      1. Try exact match (case-insensitive)
      2. Try substring match (e.g. "londон" → "London")
      3. If no match → remove from changes, add to rejected list

    Returns (validated_changes, rejected_messages)
    """
    validated = dict(changes)
    rejected  = []

    def _match_single(value: str, valid_list: list[str]) -> Optional[str]:
        """Return canonical value if found, else None."""
        if not valid_list:
            return value  # no schema available, pass through
        lc = value.strip().lower()
        # 1. exact
        for v in valid_list:
            if v.lower() == lc:
                return v
        # 2. substring
        for v in valid_list:
            if lc in v.lower() or v.lower() in lc:
                return v
        return None

    def _match_list(values, valid_list: list[str]) -> tuple[list[str], list[str]]:
        """Validate a list of values. Returns (valid_values, rejected_values)."""
        if isinstance(values, str):
            values = [values]
        good, bad = [], []
        for v in values:
            matched = _match_single(str(v), valid_list)
            if matched:
                good.append(matched)
            else:
                bad.append(v)
        return good, bad

    # ── city ─────────────────────────────────────────────────────────────────
    if "city" in validated:
        matched = _match_single(str(validated["city"]), schema.get("city", []))
        if matched:
            validated["city"] = matched
        else:
            rejected.append(f"city='{validated['city']}' is not in the database — ignored")
            del validated["city"]
            logger.warning(f"refine validate: invalid city '{changes['city']}' removed")

    # ── sdg_tags ─────────────────────────────────────────────────────────────
    if "sdg_tags" in validated:
        good, bad = _match_list(validated["sdg_tags"], schema.get("sdg_tags", []))
        if bad:
            rejected.append(f"sdg_tags {bad} not recognised — ignored")
            logger.warning(f"refine validate: invalid sdg_tags {bad} removed")
        if good:
            validated["sdg_tags"] = good
        else:
            del validated["sdg_tags"]

    # ── categories ───────────────────────────────────────────────────────────
    if "categories" in validated:
        good, bad = _match_list(validated["categories"], schema.get("categories", []))
        if bad:
            rejected.append(f"categories {bad} not recognised — ignored")
            logger.warning(f"refine validate: invalid categories {bad} removed")
        if good:
            validated["categories"] = good
        else:
            del validated["categories"]

    # ── business_type ─────────────────────────────────────────────────────────
    if "business_type" in validated:
        matched = _match_single(str(validated["business_type"]), schema.get("business_type", []))
        if matched:
            validated["business_type"] = matched
        else:
            rejected.append(f"business_type='{validated['business_type']}' not recognised — ignored")
            del validated["business_type"]

    # ── job_sector ────────────────────────────────────────────────────────────
    if "job_sector" in validated:
        matched = _match_single(str(validated["job_sector"]), schema.get("job_sector", []))
        if matched:
            validated["job_sector"] = matched
        else:
            rejected.append(f"job_sector='{validated['job_sector']}' not recognised — ignored")
            del validated["job_sector"]

    # ── company_size ──────────────────────────────────────────────────────────
    if "company_size" in validated:
        matched = _match_single(str(validated["company_size"]), schema.get("company_size", []))
        if matched:
            validated["company_size"] = matched
        else:
            rejected.append(f"company_size='{validated['company_size']}' not recognised — ignored")
            del validated["company_size"]

    return validated, rejected


@router.post("/refine/{session_id}", response_model=RefineResponse)
def refine_search(session_id: str, req: RefineRequest):
    """
    Interpret user feedback and return updated search parameters.

    Flow:
      1. Load current session → extract current params
      2. Call refine_agent LLM
      3. Merge changes onto current params
      4. Return new_search_params in SearchRequest format (for frontend to pass to /api/search)
    """
    from agent.refine_agent import run_refine_agent

    # ── 1. Load session ──────────────────────────────────────────────────────
    session = load_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired",
        )

    hard_filters: dict = session.get("filters", {})
    soft_filters: dict = session.get("soft_filters", {})

    current_params = {
        "user_company_desc":  session.get("user_company_desc", ""),
        "partner_type_desc":  session.get("partner_type_desc", ""),
        "other_requirements": session.get("other_requirements", ""),
    }
    # Add active filter values for display in prompt
    for k in ("city", "sdg_tags", "categories", "business_type", "job_sector", "company_size", "claimed"):
        if k in hard_filters:
            current_params[k] = hard_filters[k]
        elif k in soft_filters:
            current_params[k] = soft_filters[k]

    # ── 2. Load schema ───────────────────────────────────────────────────────
    schema_raw = _get_schema_safe()
    schema = {
        "city":          schema_raw.get("city", []),
        "sdg_tags":      schema_raw.get("sdg_tags", []),
        "categories":    schema_raw.get("categories", []),
        "business_type": schema_raw.get("business_type", []),
        "job_sector":    schema_raw.get("job_sector", []),
        "company_size":  schema_raw.get("company_size", []),
    }

    # ── 3. Run refine agent ──────────────────────────────────────────────────
    result = run_refine_agent(
        current_params=current_params,
        schema=schema,
        liked=[c.model_dump(exclude_none=True) for c in req.liked],
        disliked=[c.model_dump(exclude_none=True) for c in req.disliked],
        user_text=req.user_text,
    )

    if result.get("action") != "refine":
        return RefineResponse(
            action="unclear",
            summary=result.get("summary", "Could not understand your request — please rephrase"),
        )

    changes: dict = result.get("changes", {})
    llm_modes: dict = result.get("modes", {})  # LLM-inferred hard/soft per field

    # ── 3b. Detect explicit removals BEFORE validation strips empty lists ─────
    # LLM signals "remove this filter" by outputting [] or null for a B-class field.
    # _validate_b_class would delete those keys (treating empty as no-change),
    # so we capture them here first.
    _b_class = ("city", "sdg_tags", "categories", "business_type", "job_sector", "company_size", "claimed")
    explicit_removals: set = set()
    for field in _b_class:
        if field in changes:
            val = changes[field]
            if val is None or val == [] or val == "":
                explicit_removals.add(field)
    if explicit_removals:
        logger.info(f"refine: explicit removals detected: {explicit_removals}")
        for field in explicit_removals:
            changes.pop(field, None)  # must remove before _validate_b_class, or None crashes iteration

    # ── 3c. Post-process: fix misplaced fields, then validate values ─────────
    changes = _fix_misplaced_fields(changes, schema)
    changes, rejected = _validate_b_class(changes, schema)

    # ── 3d. Remove A-class echoes (LLM rule-1 violation safety net) ──────────
    # If LLM returns the same value as current for a free-text field, it's an
    # echo — not a real change. Drop it so it doesn't pollute the summary.
    # Note: partner_type_desc="" (intentional clear) won't match a non-empty
    # current value, so it's safe to keep.
    for a_field in ("partner_type_desc", "other_requirements", "user_company_desc"):
        if a_field in changes:
            if str(changes[a_field]).strip() == current_params.get(a_field, "").strip():
                del changes[a_field]
                logger.info(f"refine: removed echoed A-class field '{a_field}'")

    logger.info(f"refine: changes={list(changes.keys())} modes={llm_modes} rejected={rejected}")

    # ── 4. Build new_search_params in SearchRequest format ───────────────────
    # Start from original free-text params
    new_params: dict = {
        "user_company_desc":    changes.get("user_company_desc",  current_params.get("user_company_desc", "")),
        "partner_type_desc":    changes.get("partner_type_desc",  current_params.get("partner_type_desc", "")),
        "other_requirements":   changes.get("other_requirements", current_params.get("other_requirements", "")),
        "parent_id":            session_id,  # track lineage
        "allow_global_fallback": req.allow_global_fallback,
    }

    # Merge B-class filter fields — mode priority: LLM explicit > original session > default hard
    def _build_filter_entry(field: str, new_val) -> Optional[dict]:
        """Wrap a value in FilterEntry format."""
        if new_val is None:
            return None
        if field in llm_modes and llm_modes[field] in ("hard", "soft"):
            mode = llm_modes[field]          # LLM explicitly said hard/soft
        elif field in soft_filters:
            mode = "soft"                    # was soft in original session
        else:
            mode = "hard"                    # default
        return {"value": new_val, "mode": mode}

    b_class_fields = ("city", "sdg_tags", "categories", "business_type", "job_sector", "company_size", "claimed")
    for field in b_class_fields:
        if field in explicit_removals:
            pass  # user explicitly removed this filter — don't carry it forward
        elif field in changes:
            entry = _build_filter_entry(field, changes[field])
            if entry is not None:
                new_params[field] = entry
        elif field in hard_filters:
            new_params[field] = {"value": hard_filters[field], "mode": "hard"}
        elif field in soft_filters:
            new_params[field] = {"value": soft_filters[field], "mode": "soft"}

    return RefineResponse(
        action="refine",
        summary=result.get("summary", ""),
        new_search_params=new_params,
        rejected=rejected,
    )
