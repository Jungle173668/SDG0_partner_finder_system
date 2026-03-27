"""
Search routes:
  POST /api/search        — start a new pipeline run, return session_id immediately
  GET  /api/search/{id}   — poll run status + results
  GET  /api/report/{id}   — return the raw HTML report (for iframe embed)
"""

import logging
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.session_store import load_session, save_session, update_session_status

logger = logging.getLogger(__name__)
router = APIRouter()

# Single background thread pool — pipeline is CPU+IO bound, one thread is fine
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class FilterEntry(BaseModel):
    """A single filter field with Hard/Soft mode."""
    value: Union[str, List[str], bool]
    mode: str = "hard"  # "hard" | "soft"


class SearchRequest(BaseModel):
    user_company_desc: str = Field(..., min_length=3, description="Your company description")
    partner_type_desc: str = Field("", description="What kind of partner are you looking for")
    other_requirements: str = Field("", description="Additional free-text requirements")
    parent_id: Optional[str] = Field(None, description="Parent session ID (for HITL refinement lineage)")

    # Filter fields — each has a value + hard/soft mode
    city: Optional[FilterEntry] = None
    business_type: Optional[FilterEntry] = None
    job_sector: Optional[FilterEntry] = None
    company_size: Optional[FilterEntry] = None
    claimed: Optional[FilterEntry] = None
    sdg_tags: Optional[FilterEntry] = None
    categories: Optional[FilterEntry] = None


class SearchResponse(BaseModel):
    session_id: str
    status: str  # "running" | "done" | "error"
    message: str = ""


class StatusResponse(BaseModel):
    session_id: str
    status: str  # "running" | "done" | "error"
    # Only populated when status == "done"
    scored_companies: List[dict] = []
    search_fallback_level: int = 0
    filters: dict = {}
    soft_filters: dict = {}
    partner_type_desc: str = ""
    user_company_desc: str = ""
    errors: List[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_hard_soft(req: SearchRequest) -> tuple[dict, dict]:
    """
    Split request filter fields into hard_filters and soft_filters dicts.

    Hard filters → passed to SearchAgent WHERE clause (exact match required).
    Soft filters → passed to ScoringAgent as bonus scoring criteria.
    """
    hard: dict = {}
    soft: dict = {}

    filter_fields = ["city", "business_type", "job_sector", "company_size", "claimed", "sdg_tags", "categories"]
    for field_name in filter_fields:
        entry: Optional[FilterEntry] = getattr(req, field_name, None)
        if entry is None:
            continue
        # Skip empty values
        val = entry.value
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        if isinstance(val, list) and not val:
            continue

        if entry.mode == "soft":
            soft[field_name] = val
        else:
            hard[field_name] = val

    return hard, soft


def _run_pipeline_background(session_id: str, req: SearchRequest) -> None:
    """
    Run the full pipeline in a background thread.
    Writes progress to session store: running → done / error.
    """
    try:
        from agent.graph import run_pipeline

        hard_filters, soft_filters = _split_hard_soft(req)

        logger.info(
            f"Pipeline starting — session={session_id} "
            f"hard={list(hard_filters.keys())} soft={list(soft_filters.keys())}"
        )

        state = run_pipeline(
            user_company_desc=req.user_company_desc,
            partner_type_desc=req.partner_type_desc,
            filters=hard_filters,
            soft_filters=soft_filters,
            other_requirements=req.other_requirements,
            session_id=session_id,
        )

        # Persist full state as JSON
        save_session(session_id, {
            "session_id": session_id,
            "status": "done",
            "user_company_desc": state.get("user_company_desc", ""),
            "partner_type_desc": state.get("partner_type_desc", ""),
            "filters": hard_filters,
            "soft_filters": soft_filters,
            "other_requirements": req.other_requirements,
            "parent_id": req.parent_id,
            "scored_companies": state.get("scored_companies", []),
            "candidate_companies": state.get("candidate_companies", []),
            "search_fallback_level": state.get("search_fallback_level", 0),
            "search_method": state.get("search_method", ""),
            "errors": state.get("errors", []),
            # report field is the HTML path/string — not serialized here
        })
        logger.info(f"Pipeline done — session={session_id}")

    except Exception as e:
        logger.error(f"Pipeline error — session={session_id}: {e}", exc_info=True)
        update_session_status(session_id, "error", error=str(e))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/search", response_model=SearchResponse)
def start_search(req: SearchRequest):
    """
    Start a new pipeline run.

    Returns a session_id immediately — frontend polls GET /api/search/{id}
    until status == "done".
    """
    session_id = secrets.token_urlsafe(4)[:6]

    # Mark as running immediately so frontend can start polling
    update_session_status(session_id, "running")

    # Submit pipeline to background thread
    _executor.submit(_run_pipeline_background, session_id, req)

    logger.info(f"Search submitted — session={session_id}")
    return SearchResponse(
        session_id=session_id,
        status="running",
        message="Pipeline started. Poll GET /api/search/{session_id} for results.",
    )


@router.get("/search/{session_id}", response_model=StatusResponse)
def get_search_status(session_id: str):
    """
    Poll pipeline status.

    Returns status 'running' while in progress, 'done' with results when complete.
    """
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired")

    status = session.get("status", "running")

    if status != "done":
        return StatusResponse(session_id=session_id, status=status)

    return StatusResponse(
        session_id=session_id,
        status="done",
        scored_companies=session.get("scored_companies", []),
        search_fallback_level=session.get("search_fallback_level", 0),
        filters=session.get("filters", {}),
        soft_filters=session.get("soft_filters", {}),
        partner_type_desc=session.get("partner_type_desc", ""),
        user_company_desc=session.get("user_company_desc", ""),
        errors=session.get("errors", []),
    )


@router.get("/report/{session_id}")
def get_report(session_id: str):
    """
    Return the HTML report for a session.

    The report file is generated by ReportAgent at reports/{session_id}.html
    """
    from fastapi.responses import FileResponse, HTMLResponse

    report_path = Path("reports") / f"{session_id}.html"
    if not report_path.exists():
        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Report Expired</title>
<style>
  body {{ font-family: system-ui, sans-serif; display: flex; align-items: center;
         justify-content: center; min-height: 100vh; margin: 0;
         background: #f8f9fb; color: #0a1f3c; }}
  .box {{ text-align: center; max-width: 380px; padding: 40px 32px;
          background: white; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }}
  .icon {{ font-size: 48px; margin-bottom: 16px; }}
  h2 {{ margin: 0 0 10px; font-size: 20px; }}
  p {{ color: #6b7280; font-size: 14px; line-height: 1.6; margin: 0; }}
  .id {{ font-family: monospace; font-size: 12px; color: #9ca3af; margin-top: 16px; }}
</style></head>
<body>
  <div class="box">
    <div class="icon">⏳</div>
    <h2>Report No Longer Available</h2>
    <p>This report has expired and been deleted.<br>
       Reports are kept for 15 days. Please run a new search to generate a fresh report.</p>
    <div class="id">Session: {session_id}</div>
  </div>
</body></html>"""
        return HTMLResponse(content=html, status_code=404)

    return FileResponse(str(report_path), media_type="text/html")
