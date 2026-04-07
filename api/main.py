"""
FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /api/search              — start pipeline, return session_id
    GET  /api/search/{id}         — poll status + results
    GET  /api/report/{id}         — serve HTML report (for iframe)
    GET  /api/schema              — filter dropdown values
    GET  /api/health              — health check
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SDGZero Partner Finder API",
    description="Multi-Agent Pipeline for sustainability-focused business partner matching.",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow Next.js dev server (localhost:3000) and production domain
# ---------------------------------------------------------------------------

_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
if os.getenv("FRONTEND_URL"):
    _FRONTEND_ORIGINS.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files — serve generated reports and static assets
# ---------------------------------------------------------------------------

# Serve reports directory so iframe src="/reports/{id}.html" works
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Serve static assets (SDGZero logo, SDG icons)
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

from api.routes.search import router as search_router
from api.routes.schema import router as schema_router
from api.routes.refine import router as refine_router

app.include_router(search_router, prefix="/api")
app.include_router(schema_router, prefix="/api")
app.include_router(refine_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


# Clean up expired sessions on startup
from api.session_store import cleanup_expired
cleanup_expired()

# Warm up embedding model and DB connection pool so first request is fast
try:
    from agent.tools import _get_encoder, _get_store
    _get_encoder()
    _get_store()
    logger.info("Warm-up complete — encoder and DB pool ready")
except Exception as e:
    logger.warning(f"Warm-up failed (non-fatal): {e}")

logger.info("SDGZero Partner Finder API ready — docs at /api/docs")
