"""
Session store — persists pipeline state as JSON files in sessions/.

Design rationale:
  Simple JSON files = zero infrastructure, works offline, git-ignorable.
  Each session is a single file: sessions/{session_id}.json
  Supports URL sharing: /results/{id} reads state from this file.
  30-day TTL enforced at read time (stale sessions return None).
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Sessions directory — relative to repo root, excluded in .gitignore
_SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "./sessions"))
_REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "./reports"))
_TTL_DAYS = int(os.getenv("SESSION_TTL_DAYS", "15"))


def _sessions_dir() -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


def save_session(session_id: str, data: dict) -> None:
    """Write session state to sessions/{session_id}.json."""
    path = _sessions_dir() / f"{session_id}.json"
    payload = {
        **data,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=_TTL_DAYS)).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)
    logger.debug(f"session_store: saved {session_id}")


def load_session(session_id: str) -> Optional[dict]:
    """
    Load session state.

    Returns None if:
      - File doesn't exist
      - Session has expired (> 30 days)
    """
    path = _sessions_dir() / f"{session_id}.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expires_at_str = data.get("expires_at")
    if expires_at_str:
        expires_at = datetime.fromisoformat(expires_at_str)
        if datetime.now(timezone.utc) > expires_at:
            path.unlink(missing_ok=True)
            (_REPORTS_DIR / f"{session_id}.html").unlink(missing_ok=True)
            logger.info(f"session_store: expired session {session_id} deleted")
            return None

    return data


def cleanup_expired() -> int:
    """
    Scan sessions/ and delete all expired session JSON files + their HTML reports.
    Returns the number of sessions deleted.
    Called once on API startup to proactively free disk space.
    """
    sessions_dir = _sessions_dir()
    deleted = 0
    now = datetime.now(timezone.utc)

    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            expires_at_str = data.get("expires_at")
            if not expires_at_str:
                continue
            if now > datetime.fromisoformat(expires_at_str):
                session_id = path.stem
                path.unlink(missing_ok=True)
                (_REPORTS_DIR / f"{session_id}.html").unlink(missing_ok=True)
                deleted += 1
        except Exception as e:
            logger.warning(f"session_store: cleanup failed for {path.name} — {e}")

    if deleted:
        logger.info(f"session_store: cleanup removed {deleted} expired session(s)")
    return deleted


def session_exists(session_id: str) -> bool:
    """Check if a session file exists (without loading full content)."""
    return (_sessions_dir() / f"{session_id}.json").exists()


def update_session_status(session_id: str, status: str, error: Optional[str] = None) -> None:
    """
    Update the status field of an existing session.

    Used to mark pipeline as 'running' → 'done' or 'error' without
    rewriting the entire state (avoids race conditions with the worker thread).
    """
    path = _sessions_dir() / f"{session_id}.json"
    if not path.exists():
        # Create minimal placeholder so frontend can poll immediately
        save_session(session_id, {"session_id": session_id, "status": status})
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["status"] = status
    if error:
        data["error"] = error

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)
