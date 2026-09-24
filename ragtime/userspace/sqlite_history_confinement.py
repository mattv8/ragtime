"""Cheap process-local availability check for confined SQLite operations."""

from __future__ import annotations

from threading import Lock
from time import monotonic

from fastapi import HTTPException

from ragtime.core.logging import get_logger
from runtime.worker import mount_sync_launcher as _landlock

_TTL_SECONDS = 60.0
_lock = Lock()
_cached_at: float | None = None
_cached_available: bool | None = None
_reported_unavailable = False
logger = get_logger(__name__)


def confinement_available() -> bool:
    """Probe Landlock ruleset creation without installing a sandbox."""
    global _cached_at, _cached_available, _reported_unavailable
    now = monotonic()
    with _lock:
        if _cached_at is not None and now - _cached_at < _TTL_SECONDS:
            return bool(_cached_available)
        try:
            _landlock.trial_ruleset()
        except OSError as exc:
            _cached_available = False
            _cached_at = now
            if not _reported_unavailable:
                logger.warning("Secure SQLite confinement unavailable; host Landlock ABI 3 or newer is required: %s", exc)
                _reported_unavailable = True
            return False
        was_unavailable = _reported_unavailable
        _cached_available = True
        _cached_at = now
        _reported_unavailable = False
        if was_unavailable:
            logger.info("Secure SQLite confinement preflight recovered")
        return True


def require_confinement() -> None:
    if not confinement_available():
        raise HTTPException(status_code=503, detail="Secure SQLite confinement is unavailable; host Landlock ABI 3 or newer is required")
