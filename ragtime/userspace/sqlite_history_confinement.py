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
_cached_unavailable_reason: str | None = None
_reported_unavailable = False
logger = get_logger(__name__)


def confinement_available() -> bool:
    """Probe Landlock ruleset creation without installing a sandbox."""
    return unavailable_reason() is None


def unavailable_reason() -> str | None:
    """Return the cached public confinement diagnostic, if unavailable."""
    global _cached_at, _cached_available, _cached_unavailable_reason, _reported_unavailable
    now = monotonic()
    with _lock:
        if _cached_at is not None and now - _cached_at < _TTL_SECONDS:
            return _cached_unavailable_reason
        reason = _landlock.confinement_unavailable_reason()
        if reason is not None:
            _cached_available = False
            _cached_unavailable_reason = reason
            _cached_at = now
            if not _reported_unavailable:
                logger.warning("Secure SQLite confinement unavailable: %s", reason)
                _reported_unavailable = True
            return reason
        was_unavailable = _reported_unavailable
        _cached_available = True
        _cached_unavailable_reason = None
        _cached_at = now
        _reported_unavailable = False
        if was_unavailable:
            logger.info("Secure SQLite confinement preflight recovered")
        return None


def require_confinement() -> None:
    if reason := unavailable_reason():
        raise HTTPException(status_code=503, detail=reason)
