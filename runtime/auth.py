"""Shared authentication utilities for runtime manager and worker APIs."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends, Header, HTTPException

logger = logging.getLogger(__name__)


def get_runtime_auth_token() -> str:
    """Resolve the single shared Ragtime <-> runtime bearer token.

    All manager and worker routes validate the same ``RUNTIME_AUTH_TOKEN``.
    ``RUNTIME_MANAGER_AUTH_TOKEN`` is a DEPRECATED legacy bridge: older compose
    files set it (on both containers) instead of the shared token, so we fall
    back to it to keep those deployments working after upgrade. The Ragtime app
    applies the same bridge (``ragtime/config/settings.py``) and surfaces a
    migration warning in the admin UI. Remove the bridge once legacy
    deployments have migrated.
    """
    return os.getenv("RUNTIME_AUTH_TOKEN", "").strip() or os.getenv("RUNTIME_MANAGER_AUTH_TOKEN", "").strip()


def _create_runtime_auth_dependency() -> Any:
    """Create a FastAPI dependency that validates the shared Bearer token.

    The token is read once at dependency creation time and cached for the
    process lifetime. If no token is configured, all requests are **rejected**
    to avoid accidentally running without auth.
    """
    cached_token = get_runtime_auth_token()

    if not cached_token:
        logger.warning(
            "RUNTIME_AUTH_TOKEN is empty or unset – all requests to guarded "
            "runtime routes will be rejected. Set it to a secure random value."
        )

    async def _verify_runtime_auth(
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> None:
        if not cached_token:
            raise HTTPException(
                status_code=503,
                detail="Runtime auth is not configured (RUNTIME_AUTH_TOKEN is unset)",
            )
        value = (authorization or "").strip()
        if not value.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing runtime auth token")
        if value[7:] != cached_token:
            raise HTTPException(status_code=403, detail="Invalid runtime auth token")

    return _verify_runtime_auth


# Manager and worker routes share one trust boundary (Ragtime <-> runtime
# service), so both dependency names validate the same shared token. The
# distinct names are kept for route readability.
require_manager_auth = _create_runtime_auth_dependency()
require_worker_auth = require_manager_auth

# Convenience aliases usable as ``Depends(require_manager_auth)`` in route signatures.
ManagerAuth = Depends(require_manager_auth)
WorkerAuth = Depends(require_worker_auth)


def _create_optional_runtime_auth_dependency() -> Any:
    """Create a non-raising bearer-token check used for soft-gating responses.

    Returns ``True`` only when the configured token is present **and** the
    caller supplied a matching ``Authorization: Bearer`` value. Used for
    ``/health`` endpoints where unauthenticated container healthchecks must
    keep working but verbose pool/session details should be withheld.
    """
    cached_token = get_runtime_auth_token()

    async def _check_optional_runtime_auth(
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> bool:
        if not cached_token:
            return False
        value = (authorization or "").strip()
        if not value.startswith("Bearer "):
            return False
        return value[7:] == cached_token

    return _check_optional_runtime_auth


check_optional_manager_auth = _create_optional_runtime_auth_dependency()
check_optional_worker_auth = check_optional_manager_auth

OptionalManagerAuth = Depends(check_optional_manager_auth)
OptionalWorkerAuth = Depends(check_optional_worker_auth)
