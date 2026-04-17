"""Shared authentication utilities for runtime manager and worker APIs."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends, Header, HTTPException

logger = logging.getLogger(__name__)


def _create_runtime_auth_dependency(
    env_var: str,
) -> Any:
    """Create a FastAPI dependency that validates a Bearer token from an env var.

    The token is read once at dependency creation time and cached for the process
    lifetime. If the env var is empty or unset, all requests are **rejected** to
    avoid accidentally running without auth.
    """
    cached_token = os.getenv(env_var, "").strip()

    if not cached_token:
        logger.warning(
            "Runtime auth env var %s is empty or unset – "
            "all requests to routes guarded by this dependency will be rejected. "
            "Set the variable to a secure random value.",
            env_var,
        )

    async def _verify_runtime_auth(
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> None:
        if not cached_token:
            raise HTTPException(
                status_code=503,
                detail=f"Runtime auth is not configured ({env_var} is unset)",
            )
        value = (authorization or "").strip()
        if not value.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing runtime auth token")
        if value[7:] != cached_token:
            raise HTTPException(status_code=403, detail="Invalid runtime auth token")

    return _verify_runtime_auth


require_manager_auth = _create_runtime_auth_dependency("RUNTIME_MANAGER_AUTH_TOKEN")
require_worker_auth = _create_runtime_auth_dependency("RUNTIME_WORKER_AUTH_TOKEN")

# Convenience aliases usable as ``Depends(require_manager_auth)`` in route signatures.
ManagerAuth = Depends(require_manager_auth)
WorkerAuth = Depends(require_worker_auth)


def _create_optional_runtime_auth_dependency(env_var: str) -> Any:
    """Create a non-raising bearer-token check used for soft-gating responses.

    Returns ``True`` only when the configured token is present **and** the
    caller supplied a matching ``Authorization: Bearer`` value. Used for
    ``/health`` endpoints where unauthenticated container healthchecks must
    keep working but verbose pool/session details should be withheld.
    """
    cached_token = os.getenv(env_var, "").strip()

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


check_optional_manager_auth = _create_optional_runtime_auth_dependency(
    "RUNTIME_MANAGER_AUTH_TOKEN"
)
check_optional_worker_auth = _create_optional_runtime_auth_dependency(
    "RUNTIME_WORKER_AUTH_TOKEN"
)

OptionalManagerAuth = Depends(check_optional_manager_auth)
OptionalWorkerAuth = Depends(check_optional_worker_auth)
