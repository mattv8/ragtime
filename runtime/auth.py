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
