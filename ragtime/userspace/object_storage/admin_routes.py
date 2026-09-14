"""Admin-only proxy routes for object-storage configuration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from ragtime.core.security import require_admin
from ragtime.userspace.object_storage import control

router = APIRouter(tags=["User Space"])


def _public_settings(value: dict[str, Any]) -> dict[str, Any]:
    """Defence in depth: browser routes never return provider/workspace secrets."""
    allowed = {
        "mode",
        "default_backend_id",
        "endpoint",
        "region",
        "bucket",
        "access_key_configured",
        "secret_key_configured",
        "existing_local_workspaces",
        "migrations",
    }
    return {key: value[key] for key in allowed if key in value}


@router.get("/admin/object-storage")
async def get_settings(_user: Any = Depends(require_admin)) -> dict[str, Any]:
    return _public_settings(await control.request("GET", "/v1/settings"))


@router.put("/admin/object-storage")
async def put_settings(payload: dict[str, Any] = Body(...), _user: Any = Depends(require_admin)) -> dict[str, Any]:
    return _public_settings(await control.request("PUT", "/v1/settings", payload))


@router.post("/admin/object-storage/test")
async def test_settings(payload: dict[str, Any] = Body(...), _user: Any = Depends(require_admin)) -> dict[str, Any]:
    result = await control.request("POST", "/v1/settings/test", payload)
    return {"success": bool(result.get("success"))}


@router.get("/admin/object-storage/migrations")
async def list_migrations(_user: Any = Depends(require_admin)) -> dict[str, Any]:
    result = await control.request("GET", "/v1/migrations")
    return {"jobs": result.get("jobs", [])}


@router.post("/admin/object-storage/migrations")
async def start_migrations(payload: dict[str, Any] = Body(default={}), _user: Any = Depends(require_admin)) -> dict[str, Any]:
    result = await control.request("POST", "/v1/migrations", payload)
    return {"jobs": result.get("jobs", [])}


@router.post("/admin/object-storage/migrations/{job_id}/retry")
async def retry_migration(job_id: str, _user: Any = Depends(require_admin)) -> dict[str, Any]:
    return await control.request("POST", f"/v1/migrations/{job_id}/retry")
