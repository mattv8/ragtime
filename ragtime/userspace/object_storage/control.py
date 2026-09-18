"""Authenticated, bounded client for the object-storage control service."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

import httpx
from fastapi import HTTPException

from ragtime.config.settings import settings

_BASE_URL = os.environ.get("OBJECT_STORAGE_CONTROL_URL", "http://runtime-s3:9001").rstrip("/")
_client: httpx.AsyncClient | None = None


def _control_token() -> str:
    key = (settings.encryption_key or "").strip().encode("utf-8")
    if not key:
        raise HTTPException(status_code=503, detail="Object storage control service is unavailable")
    return hmac.new(key, b"ragtime-object-storage-control-v1", hashlib.sha256).hexdigest()


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=10.0, write=10.0, pool=2.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=False,
        )
    return _client


async def request(method: str, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the gateway without exposing gateway or provider error details."""
    if not path.startswith("/"):
        raise ValueError("control path must be absolute")
    try:
        response = await _get_client().request(
            method,
            f"{_BASE_URL}{path}",
            json=json,
            headers={"Authorization": f"Bearer {_control_token()}"},
        )
    except (httpx.HTTPError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Object storage control service is unavailable") from exc
    if response.status_code >= 400:
        # Gateway errors are intentionally not relayed: they can contain endpoint
        # details from an external provider.
        detail = "Object storage request was rejected"
        if response.status_code == 404:
            detail = "Object storage resource was not found"
        elif response.status_code in (401, 403):
            detail = "Object storage control request was unauthorized"
        elif response.status_code >= 500:
            detail = "Object storage control service is unavailable"
        raise HTTPException(status_code=503 if response.status_code >= 500 else response.status_code, detail=detail)
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Object storage control service returned an invalid response") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=503, detail="Object storage control service returned an invalid response")
    return payload


async def ensure_workspace(workspace_id: str, legacy: dict[str, Any] | None = None) -> dict[str, Any]:
    return await request("POST", f"/v1/workspaces/{workspace_id}/ensure", legacy)


async def get_workspace(workspace_id: str) -> dict[str, Any]:
    return await request("GET", f"/v1/workspaces/{workspace_id}")


async def delete_workspace(workspace_id: str) -> None:
    await request("DELETE", f"/v1/workspaces/{workspace_id}")


async def import_legacy(workspace_id: str) -> dict[str, Any]:
    return await request("POST", f"/v1/workspaces/{workspace_id}/import-legacy")


async def submit_legacy_import(workspace_id: str, generation: str, manifest_sha256: str) -> dict[str, Any]:
    return await request("POST", f"/v1/workspaces/{workspace_id}/legacy-import", {"generation": generation, "manifest_sha256": manifest_sha256})


async def get_legacy_import(workspace_id: str) -> dict[str, Any]:
    return await request("GET", f"/v1/workspaces/{workspace_id}/legacy-import")


async def acknowledge_legacy_gc(workspace_id: str, generation: str, manifest_sha256: str) -> dict[str, Any]:
    return await request("POST", f"/v1/workspaces/{workspace_id}/legacy-import/gc", {"generation": generation, "manifest_sha256": manifest_sha256})
