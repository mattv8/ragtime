"""Global authenticated server-backup transfer routes."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .transfer import RuntimeHistoryTransfers


class ExportRequest(BaseModel):
    include_repository_key: bool = False


def create_transfer_router(
    coordinator_getter: Callable[[], Any],
    dependency: Any,
    *,
    prefix: str = "",
) -> APIRouter:
    """Mount separately from workspace routes; all methods are fixed contracts.

    Args:
        coordinator_getter: Callable that returns the singleton coordinator
        dependency: FastAPI dependency for authentication
        prefix: Route prefix (e.g., "" for top-level, "/worker" for worker-scoped)
    """
    router = APIRouter(prefix=prefix, tags=["Runtime SQLite History Transfers"])

    # Resolve the coordinator lazily per request and reuse its cached
    # transfers instance so the background task registry and per-transfer
    # liveness locks survive individual requests and drain at shutdown.
    def get_transfers() -> RuntimeHistoryTransfers:
        coordinator = coordinator_getter()
        cached = getattr(coordinator, "get_history_transfers", None)
        if cached is not None:
            return cached()
        return RuntimeHistoryTransfers(coordinator)

    @router.get("/sqlite-history/exports/status")
    async def status(_auth: None = dependency) -> dict[str, Any]:
        return await get_transfers().status()

    @router.post("/sqlite-history/exports", status_code=202)
    async def export(payload: ExportRequest, _auth: None = dependency) -> dict[str, Any]:
        return await get_transfers().accept_export(include_repository_key=payload.include_repository_key)

    @router.get("/sqlite-history/exports/{export_id}")
    async def export_receipt(export_id: str, _auth: None = dependency) -> dict[str, Any]:
        return get_transfers().export_receipt(export_id)

    @router.get("/sqlite-history/exports/{export_id}/download")
    async def download(export_id: str, _auth: None = dependency) -> StreamingResponse:
        path = get_transfers().export_bundle(export_id)

        async def stream() -> AsyncIterator[bytes]:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(stream(), media_type="application/octet-stream")

    @router.post("/sqlite-history/imports", status_code=202)
    async def import_bundle(request: Request, _auth: None = dependency) -> dict[str, Any]:
        try:
            metadata = json.loads(request.headers["X-Ragtime-History-Metadata"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Runtime history import metadata is invalid") from exc
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="Runtime history import metadata is invalid")

        transfers = get_transfers()
        # Ensure tempfile root exists and is safe before creating temp file
        tmpdir = transfers._root() / "temp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        if tmpdir.is_symlink() or not tmpdir.is_dir():
            raise HTTPException(status_code=500, detail="Import tempfile directory is unsafe")

        fd, name = tempfile.mkstemp(prefix="sqlite-history-import-", dir=str(tmpdir))
        size = 0
        try:
            with os.fdopen(fd, "wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 64 * 1024 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="Runtime history import is too large")
                    output.write(chunk)
            return await transfers.accept_import(Path(name), metadata)
        except Exception:
            Path(name).unlink(missing_ok=True)
            raise

    @router.get("/sqlite-history/imports/{import_id}")
    async def import_receipt(import_id: str, _auth: None = dependency) -> dict[str, Any]:
        return get_transfers().import_receipt(import_id)

    return router
