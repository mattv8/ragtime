"""Authenticated internal HTTP contract for runtime SQLite history."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .coordinator import SqliteHistoryCoordinator
from .inspector_import import RuntimeInspectorImport
from .migration import LegacyHistoryMigration


class CaptureRequest(BaseModel):
    operation_id: str
    creator_id: str
    trigger: str = "manual"
    database_names: list[str] | None = Field(default=None, description="Explicit names or null for runtime enumeration")
    snapshot_id: str | None = None
    snapshot_git_commit_hash: str | None = None
    mandatory: bool = False
    request_digest: str | None = None


class LegacyMigrationRequest(BaseModel):
    operation_id: str
    user_id: str = Field(min_length=1)


class PreviewRequest(BaseModel):
    user_id: str = Field(min_length=1)
    mode: str
    conflict_policy: str
    table_policies: dict[str, Any] = Field(default_factory=dict)


class ApplyRequest(BaseModel):
    user_id: str = Field(min_length=1)


class RecoverRequest(BaseModel):
    action: str


class GuardedRestoreBeginRequest(BaseModel):
    operation_id: str
    user_id: str = Field(min_length=1)


class GuardedRestoreFinishRequest(BaseModel):
    user_id: str = Field(min_length=1)
    git_error: str | None = None


class DueClaimRequest(BaseModel):
    workspace_ids: list[str] = Field(max_length=100)


class DueAckRequest(BaseModel):
    occurrence_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)


def _receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "accepted_payload"}


def history_router(prefix: str, auth: Any, coordinator: Callable[[], SqliteHistoryCoordinator]) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["Runtime SQLite History"])

    @router.get("/sqlite-history/activation")
    async def activation(_auth: None = auth) -> dict[str, Any]:
        service = coordinator()
        return {"version": 2, "active": service._active(), "capability": service.capability()}

    @router.post("/sqlite-history/activation", status_code=201)
    async def activate(_auth: None = auth) -> dict[str, Any]:
        return coordinator().activate()

    @router.post("/sqlite-history/due/claim")
    async def claim_due(payload: DueClaimRequest, _auth: None = auth) -> dict[str, Any]:
        return {"claims": await coordinator().claim_due(payload.workspace_ids)}

    @router.post("/workspaces/{workspace_id}/sqlite-history/due/ack")
    async def ack_due(workspace_id: str, payload: DueAckRequest, _auth: None = auth) -> dict[str, Any]:
        return {"acknowledged": await coordinator().ack_due(workspace_id, payload.occurrence_id, payload.job_id)}

    @router.get("/workspaces/{workspace_id}/sqlite-history")
    async def list_history(workspace_id: str, database_name: str | None = None, snapshot_id: str | None = None, _auth: None = auth) -> dict[str, Any]:
        service = coordinator()
        return {
            "backups": await service.list_backups(workspace_id, database_name=database_name, snapshot_id=snapshot_id),
            "interrupted_maintenance": await service.interrupted_maintenance(workspace_id),
        }

    @router.post("/workspaces/{workspace_id}/sqlite-history/captures", status_code=202)
    async def accept_capture(workspace_id: str, payload: CaptureRequest, _auth: None = auth) -> dict[str, Any]:
        return _receipt(await coordinator().accept_capture(workspace_id, payload.model_dump()))

    @router.post("/workspaces/{workspace_id}/sqlite-history/import-database/{database_name}")
    async def import_database(workspace_id: str, database_name: str, request: Request, _auth: None = auth) -> dict[str, Any]:
        creator_id = request.headers.get("X-Ragtime-Creator-User", "").strip()
        if not creator_id:
            raise HTTPException(status_code=400, detail="SQLite import creator is required")
        history = coordinator()._service()
        staging = history._runtime.root / "_sqlite_history" / "inspector-imports"
        staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        if staging.is_symlink() or not staging.is_dir():
            raise HTTPException(status_code=503, detail="SQLite import staging is unavailable")
        fd, raw_name = tempfile.mkstemp(prefix="sqlite-import-", suffix=".sqlite3", dir=staging)
        size = 0
        try:
            with os.fdopen(fd, "wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 1024 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="Uploaded SQLite database is too large")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            return await RuntimeInspectorImport(history).import_database(workspace_id, database_name, creator_id, Path(raw_name))
        finally:
            Path(raw_name).unlink(missing_ok=True)

    @router.get("/workspaces/{workspace_id}/sqlite-history/captures/{operation_id}")
    async def get_capture(workspace_id: str, operation_id: str, _auth: None = auth) -> dict[str, Any]:
        return _receipt(await coordinator().get_operation(workspace_id, operation_id))

    @router.post("/workspaces/{workspace_id}/sqlite-history/migrations", status_code=202)
    async def migrate_legacy_history(workspace_id: str, payload: LegacyMigrationRequest, _auth: None = auth) -> dict[str, Any]:
        return _receipt(await LegacyHistoryMigration(coordinator()).accept(workspace_id, payload.model_dump()))

    @router.get("/workspaces/{workspace_id}/sqlite-history/migrations/{operation_id}")
    async def get_legacy_history_migration(workspace_id: str, operation_id: str, _auth: None = auth) -> dict[str, Any]:
        return _receipt(await LegacyHistoryMigration(coordinator()).get(workspace_id, operation_id))

    @router.post("/workspaces/{workspace_id}/sqlite-history/captures/{operation_id}/cancel")
    async def cancel_capture(workspace_id: str, operation_id: str, _auth: None = auth) -> dict[str, Any]:
        return _receipt(await coordinator().cancel(workspace_id, operation_id))

    @router.post("/workspaces/{workspace_id}/sqlite-history/captures/{operation_id}/ack")
    async def acknowledge_capture(workspace_id: str, operation_id: str, _auth: None = auth) -> dict[str, Any]:
        return _receipt(await coordinator().acknowledge(workspace_id, operation_id))

    @router.post("/workspaces/{workspace_id}/sqlite-history/backups/{backup_id}/preview")
    async def preview(workspace_id: str, backup_id: str, payload: PreviewRequest, _auth: None = auth) -> dict[str, Any]:
        return await coordinator().preview(workspace_id, backup_id, **payload.model_dump())

    @router.post("/workspaces/{workspace_id}/sqlite-history/previews/{preview_id}/apply")
    async def apply(workspace_id: str, preview_id: str, payload: ApplyRequest, _auth: None = auth) -> dict[str, Any]:
        return await coordinator().apply(workspace_id, preview_id, payload.user_id)

    @router.post("/workspaces/{workspace_id}/sqlite-history/recover/{operation_id}")
    async def recover(workspace_id: str, operation_id: str, payload: RecoverRequest, _auth: None = auth) -> dict[str, Any]:
        if payload.action not in {"complete", "abort"}:
            raise HTTPException(status_code=400, detail="Invalid recovery action")
        return await coordinator().recover(workspace_id, operation_id, payload.action)

    @router.delete("/workspaces/{workspace_id}/sqlite-history/backups/{backup_id}", status_code=204)
    async def delete(workspace_id: str, backup_id: str, _auth: None = auth) -> None:
        await coordinator().delete(workspace_id, backup_id)

    @router.get("/workspaces/{workspace_id}/sqlite-history/backups/{backup_id}/download")
    async def download(workspace_id: str, backup_id: str, _auth: None = auth) -> StreamingResponse:
        path = await coordinator().download_path(workspace_id, backup_id)

        async def stream() -> AsyncIterator[bytes]:
            try:
                with path.open("rb") as source:
                    while chunk := source.read(64 * 1024):
                        yield chunk
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(path)

        async def cleanup() -> None:
            with contextlib.suppress(OSError):
                os.unlink(path)

        return StreamingResponse(
            stream(),
            media_type="application/x-sqlite3",
            headers={"Content-Disposition": f'attachment; filename="{backup_id}.sqlite3"'},
            background=BackgroundTask(cleanup),
        )

    @router.post("/workspaces/{workspace_id}/sqlite-history/guarded-code-restores/begin")
    async def begin_guarded_restore(workspace_id: str, payload: GuardedRestoreBeginRequest, _auth: None = auth) -> dict[str, Any]:
        return await coordinator().begin_guarded_code_restore(workspace_id, payload.operation_id, payload.user_id)

    @router.post("/workspaces/{workspace_id}/sqlite-history/guarded-code-restores/{operation_id}/finish")
    async def finish_guarded_restore(workspace_id: str, operation_id: str, payload: GuardedRestoreFinishRequest, _auth: None = auth) -> dict[str, Any]:
        return await coordinator().finish_guarded_code_restore(workspace_id, operation_id, payload.user_id, payload.git_error)

    return router
