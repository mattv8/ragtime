"""Owner/admin-only API for protected SQLite history."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ragtime.core.logging import get_logger
from ragtime.core.runtime_manager_client import get_runtime_manager_request_config
from ragtime.core.security import get_current_user
from ragtime.userspace.service import userspace_service
from ragtime.userspace.sqlite_backup_queue import get_sqlite_backup_queue_service
from ragtime.userspace.sqlite_history import get_sqlite_history_service
from ragtime.userspace.sqlite_history_models import (
    SqliteHistoryCaptureJobListResponse,
    SqliteHistoryCaptureJobRequest,
    SqliteHistoryCaptureJobResponse,
    SqliteHistoryCaptureRequest,
    SqliteHistoryCaptureResponse,
    SqliteHistoryListResponse,
    SqliteHistoryPreview,
    SqliteHistoryPreviewRequest,
    SqliteHistoryRecoveryRequest,
    SqliteHistoryRestoreRequest,
    SqliteHistoryRestoreResponse,
)

router = APIRouter(prefix="/indexes/userspace", tags=["User Space SQLite History"])
logger = get_logger(__name__)


async def _manage(workspace_id: str, user: Any) -> None:
    await userspace_service._enforce_workspace_access(workspace_id, user.id, required_role="owner", is_admin=getattr(user, "role", "") == "admin")


def _public_capture_job(job: dict[str, Any]) -> dict[str, Any]:
    """Exclude queue ownership and idempotency internals from API responses."""
    fields = {
        "id",
        "workspace_id",
        "trigger",
        "database_names",
        "snapshot_id",
        "snapshot_git_commit_hash",
        "status",
        "created_at",
        "available_at",
        "started_at",
        "finished_at",
        "updated_at",
        "completed_databases",
        "total_databases",
        "backup_ids",
        "error_message",
        "cancel_requested",
    }
    return {field: job[field] for field in fields if field in job}


async def _sqlite_history_list_payload(
    workspace_id: str,
    *,
    database_name: str | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    service = get_sqlite_history_service()
    response = SqliteHistoryListResponse.model_validate(
        {
            "workspace_id": workspace_id,
            "backups": await service.list_backups(workspace_id, database_name=database_name, snapshot_id=snapshot_id),
            "can_manage": True,
            "interrupted_maintenance": await service.interrupted_maintenance(workspace_id),
        }
    )
    return response.model_dump(mode="json")


async def _sqlite_history_capture_jobs_payload(
    workspace_id: str,
    *,
    database_name: str | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    jobs = await get_sqlite_backup_queue_service().list_jobs(
        workspace_id,
        database_name=database_name,
        snapshot_id=snapshot_id,
        limit=50,
    )
    return SqliteHistoryCaptureJobListResponse.model_validate({"jobs": [_public_capture_job(job) for job in jobs]}).model_dump(mode="json")


async def _sqlite_history_event_payload(
    workspace_id: str,
    *,
    database_name: str | None = None,
    snapshot_id: str | None = None,
) -> tuple[str, bool]:
    history, jobs = await asyncio.gather(
        _sqlite_history_list_payload(workspace_id, database_name=database_name, snapshot_id=snapshot_id),
        _sqlite_history_capture_jobs_payload(workspace_id, database_name=database_name, snapshot_id=snapshot_id),
    )
    payload = json.dumps({"history": history, "jobs": jobs}, sort_keys=True, separators=(",", ":"))
    active = any(job["status"] in {"pending", "running"} for job in jobs["jobs"])
    return payload, active


@router.get("/workspaces/{workspace_id}/sqlite-history", response_model=SqliteHistoryListResponse)
async def list_sqlite_history(
    workspace_id: str, database_name: str | None = Query(default=None), snapshot_id: str | None = Query(default=None), user: Any = Depends(get_current_user)
):
    await _manage(workspace_id, user)
    return await _sqlite_history_list_payload(workspace_id, database_name=database_name, snapshot_id=snapshot_id)


@router.get("/workspaces/{workspace_id}/sqlite-history/events")
async def stream_sqlite_history_events(
    workspace_id: str,
    request: Request,
    database_name: str | None = Query(default=None),
    snapshot_id: str | None = Query(default=None),
    user: Any = Depends(get_current_user),
) -> StreamingResponse:
    await _manage(workspace_id, user)
    initial_payload, initial_active = await _sqlite_history_event_payload(
        workspace_id,
        database_name=database_name,
        snapshot_id=snapshot_id,
    )

    async def event_stream() -> AsyncIterator[str]:
        payload = initial_payload
        active = initial_active
        first_iteration = True
        last_payload: str | None = None
        while True:
            if await request.is_disconnected():
                return
            try:
                if not first_iteration:
                    await _manage(workspace_id, user)
                    payload, active = await _sqlite_history_event_payload(
                        workspace_id,
                        database_name=database_name,
                        snapshot_id=snapshot_id,
                    )
                else:
                    first_iteration = False
                if payload != last_payload:
                    last_payload = payload
                    yield "event: history_changed\ndata: {}\n\n"
                else:
                    yield ": keepalive\n\n"
            except asyncio.CancelledError:
                raise
            except HTTPException as exc:
                if exc.status_code in {401, 403}:
                    yield "event: access_revoked\ndata: {}\n\n"
                    return
                logger.exception("SQLite history event stream failed workspace_id=%s", workspace_id)
                return
            except Exception:
                logger.exception("SQLite history event stream failed workspace_id=%s", workspace_id)
                return
            await asyncio.sleep(2 if active else 5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.post("/workspaces/{workspace_id}/sqlite-history", response_model=SqliteHistoryCaptureResponse)
async def capture_sqlite_history(workspace_id: str, request: SqliteHistoryCaptureRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    queue = get_sqlite_backup_queue_service()
    enqueued_job = await queue.enqueue(
        workspace_id,
        trigger="manual",
        database_names={request.database_name},
        requested_by_id=user.id,
        request_key=request.request_id,
    )
    job = await queue.wait_for_job(workspace_id, enqueued_job["id"])
    if job is None:
        raise HTTPException(status_code=404, detail="Capture job not found")
    if job["status"] in {"cancelled", "interrupted"}:
        raise HTTPException(status_code=409, detail=f"Capture job {job['id']} is {job['status']}")

    backups = await get_sqlite_history_service().list_backups(workspace_id, database_name=request.database_name)
    backup = next((item for item in backups if item.get("id") in set(job.get("backup_ids") or [])), None)
    if backup is not None:
        return {"backup": backup}
    if job["status"] == "failed":
        raise HTTPException(status_code=503, detail=f"Capture job {job['id']} failed before creating a catalog record")
    raise HTTPException(status_code=404, detail="Database not found")


@router.post(
    "/workspaces/{workspace_id}/sqlite-history/capture-jobs",
    response_model=SqliteHistoryCaptureJobResponse,
    status_code=202,
)
async def enqueue_sqlite_history_capture_job(workspace_id: str, request: SqliteHistoryCaptureJobRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    job = await get_sqlite_backup_queue_service().enqueue(
        workspace_id,
        trigger="manual",
        database_names={request.database_name},
        requested_by_id=user.id,
        request_key=request.request_id,
    )
    return {"job": _public_capture_job(job)}


@router.get("/workspaces/{workspace_id}/sqlite-history/capture-jobs", response_model=SqliteHistoryCaptureJobListResponse)
async def list_sqlite_history_capture_jobs(
    workspace_id: str,
    database_name: str | None = Query(default=None),
    snapshot_id: str | None = Query(default=None),
    user: Any = Depends(get_current_user),
):
    await _manage(workspace_id, user)
    return await _sqlite_history_capture_jobs_payload(workspace_id, database_name=database_name, snapshot_id=snapshot_id)


@router.get("/workspaces/{workspace_id}/sqlite-history/capture-jobs/{job_id}", response_model=SqliteHistoryCaptureJobResponse)
async def get_sqlite_history_capture_job(workspace_id: str, job_id: str, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    job = await get_sqlite_backup_queue_service().get_job(workspace_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Capture job not found")
    return {"job": _public_capture_job(job)}


@router.post("/workspaces/{workspace_id}/sqlite-history/capture-jobs/{job_id}/cancel", response_model=SqliteHistoryCaptureJobResponse)
async def cancel_sqlite_history_capture_job(workspace_id: str, job_id: str, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    job = await get_sqlite_backup_queue_service().cancel(workspace_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Capture job not found")
    return {"job": _public_capture_job(job)}


@router.get("/workspaces/{workspace_id}/sqlite-history/{backup_id}/download")
async def download_sqlite_history(workspace_id: str, backup_id: str, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    if not await get_sqlite_history_service().runtime_history_active():
        # Compatibility before coordinated activation only.  Once a runtime is
        # configured, history bytes are never opened by this control plane.
        path = await get_sqlite_history_service().download_path(workspace_id, backup_id)

        async def legacy_stream() -> AsyncIterator[bytes]:
            try:
                with path.open("rb") as source:
                    while chunk := source.read(64 * 1024):
                        yield chunk
            finally:
                path.unlink(missing_ok=True)

        return StreamingResponse(
            legacy_stream(),
            media_type="application/vnd.sqlite3",
            headers={"Cache-Control": "no-store", "Content-Disposition": f'attachment; filename="{backup_id}.sqlite3"'},
        )

    config = get_runtime_manager_request_config()
    client = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout_seconds), follow_redirects=True)
    response = await client.send(
        client.build_request("GET", f"{config.base_url}/workspaces/{workspace_id}/sqlite-history/backups/{backup_id}/download", headers=config.headers),
        stream=True,
    )
    if response.status_code >= 400:
        detail = (await response.aread()).decode(errors="replace")[:256] or "Runtime SQLite history download failed"
        await response.aclose()
        await client.aclose()
        raise HTTPException(status_code=response.status_code if response.status_code < 500 else 502, detail=detail)

    async def runtime_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        runtime_stream(),
        media_type=response.headers.get("content-type", "application/vnd.sqlite3"),
        headers={"Cache-Control": "no-store", "Content-Disposition": f'attachment; filename="{backup_id}.sqlite3"'},
    )


@router.delete("/workspaces/{workspace_id}/sqlite-history/{backup_id}")
async def delete_sqlite_history(workspace_id: str, backup_id: str, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    await get_sqlite_history_service().delete(workspace_id, backup_id)
    return {"success": True}


@router.post("/workspaces/{workspace_id}/sqlite-history/{backup_id}/preview", response_model=SqliteHistoryPreview)
async def preview_sqlite_history(workspace_id: str, backup_id: str, request: SqliteHistoryPreviewRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    return await get_sqlite_history_service().preview(
        workspace_id,
        backup_id,
        mode=request.mode,
        conflict_policy=request.conflict_policy,
        table_policies=dict(request.table_policies) if request.table_policies is not None else None,
        user_id=user.id,
    )


@router.post("/workspaces/{workspace_id}/sqlite-history/restore", response_model=SqliteHistoryRestoreResponse)
async def restore_sqlite_history(workspace_id: str, request: SqliteHistoryRestoreRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    return await get_sqlite_history_service().apply(workspace_id, request.preview_id, user_id=user.id)


@router.post("/workspaces/{workspace_id}/sqlite-history/maintenance/{operation_id}/recover")
async def recover_sqlite_history(workspace_id: str, operation_id: str, request: SqliteHistoryRecoveryRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    return await get_sqlite_history_service().recover_operation(workspace_id, operation_id, action=request.action)
