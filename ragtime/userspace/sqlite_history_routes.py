"""Owner/admin-only API for protected SQLite history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ragtime.core.security import get_current_user
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


async def _manage(workspace_id: str, user: Any) -> None:
    from ragtime.userspace.service import userspace_service

    await userspace_service._enforce_workspace_access(workspace_id, user.id, required_role="owner", is_admin=getattr(user, "role", "") == "admin")


def _public_capture_job(job: dict[str, Any]) -> dict[str, Any]:
    """Exclude queue ownership and idempotency internals from API responses."""
    fields = {
        "id", "workspace_id", "trigger", "database_names", "snapshot_id",
        "snapshot_git_commit_hash", "status", "created_at", "available_at",
        "started_at", "finished_at", "updated_at", "completed_databases",
        "total_databases", "backup_ids", "error_message", "cancel_requested",
    }
    return {field: job[field] for field in fields if field in job}


@router.get("/workspaces/{workspace_id}/sqlite-history", response_model=SqliteHistoryListResponse)
async def list_sqlite_history(
    workspace_id: str, database_name: str | None = Query(default=None), snapshot_id: str | None = Query(default=None), user: Any = Depends(get_current_user)
):
    await _manage(workspace_id, user)
    service = get_sqlite_history_service()
    return {
        "workspace_id": workspace_id,
        "backups": await service.list_backups(workspace_id, database_name=database_name, snapshot_id=snapshot_id),
        "can_manage": True,
        "interrupted_maintenance": await service.interrupted_maintenance(workspace_id),
    }


@router.post("/workspaces/{workspace_id}/sqlite-history", response_model=SqliteHistoryCaptureResponse)
async def capture_sqlite_history(workspace_id: str, request: SqliteHistoryCaptureRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    from ragtime.userspace.sqlite_backup_queue import get_sqlite_backup_queue_service

    queue = get_sqlite_backup_queue_service()
    job = await queue.enqueue(
        workspace_id,
        trigger="manual",
        database_names={request.database_name},
        requested_by_id=user.id,
        request_key=request.request_id,
    )
    job = await queue.wait_for_job(workspace_id, job["id"])
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
async def enqueue_sqlite_history_capture_job(
    workspace_id: str, request: SqliteHistoryCaptureJobRequest, user: Any = Depends(get_current_user)
):
    await _manage(workspace_id, user)
    from ragtime.userspace.sqlite_backup_queue import get_sqlite_backup_queue_service

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
    from ragtime.userspace.sqlite_backup_queue import get_sqlite_backup_queue_service

    jobs = await get_sqlite_backup_queue_service().list_jobs(
        workspace_id, database_name=database_name, snapshot_id=snapshot_id, limit=50
    )
    return {"jobs": [_public_capture_job(job) for job in jobs]}


@router.get("/workspaces/{workspace_id}/sqlite-history/capture-jobs/{job_id}", response_model=SqliteHistoryCaptureJobResponse)
async def get_sqlite_history_capture_job(workspace_id: str, job_id: str, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    from ragtime.userspace.sqlite_backup_queue import get_sqlite_backup_queue_service

    job = await get_sqlite_backup_queue_service().get_job(workspace_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Capture job not found")
    return {"job": _public_capture_job(job)}


@router.post("/workspaces/{workspace_id}/sqlite-history/capture-jobs/{job_id}/cancel", response_model=SqliteHistoryCaptureJobResponse)
async def cancel_sqlite_history_capture_job(workspace_id: str, job_id: str, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    from ragtime.userspace.sqlite_backup_queue import get_sqlite_backup_queue_service

    job = await get_sqlite_backup_queue_service().cancel(workspace_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Capture job not found")
    return {"job": _public_capture_job(job)}


@router.get("/workspaces/{workspace_id}/sqlite-history/{backup_id}/download")
async def download_sqlite_history(workspace_id: str, backup_id: str, background_tasks: BackgroundTasks, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    path = await get_sqlite_history_service().download_path(workspace_id, backup_id)

    def cleanup_temp_download(temp_path: Path) -> None:
        """Remove temporary download copy after transmission."""
        temp_path.unlink(missing_ok=True)

    background_tasks.add_task(cleanup_temp_download, path)
    return FileResponse(path, filename=f"{backup_id}.sqlite3", media_type="application/vnd.sqlite3", headers={"Cache-Control": "no-store"})


@router.delete("/workspaces/{workspace_id}/sqlite-history/{backup_id}")
async def delete_sqlite_history(workspace_id: str, backup_id: str, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    await get_sqlite_history_service().delete(workspace_id, backup_id)
    return {"success": True}


@router.post("/workspaces/{workspace_id}/sqlite-history/{backup_id}/preview", response_model=SqliteHistoryPreview)
async def preview_sqlite_history(workspace_id: str, backup_id: str, request: SqliteHistoryPreviewRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    return await get_sqlite_history_service().preview(
        workspace_id, backup_id, mode=request.mode, conflict_policy=request.conflict_policy, table_policies=request.table_policies, user_id=user.id
    )


@router.post("/workspaces/{workspace_id}/sqlite-history/restore", response_model=SqliteHistoryRestoreResponse)
async def restore_sqlite_history(workspace_id: str, request: SqliteHistoryRestoreRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    return await get_sqlite_history_service().apply(workspace_id, request.preview_id, user_id=user.id)


@router.post("/workspaces/{workspace_id}/sqlite-history/maintenance/{operation_id}/recover")
async def recover_sqlite_history(workspace_id: str, operation_id: str, request: SqliteHistoryRecoveryRequest, user: Any = Depends(get_current_user)):
    await _manage(workspace_id, user)
    return await get_sqlite_history_service().recover_operation(workspace_id, operation_id, action=request.action)
