from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ragtime.core.security import require_admin
from ragtime.indexer.server_backup_service import server_backup_service


class CreateBackupJobRequest(BaseModel):
    scope: str = Field(description="Backup scope: full, database, or files")
    encrypt: bool = Field(default=True, description="Whether to emit an encrypted backup artifact")
    password: str | None = Field(default=None, description="Encryption password kept only in request/job memory")


class CreateRestoreJobRequest(BaseModel):
    upload_id: str = Field(description="Previously staged upload identifier")
    password: str | None = Field(default=None, description="Backup password kept only in request/job memory")
    scope_override: str | None = Field(default=None, description="Optional restore scope override")
    skip_migrations: bool = Field(default=False, description="Skip migrations during restore")
    postgres_data_only: bool = Field(default=False, description="Restore PostgreSQL data only")
    replace_data: bool = Field(default=False, description="Replace application data during restore")
    mirror_local_admin_access: bool = Field(default=False, description="Mirror local admin access after restore")
    mirror_local_admin_from: str = Field(default="auto", description="Source to mirror local admin access from")
    local_admin_username: str | None = Field(default=None, description="Optional local admin username to mirror")


class CommitRestoreJobRequest(BaseModel):
    confirmation_text: str = Field(description="Typed destructive confirmation phrase")
    acknowledge_legacy_key: bool = Field(default=False, description="Acknowledge legacy embedded key restore when required")


router = APIRouter(prefix="/indexes/server-backups", tags=["Server Backup"])


@router.post("/jobs", status_code=202)
async def create_backup_job(request: CreateBackupJobRequest, _user: Any = Depends(require_admin)):
    return await server_backup_service.create_backup_job(
        scope=request.scope,
        encrypt=request.encrypt,
        password=request.password,
    )


@router.get("/active-jobs")
async def get_active_jobs(_user: Any = Depends(require_admin)):
    return await server_backup_service.get_active_jobs()


@router.get("/jobs/{job_id}")
async def get_backup_job(job_id: str, _user: Any = Depends(require_admin)):
    return await server_backup_service.get_backup_job(job_id)


@router.post("/jobs/{job_id}/cancel")
async def cancel_backup_job(job_id: str, _user: Any = Depends(require_admin)):
    return await server_backup_service.cancel_backup_job(job_id)


@router.get("/jobs/{job_id}/download")
async def download_backup_job(job_id: str, _user: Any = Depends(require_admin)):
    return await server_backup_service.get_backup_download_response(job_id)


@router.get("/exports")
async def list_backup_exports(_user: Any = Depends(require_admin)):
    return await server_backup_service.list_backup_exports()


@router.delete("/jobs/{job_id}")
async def delete_backup_job(job_id: str, _user: Any = Depends(require_admin)):
    return await server_backup_service.delete_backup_job(job_id)


@router.post("/uploads", status_code=202)
async def upload_restore_archive(
    file: UploadFile = File(..., description="Backup archive to validate and restore"),
    _user: Any = Depends(require_admin),
):
    return await server_backup_service.stage_restore_upload(file, filename=file.filename)


@router.post("/restore-jobs", status_code=202)
async def create_restore_job(request: CreateRestoreJobRequest, _user: Any = Depends(require_admin)):
    return await server_backup_service.create_restore_job(
        upload_id=request.upload_id,
        password=request.password,
        scope_override=request.scope_override,
        skip_migrations=request.skip_migrations,
        postgres_data_only=request.postgres_data_only,
        replace_data=request.replace_data,
        mirror_local_admin_access=request.mirror_local_admin_access,
        mirror_local_admin_from=request.mirror_local_admin_from,
        local_admin_username=request.local_admin_username,
    )


@router.get("/restore-jobs/{job_id}")
async def get_restore_job(job_id: str, _user: Any = Depends(require_admin)):
    return await server_backup_service.get_restore_job(job_id)


@router.post("/restore-jobs/{job_id}/deployment-environment")
async def recover_restore_job_deployment_environment(job_id: str, _user: Any = Depends(require_admin)):
    payload = await server_backup_service.recover_deployment_environment(job_id)
    return JSONResponse(
        content=payload,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.post("/restore-jobs/{job_id}/commit", status_code=202)
async def commit_restore_job(job_id: str, request: CommitRestoreJobRequest, _user: Any = Depends(require_admin)):
    return await server_backup_service.commit_restore_job(
        job_id,
        confirmation_text=request.confirmation_text,
        acknowledge_legacy_key=request.acknowledge_legacy_key,
    )
