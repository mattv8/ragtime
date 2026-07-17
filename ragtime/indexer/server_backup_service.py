from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import threading
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryFile
from typing import Any, Callable

import httpx
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.core.maintenance import ProcessMaintenanceState

_UPLOAD_CHUNK_BYTES = 1024 * 1024
_DEFAULT_UPLOAD_LIMIT_BYTES = 10 * 1024 * 1024 * 1024
_DEFAULT_MIN_FREE_BYTES = 512 * 1024 * 1024
_DEFAULT_UPLOAD_TTL_SECONDS = 24 * 60 * 60
_DEFAULT_JOB_TTL_SECONDS = 7 * 24 * 60 * 60
_DEFAULT_MAX_PASSWORD_BYTES = 8192
_DEFAULT_MAINTENANCE_HEARTBEAT_INTERVAL_SECONDS = 30.0
_ACTIVE_BACKUP_STATUSES = {"pending", "running"}
_ACTIVE_RESTORE_STATUSES = {"validating", "ready_for_commit", "restoring"}
_RECOVERABLE_RESTORE_STATUSES = {"validating", "restoring"}
_RECOVERABLE_BACKUP_STATUSES = {"pending", "running"}
_TERMINAL_BACKUP_STATUSES = {"completed", "delivered", "failed", "cancelled", "interrupted"}
_TERMINAL_RESTORE_STATUSES = {"completed", "failed", "cancelled"}
_PERSISTENT_EXPORT_BACKUP_STATUSES = {"completed", "delivered"}

logger = get_logger(__name__)

_PROGRESS_DETAIL_ALLOWLIST = {
    "item_count",
    "data_item_count",
    "current_item",
    "processed_items",
    "total_items",
    "scope",
    "encrypted",
    "format",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class ServerBackupService:
    def __init__(
        self,
        *,
        root_dir: Path | None = None,
        upload_ttl_seconds: int = _DEFAULT_UPLOAD_TTL_SECONDS,
        job_ttl_seconds: int = _DEFAULT_JOB_TTL_SECONDS,
        upload_limit_bytes: int = _DEFAULT_UPLOAD_LIMIT_BYTES,
        min_free_bytes: int = _DEFAULT_MIN_FREE_BYTES,
        max_password_bytes: int = _DEFAULT_MAX_PASSWORD_BYTES,
    ) -> None:
        self._root_dir = root_dir or (Path(settings.index_data_path) / "_server_backups")
        self._backup_dir = self._root_dir / "backup_jobs"
        self._upload_dir = self._root_dir / "restore_uploads"
        self._restore_dir = self._root_dir / "restore_jobs"
        self._backup_jobs: dict[str, dict[str, Any]] = {}
        self._restore_jobs: dict[str, dict[str, Any]] = {}
        self._uploads: dict[str, dict[str, Any]] = {}
        self._backup_tasks: dict[str, asyncio.Task[None]] = {}
        self._restore_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._restart_signaler: Callable[[str], None] = lambda _job_id: None
        self._maintenance_state: ProcessMaintenanceState | None = None
        self._upload_ttl = max(upload_ttl_seconds, 0)
        self._job_ttl = max(job_ttl_seconds, 0)
        self._upload_limit_bytes = max(upload_limit_bytes, 1)
        self._min_free_bytes = max(min_free_bytes, 0)
        self._max_password_bytes = max(max_password_bytes, 1)
        self._maintenance_heartbeat_interval_seconds = _DEFAULT_MAINTENANCE_HEARTBEAT_INTERVAL_SECONDS

    async def startup(self) -> None:
        for path in (self._root_dir, self._backup_dir, self._upload_dir, self._restore_dir):
            path.mkdir(parents=True, exist_ok=True)
            with suppress(OSError):
                path.chmod(stat.S_IRWXU)
        await self._load_sidecars()
        self._mark_orphaned_jobs_interrupted()
        self._cleanup_expired_state()

    async def shutdown(self) -> None:
        tasks = [*self._backup_tasks.values(), *self._restore_tasks.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def set_restart_signaler(self, callback: Callable[[str], None]) -> None:
        self._restart_signaler = callback

    def set_process_maintenance_state(self, state: ProcessMaintenanceState) -> None:
        self._maintenance_state = state

    async def create_backup_job(self, *, scope: str, encrypt: bool, password: str | None) -> dict[str, Any]:
        module = import_module("ragtime.core.server_backup")
        backup_scope = module.BackupScope(scope)
        if encrypt and not password:
            raise HTTPException(status_code=400, detail="Password is required for encrypted backups")

        job_id = uuid.uuid4().hex
        created_at = _utcnow()
        suffix = ".ragbak" if encrypt else ".tar.gz"
        artifact_path = self._backup_dir / f"{job_id}{suffix}"
        record = {
            "id": job_id,
            "status": "pending",
            "phase": "queued",
            "progress": 0,
            "message": "Backup queued",
            "scope": backup_scope.value,
            "encrypt": encrypt,
            "created_at": created_at,
            "updated_at": created_at,
            "artifact_path": str(artifact_path),
            "download_name": f"ragtime-backup-{backup_scope.value}-{created_at.strftime('%Y%m%dT%H%M%SZ')}{suffix}",
            "manifest": None,
            "error": None,
            "delivered_at": None,
            "cancel_requested": False,
        }
        async with self._lock:
            self._assert_no_active_job_locked()
            self._backup_jobs[job_id] = record
        self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(record))
        self._backup_tasks[job_id] = asyncio.create_task(self._run_backup_job(job_id, artifact_path, password))
        return self._snapshot(record)

    async def get_backup_job(self, job_id: str) -> dict[str, Any]:
        job = self._backup_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Backup job not found")
        return self._snapshot(job)

    async def get_active_jobs(self) -> dict[str, Any]:
        self._cleanup_expired_state()
        backup_job = self._latest_active_job(self._backup_jobs, _ACTIVE_BACKUP_STATUSES)
        restore_job = self._latest_active_job(self._restore_jobs, _ACTIVE_RESTORE_STATUSES)
        return {
            "backup_job": self._snapshot(backup_job) if backup_job is not None else None,
            "restore_job": self._snapshot(restore_job) if restore_job is not None else None,
        }

    async def cancel_backup_job(self, job_id: str) -> dict[str, Any]:
        job = self._backup_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Backup job not found")
        if job["status"] not in _ACTIVE_BACKUP_STATUSES:
            return self._snapshot(job)
        job["cancel_requested"] = True
        job["status"] = "cancelled"
        job["phase"] = "cancelled"
        job["progress"] = job.get("progress") or 0
        job["message"] = "Backup cancelled"
        job["updated_at"] = _utcnow()
        Path(job["artifact_path"]).unlink(missing_ok=True)
        task = self._backup_tasks.get(job_id)
        if task is not None:
            task.cancel()
        self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(job))
        return self._snapshot(job)

    async def get_backup_download_response(self, job_id: str) -> FileResponse:
        job = self._backup_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Backup job not found")
        if job["status"] not in _PERSISTENT_EXPORT_BACKUP_STATUSES:
            raise HTTPException(status_code=409, detail="Backup artifact is not ready for download")
        artifact_path = Path(job["artifact_path"])
        if not artifact_path.exists():
            raise HTTPException(status_code=410, detail="Backup artifact is no longer available")
        return FileResponse(
            path=artifact_path,
            filename=job["download_name"],
            media_type="application/octet-stream",
            background=BackgroundTask(self._complete_backup_delivery, job_id),
        )

    async def list_backup_exports(self) -> dict[str, list[dict[str, Any]]]:
        self._cleanup_expired_state()
        exports: list[dict[str, Any]] = []
        for job in self._backup_jobs.values():
            if job.get("status") not in _PERSISTENT_EXPORT_BACKUP_STATUSES:
                continue
            artifact_path = Path(job["artifact_path"])
            if not artifact_path.exists():
                continue
            created_at = job.get("created_at")
            exports.append(
                {
                    "job_id": job["id"],
                    "file_name": job["download_name"],
                    "size_bytes": artifact_path.stat().st_size,
                    "created_at": _isoformat(created_at),
                    "scope": job.get("scope"),
                    "encrypted": job.get("encrypt"),
                    "delivered_at": _isoformat(job.get("delivered_at")),
                    "_sort_created_at": created_at,
                }
            )
        exports.sort(
            key=lambda item: item.get("_sort_created_at") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        for item in exports:
            item.pop("_sort_created_at", None)
        return {"exports": exports}

    async def delete_backup_job(self, job_id: str) -> dict[str, Any]:
        job = self._backup_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Backup job not found")
        if job.get("status") in _ACTIVE_BACKUP_STATUSES:
            raise HTTPException(status_code=409, detail="Active backup jobs cannot be deleted")
        Path(job["artifact_path"]).unlink(missing_ok=True)
        self._backup_job_sidecar_path(job_id).unlink(missing_ok=True)
        self._backup_jobs.pop(job_id, None)
        return {"success": True, "job_id": job_id}

    async def stage_restore_upload(self, file: UploadFile, *, filename: str | None = None) -> dict[str, Any]:
        self._cleanup_expired_state()
        upload_name = filename or file.filename or f"restore-{uuid.uuid4().hex}.bin"
        upload_id = uuid.uuid4().hex
        content_length = self._extract_content_length(file)
        if content_length is not None and content_length > self._upload_limit_bytes:
            raise HTTPException(status_code=413, detail="Restore upload exceeds the maximum size limit")
        self._ensure_free_space(content_length or 0)

        target_path = self._upload_dir / f"{upload_id}-{Path(upload_name).name}"
        size_bytes = 0
        try:
            with target_path.open("wb") as handle:
                while True:
                    chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > self._upload_limit_bytes:
                        raise HTTPException(status_code=413, detail="Restore upload exceeds the maximum size limit")
                    self._ensure_free_space(0)
                    await asyncio.to_thread(handle.write, chunk)
        except Exception:
            target_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        record = {
            "upload_id": upload_id,
            "filename": upload_name,
            "stored_path": str(target_path),
            "size_bytes": size_bytes,
            "created_at": _utcnow(),
        }
        self._uploads[upload_id] = record
        self._write_json(self._upload_sidecar_path(upload_id), self._serialize_upload(record))
        return self._snapshot_upload(record)

    async def create_restore_job(
        self,
        *,
        upload_id: str,
        password: str | None,
        scope_override: str | None = None,
        skip_migrations: bool = False,
        postgres_data_only: bool = False,
        replace_data: bool = False,
        mirror_local_admin_access: bool = False,
        mirror_local_admin_from: str = "auto",
        local_admin_username: str | None = None,
    ) -> dict[str, Any]:
        module = import_module("ragtime.core.server_backup")
        scope_value = module.BackupScope(scope_override) if scope_override else None
        job_id = uuid.uuid4().hex
        created_at = _utcnow()
        async with self._lock:
            self._assert_no_active_job_locked()
            upload = self._uploads.get(upload_id)
            if upload is None:
                raise HTTPException(status_code=404, detail="Restore upload not found")
            if not Path(upload["stored_path"]).exists():
                raise HTTPException(status_code=404, detail="Restore upload file is missing")
            record = {
                "id": job_id,
                "status": "validating",
                "phase": "validation_pending",
                "progress": 0,
                "message": "Validating restore archive",
                "created_at": created_at,
                "updated_at": created_at,
                "upload_id": upload_id,
                "upload_filename": upload["filename"],
                "upload_path": upload["stored_path"],
                "manifest": None,
                "error": None,
                "required_confirmation": self._confirmation_phrase(),
                "restart_required": False,
                "requires_legacy_key_acknowledgement": False,
                "scope_override": scope_value.value if scope_value else None,
                "skip_migrations": skip_migrations,
                "pg_data_only": postgres_data_only,
                "replace_data": replace_data,
                "mirror_local_admin_access": mirror_local_admin_access,
                "mirror_local_admin_from": mirror_local_admin_from,
                "local_admin_username": local_admin_username,
                "acknowledge_legacy_key": False,
                "_password": password,
            }
            self._restore_jobs[job_id] = record
        self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(record))
        self._restore_tasks[job_id] = asyncio.create_task(self._run_restore_validation(job_id, password))
        return self._snapshot(record)

    async def get_restore_job(self, job_id: str) -> dict[str, Any]:
        job = self._restore_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Restore job not found")
        return self._snapshot(job)

    async def commit_restore_job(
        self,
        job_id: str,
        *,
        confirmation_text: str,
        acknowledge_legacy_key: bool,
    ) -> dict[str, Any]:
        async with self._lock:
            job = self._restore_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Restore job not found")
            if job["status"] != "ready_for_commit":
                raise HTTPException(status_code=409, detail="Restore job is not ready to commit")
            if confirmation_text != job["required_confirmation"]:
                raise HTTPException(status_code=400, detail="Confirmation text does not match")
            if job["requires_legacy_key_acknowledgement"] and not acknowledge_legacy_key:
                raise HTTPException(status_code=400, detail="Legacy key acknowledgement is required")
            job["acknowledge_legacy_key"] = acknowledge_legacy_key
            job["status"] = "restoring"
            job["phase"] = "commit_requested"
            job["message"] = "Restore commit requested"
            job["updated_at"] = _utcnow()
        self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
        self._restore_tasks[job_id] = asyncio.create_task(self._run_restore_commit(job_id))
        return self._snapshot(job)

    async def _run_backup_job(self, job_id: str, artifact_path: Path, password: str | None) -> None:
        job = self._backup_jobs[job_id]
        maintenance_lease: Any = None
        runtime_lease = self._runtime_lease_id(job_id, operation_kind="backup")
        runtime_lease_owned = False
        heartbeat_task: asyncio.Task[None] | None = None
        heartbeat_failure: asyncio.Future[None] | None = None
        try:
            job["status"] = "running"
            job["progress"] = 10
            job["phase"] = "start"
            job["message"] = "Creating backup"
            job["updated_at"] = _utcnow()
            self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(job))
            if self._backup_requires_maintenance(job.get("scope")):
                job["phase"] = "acquiring_maintenance"
                job["message"] = "Acquiring maintenance lease"
                job["updated_at"] = _utcnow()
                self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(job))
                maintenance_lease = await self._acquire_maintenance_lease(job_id, operation_kind="backup")
                runtime_lease_owned = True
                acquired_runtime_lease = await self._acquire_runtime_lease(job_id, operation_kind="backup")
                if acquired_runtime_lease is not None:
                    runtime_lease = acquired_runtime_lease
                heartbeat_task, heartbeat_failure = self._start_maintenance_heartbeat(
                    process_lease_id=maintenance_lease,
                    runtime_lease_id=runtime_lease,
                )
                manifest = await self._run_backup_with_heartbeat(
                    scope=job["scope"],
                    encrypt=job["encrypt"],
                    output_path=artifact_path,
                    password=password,
                    heartbeat_failure=heartbeat_failure,
                    progress_callback=self._job_progress_callback(job_id, backup=True),
                )
            else:
                manifest = await self._create_backup_archive(
                    scope=job["scope"],
                    encrypt=job["encrypt"],
                    output_path=artifact_path,
                    password=password,
                    progress_callback=self._job_progress_callback(job_id, backup=True),
                )
            if job["cancel_requested"]:
                artifact_path.unlink(missing_ok=True)
                job["status"] = "cancelled"
                job["phase"] = "cancelled"
                job["message"] = "Backup cancelled"
            else:
                job["status"] = "completed"
                job["phase"] = "complete"
                job["progress"] = 100
                job["message"] = "Backup ready for download"
                job["manifest"] = self._normalize_manifest(manifest)
            job["updated_at"] = _utcnow()
            self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(job))
        except asyncio.CancelledError:
            artifact_path.unlink(missing_ok=True)
            job["status"] = "cancelled"
            job["phase"] = "cancelled"
            job["message"] = "Backup cancelled"
            job["updated_at"] = _utcnow()
            self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(job))
            raise
        except Exception as exc:
            artifact_path.unlink(missing_ok=True)
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = str(exc)
            job["message"] = str(exc)
            job["updated_at"] = _utcnow()
            self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(job))
            logger.warning("Server backup job %s failed: %s", job_id, exc)
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if runtime_lease_owned and runtime_lease is not None:
                await self._release_runtime_lease(runtime_lease)
            if maintenance_lease is not None:
                await self._release_maintenance_lease(maintenance_lease)

    async def _run_restore_validation(self, job_id: str, password: str | None) -> None:
        job = self._restore_jobs[job_id]
        try:
            manifest = await self._inspect_backup_archive(
                Path(job["upload_path"]),
                password,
                self._job_progress_callback(job_id, backup=False),
            )
            normalized = self._normalize_manifest(manifest)
            job["manifest"] = normalized
            job["requires_legacy_key_acknowledgement"] = bool(normalized.get("legacy_embedded_key", False))
            job["status"] = "ready_for_commit"
            job["phase"] = "validation_complete"
            job["progress"] = max(job.get("progress") or 0, 40)
            job["message"] = "Restore validated; confirmation required"
            job["updated_at"] = _utcnow()
            self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            job["phase"] = "cancelled"
            job["message"] = "Restore validation cancelled"
            job["updated_at"] = _utcnow()
            self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
            raise
        except Exception as exc:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = str(exc)
            job["message"] = str(exc)
            job["updated_at"] = _utcnow()
            self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
            logger.warning("Server restore validation %s failed: %s", job_id, exc)

    async def _run_restore_commit(self, job_id: str) -> None:
        job = self._restore_jobs[job_id]
        maintenance_lease: Any = None
        runtime_lease: Any = self._runtime_lease_id(job_id)
        runtime_lease_owned = False
        heartbeat_task: asyncio.Task[None] | None = None
        heartbeat_failure: asyncio.Future[None] | None = None
        success = False
        should_restart = False
        try:
            job["phase"] = "acquiring_maintenance"
            job["progress"] = max(job.get("progress") or 0, 42)
            job["message"] = "Acquiring maintenance lease"
            job["updated_at"] = _utcnow()
            self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
            maintenance_lease = await self._acquire_maintenance_lease(job_id)
            runtime_lease_owned = True
            acquired_runtime_lease = await self._acquire_runtime_lease(job_id)
            if acquired_runtime_lease is not None:
                runtime_lease = acquired_runtime_lease
            heartbeat_task, heartbeat_failure = self._start_maintenance_heartbeat(
                process_lease_id=maintenance_lease,
                runtime_lease_id=runtime_lease,
            )
            manifest = await self._run_restore_with_heartbeat(
                archive_path=Path(job["upload_path"]),
                password=job.get("_password"),
                options={
                    "scope_override": job.get("scope_override"),
                    "skip_migrations": job.get("skip_migrations", False),
                    "pg_data_only": job.get("pg_data_only", False),
                    "replace_data": job.get("replace_data", False),
                    "acknowledge_legacy_key": job.get("acknowledge_legacy_key", False),
                    "mirror_local_admin_access": job.get("mirror_local_admin_access", False),
                    "mirror_local_admin_from": job.get("mirror_local_admin_from", "auto"),
                    "local_admin_username": job.get("local_admin_username"),
                },
                heartbeat_failure=heartbeat_failure,
                progress_callback=self._job_progress_callback(job_id, backup=False),
            )
            job["manifest"] = self._normalize_manifest(manifest)
            job["status"] = "completed"
            job["phase"] = "complete"
            job["progress"] = 100
            job["message"] = "Restore completed; restart requested"
            job["restart_required"] = True
            job["updated_at"] = _utcnow()
            self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
            success = True
            should_restart = True
            Path(job["upload_path"]).unlink(missing_ok=True)
            self._upload_sidecar_path(job["upload_id"]).unlink(missing_ok=True)
            self._uploads.pop(job["upload_id"], None)
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            job["phase"] = "cancelled"
            job["message"] = "Restore cancelled"
            job["updated_at"] = _utcnow()
            self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
            raise
        except Exception as exc:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = str(exc)
            job["message"] = str(exc)
            job["updated_at"] = _utcnow()
            self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
            logger.warning("Server restore commit %s failed: %s", job_id, exc)
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if runtime_lease_owned and runtime_lease is not None:
                await self._release_runtime_lease(runtime_lease)
            if maintenance_lease is not None:
                await self._release_maintenance_lease(maintenance_lease)
            job.pop("_password", None)
            if success:
                self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))
            if should_restart:
                self._restart_signaler(job_id)

    async def _create_backup_archive(
        self,
        *,
        scope: str,
        encrypt: bool,
        output_path: Path,
        password: str | None,
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        module = import_module("ragtime.core.server_backup")

        def _call() -> Any:
            with self._password_fd(password) as password_fd:
                options = self._build_backup_options(module.BackupOptions, module.BackupScope(scope), output_path, encrypt, password_fd)
                return module.create_backup(options, progress=progress_callback)

        return self._normalize_manifest(await asyncio.to_thread(_call))

    async def _inspect_backup_archive(
        self,
        archive_path: Path,
        password: str | None,
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        module = import_module("ragtime.core.server_backup")
        return self._normalize_manifest(await asyncio.to_thread(module.inspect_backup, archive_path, password, progress_callback))

    async def _restore_backup_archive(
        self,
        archive_path: Path,
        *,
        password: str | None,
        options: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        module = import_module("ragtime.core.server_backup")

        def _call() -> Any:
            with self._password_fd(password) as password_fd:
                restore_options = self._build_restore_options(module.RestoreOptions, archive_path, password_fd, options)
                return module.restore_backup(restore_options, progress=progress_callback)

        return self._normalize_manifest(await asyncio.to_thread(_call))

    async def _acquire_maintenance_lease(self, job_id: str, *, operation_kind: str = "restore") -> str | None:
        lease_id = self._runtime_lease_id(job_id, operation_kind=operation_kind)
        if self._maintenance_state is not None:
            await self._maintenance_state.acquire(lease_id, reason=f"server-{operation_kind}:{job_id}")
            return lease_id
        return None

    async def _release_maintenance_lease(self, lease: Any) -> None:
        if self._maintenance_state is not None and lease is not None:
            await self._maintenance_state.release(str(lease))

    async def _renew_process_maintenance_lease(self, lease_id: str) -> None:
        if self._maintenance_state is None:
            return
        renew = getattr(self._maintenance_state, "renew", None)
        if callable(renew):
            result = renew(lease_id)
            if asyncio.iscoroutine(result):
                await result
            return
        await self._maintenance_state.acquire(lease_id)

    async def _acquire_runtime_lease(self, job_id: str, *, operation_kind: str = "restore") -> str | None:
        token = (settings.userspace_runtime_auth_token or "").strip()
        base_url = (settings.userspace_runtime_manager_url or "").strip().rstrip("/")
        if not token or not base_url:
            return None
        lease_id = self._runtime_lease_id(job_id, operation_kind=operation_kind)
        async with httpx.AsyncClient(timeout=settings.userspace_runtime_manager_timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/maintenance/lease",
                headers={"Authorization": f"Bearer {token}"},
                json={"lease_id": lease_id, "reason": f"server-{operation_kind}:{job_id}", "retry_after_seconds": 60},
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=409, detail="Runtime maintenance lease request failed")
        return lease_id

    async def _release_runtime_lease(self, lease: Any) -> None:
        token = (settings.userspace_runtime_auth_token or "").strip()
        base_url = (settings.userspace_runtime_manager_url or "").strip().rstrip("/")
        if lease is None or not token or not base_url:
            return
        async with httpx.AsyncClient(timeout=settings.userspace_runtime_manager_timeout_seconds) as client:
            with suppress(httpx.HTTPError):
                await client.delete(
                    f"{base_url}/maintenance/lease/{lease}",
                    headers={"Authorization": f"Bearer {token}"},
                )

    async def _renew_runtime_maintenance_lease(self, lease_id: str) -> None:
        token = (settings.userspace_runtime_auth_token or "").strip()
        base_url = (settings.userspace_runtime_manager_url or "").strip().rstrip("/")
        if not token or not base_url:
            return
        async with httpx.AsyncClient(timeout=settings.userspace_runtime_manager_timeout_seconds) as client:
            response = await client.put(
                f"{base_url}/maintenance/lease/{lease_id}",
                headers={"Authorization": f"Bearer {token}"},
                json={},
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=409, detail="Runtime maintenance lease renew failed")

    def _start_maintenance_heartbeat(
        self,
        *,
        process_lease_id: str | None,
        runtime_lease_id: str | None,
    ) -> tuple[asyncio.Task[None], asyncio.Future[None]]:
        failure: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(self._maintenance_heartbeat_interval_seconds)
                    if process_lease_id is not None:
                        await self._renew_process_maintenance_lease(process_lease_id)
                    if runtime_lease_id is not None:
                        await self._renew_runtime_maintenance_lease(str(runtime_lease_id))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not failure.done():
                    failure.set_exception(exc)

        return asyncio.create_task(_heartbeat()), failure

    async def _run_restore_with_heartbeat(
        self,
        *,
        archive_path: Path,
        password: str | None,
        options: dict[str, Any],
        heartbeat_failure: asyncio.Future[None] | None,
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        restore_task = asyncio.create_task(
            self._restore_backup_archive(
                archive_path,
                password=password,
                options=options,
                progress_callback=progress_callback,
            )
        )
        heartbeat_task: asyncio.Task[None] | None = None
        try:
            if heartbeat_failure is None:
                return await restore_task
            heartbeat_task = asyncio.create_task(self._await_heartbeat_failure(heartbeat_failure))
            done, _ = await asyncio.wait({restore_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat_task in done:
                restore_task.cancel()
                await asyncio.gather(restore_task, return_exceptions=True)
                await heartbeat_task
            return await restore_task
        finally:
            if heartbeat_task is not None and not heartbeat_task.done():
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if not restore_task.done():
                restore_task.cancel()
                await asyncio.gather(restore_task, return_exceptions=True)

    async def _await_heartbeat_failure(self, heartbeat_failure: asyncio.Future[None]) -> None:
        await heartbeat_failure

    async def _run_backup_with_heartbeat(
        self,
        *,
        scope: str,
        encrypt: bool,
        output_path: Path,
        password: str | None,
        heartbeat_failure: asyncio.Future[None] | None,
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        backup_task = asyncio.create_task(
            self._create_backup_archive(
                scope=scope,
                encrypt=encrypt,
                output_path=output_path,
                password=password,
                progress_callback=progress_callback,
            )
        )
        heartbeat_task: asyncio.Task[None] | None = None
        try:
            if heartbeat_failure is None:
                return await backup_task
            heartbeat_task = asyncio.create_task(self._await_heartbeat_failure(heartbeat_failure))
            done, _ = await asyncio.wait({backup_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat_task in done:
                backup_task.cancel()
                await asyncio.gather(backup_task, return_exceptions=True)
                await heartbeat_task
            return await backup_task
        finally:
            if heartbeat_task is not None and not heartbeat_task.done():
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if not backup_task.done():
                backup_task.cancel()
                await asyncio.gather(backup_task, return_exceptions=True)

    def _build_backup_options(self, options_type: Any, scope: Any, output_path: Path, encrypt: bool, password_fd: int | None) -> Any:
        if options_type is None:
            return None
        return options_type(
            scope=scope,
            output_path=output_path,
            encrypt=encrypt,
            password_fd=password_fd,
        )

    def _build_restore_options(self, options_type: Any, archive_path: Path, password_fd: int | None, options: dict[str, Any]) -> Any:
        module = import_module("ragtime.core.server_backup")
        scope_override = options.get("scope_override")
        if isinstance(scope_override, str):
            scope_override = module.BackupScope(scope_override)
        return options_type(
            archive_path=archive_path,
            scope_override=scope_override,
            skip_migrations=options.get("skip_migrations", False),
            pg_data_only=options.get("pg_data_only", False),
            replace_data=options.get("replace_data", False),
            acknowledge_legacy_key=options.get("acknowledge_legacy_key", False),
            restore_confirmation=self._confirmation_phrase(),
            mirror_local_admin_access=options.get("mirror_local_admin_access", False),
            mirror_local_admin_from=options.get("mirror_local_admin_from", "auto"),
            local_admin_username=options.get("local_admin_username"),
            password_fd=password_fd,
        )

    def _confirmation_phrase(self) -> str:
        module = import_module("ragtime.core.server_backup")
        return module._confirmation_phrase()

    async def recover_deployment_environment(self, job_id: str) -> dict[str, object]:
        job = self._restore_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Restore job not found")
        if job.get("status") != "ready_for_commit":
            raise HTTPException(status_code=409, detail="Deployment environment recovery is only available for validated restore jobs")
        manifest = job.get("manifest")
        if not isinstance(manifest, dict):
            raise HTTPException(status_code=409, detail="Deployment environment recovery is not available for this restore job")
        if manifest.get("scope") != "full" or not manifest.get("encrypted"):
            raise HTTPException(status_code=409, detail="Deployment environment recovery is only available for encrypted full backups")
        manifest_names = manifest.get("deployment_environment_variables")
        if not isinstance(manifest_names, list) or not manifest_names or any(not isinstance(name, str) for name in manifest_names):
            raise HTTPException(status_code=409, detail="Deployment environment recovery data is not available for this restore job")
        password = job.get("_password")
        if not isinstance(password, str) or not password:
            raise HTTPException(status_code=409, detail="Deployment environment recovery requires the original in-memory password")
        upload_path_value = job.get("upload_path")
        if not isinstance(upload_path_value, str) or not upload_path_value:
            raise HTTPException(status_code=409, detail="Restore upload is no longer available")
        upload_path = Path(upload_path_value)
        if not upload_path.exists():
            raise HTTPException(status_code=409, detail="Restore upload is no longer available")

        module = import_module("ragtime.core.server_backup")
        try:
            recovered = await asyncio.to_thread(module.recover_deployment_environment, upload_path, password)
        except HTTPException:
            raise
        except Exception:
            logger.warning("Deployment environment recovery failed for restore job %s", job_id)
            raise HTTPException(status_code=422, detail="Deployment environment recovery failed") from None

        payload = self._normalize_recovered_deployment_environment(recovered)
        expected_names = sorted(manifest_names)
        if payload["variable_names"] != expected_names:
            raise HTTPException(status_code=422, detail="Deployment environment recovery failed")
        return payload

    def _normalize_recovered_deployment_environment(self, payload: Any) -> dict[str, object]:
        if hasattr(payload, "__dict__") and not isinstance(payload, dict):
            payload = vars(payload)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Deployment environment recovery failed")
        variables = payload.get("variables")
        variable_names = payload.get("variable_names")
        warnings = payload.get("warnings")
        if not isinstance(variables, dict) or any(not isinstance(name, str) or not isinstance(value, str) for name, value in variables.items()):
            raise HTTPException(status_code=422, detail="Deployment environment recovery failed")
        if not isinstance(variable_names, list) or any(not isinstance(name, str) for name in variable_names):
            raise HTTPException(status_code=422, detail="Deployment environment recovery failed")
        if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
            raise HTTPException(status_code=422, detail="Deployment environment recovery failed")
        sorted_names = sorted(variables)
        if sorted(variable_names) != sorted_names:
            raise HTTPException(status_code=422, detail="Deployment environment recovery failed")
        return {
            "variables": {name: variables[name] for name in sorted_names},
            "variable_names": sorted_names,
            "warnings": list(warnings),
        }

    def _backup_requires_maintenance(self, scope: Any) -> bool:
        return str(scope) in {"full", "files"}

    def _normalize_manifest(self, manifest: Any) -> dict[str, Any]:
        if manifest is None:
            return {}
        if hasattr(manifest, "to_dict"):
            return dict(manifest.to_dict())
        if isinstance(manifest, dict):
            return dict(manifest)
        if hasattr(manifest, "model_dump"):
            return manifest.model_dump()
        if hasattr(manifest, "dict"):
            return manifest.dict()
        return {key: value for key, value in vars(manifest).items() if not key.startswith("_")}

    async def _load_sidecars(self) -> None:
        for sidecar in self._backup_dir.glob("*.json"):
            self._backup_jobs[sidecar.stem] = self._deserialize_backup_job(json.loads(sidecar.read_text()))
        for sidecar in self._upload_dir.glob("*.json"):
            self._uploads[sidecar.stem] = self._deserialize_upload(json.loads(sidecar.read_text()))
        for sidecar in self._restore_dir.glob("*.json"):
            self._restore_jobs[sidecar.stem] = self._deserialize_restore_job(json.loads(sidecar.read_text()))

    def _mark_orphaned_jobs_interrupted(self) -> None:
        for job_id, job in list(self._backup_jobs.items()):
            if job.get("status") in _RECOVERABLE_BACKUP_STATUSES:
                job["status"] = "interrupted"
                job["phase"] = "interrupted"
                job["message"] = "Backup interrupted by server restart"
                job["updated_at"] = _utcnow()
                Path(job["artifact_path"]).unlink(missing_ok=True)
                self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(job))
        for job_id, job in list(self._restore_jobs.items()):
            if job.get("status") in _RECOVERABLE_RESTORE_STATUSES:
                job["status"] = "interrupted"
                job["phase"] = "interrupted"
                job["message"] = "Restore interrupted by server restart"
                job["updated_at"] = _utcnow()
                self._write_json(self._restore_job_sidecar_path(job_id), self._serialize_restore_job(job))

    def _cleanup_expired_state(self) -> None:
        now = _utcnow()
        upload_cutoff = now - timedelta(seconds=self._upload_ttl)
        job_cutoff = now - timedelta(seconds=self._job_ttl)

        for job_id, job in list(self._backup_jobs.items()):
            updated_at = job.get("updated_at") or job.get("created_at") or now
            # Completed and delivered exports persist until explicit deletion.
            if job.get("status") in (_TERMINAL_BACKUP_STATUSES - _PERSISTENT_EXPORT_BACKUP_STATUSES) and updated_at < job_cutoff:
                Path(job["artifact_path"]).unlink(missing_ok=True)
                self._backup_job_sidecar_path(job_id).unlink(missing_ok=True)
                self._backup_jobs.pop(job_id, None)

        for upload_id, upload in list(self._uploads.items()):
            created_at = upload.get("created_at") or now
            if created_at < upload_cutoff:
                Path(upload["stored_path"]).unlink(missing_ok=True)
                self._upload_sidecar_path(upload_id).unlink(missing_ok=True)
                self._uploads.pop(upload_id, None)

        for job_id, job in list(self._restore_jobs.items()):
            updated_at = job.get("updated_at") or job.get("created_at") or now
            if job.get("status") in _TERMINAL_RESTORE_STATUSES and updated_at < job_cutoff:
                self._restore_job_sidecar_path(job_id).unlink(missing_ok=True)
                self._restore_jobs.pop(job_id, None)

    async def _complete_backup_delivery(self, job_id: str) -> None:
        job = self._backup_jobs.get(job_id)
        if job is None:
            return
        job["status"] = "delivered"
        job["phase"] = "delivered"
        job["progress"] = 100
        job["message"] = "Backup delivered"
        delivered_at = job.get("delivered_at") or _utcnow()
        job["delivered_at"] = delivered_at
        job["updated_at"] = _utcnow()
        self._write_json(self._backup_job_sidecar_path(job_id), self._serialize_backup_job(job))

    def _assert_no_active_job_locked(self) -> None:
        self._cleanup_expired_state()
        if any(job.get("status") in _ACTIVE_BACKUP_STATUSES for job in self._backup_jobs.values()):
            raise HTTPException(status_code=409, detail="Another server backup or restore job is already active")
        if any(job.get("status") in _ACTIVE_RESTORE_STATUSES for job in self._restore_jobs.values()):
            raise HTTPException(status_code=409, detail="Another server backup or restore job is already active")

    def _serialize_backup_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            **{k: v for k, v in job.items() if k not in {"created_at", "updated_at", "delivered_at"}},
            "created_at": _isoformat(job.get("created_at")),
            "updated_at": _isoformat(job.get("updated_at")),
            "delivered_at": _isoformat(job.get("delivered_at")),
        }

    def _deserialize_backup_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload)
        record["created_at"] = _parse_datetime(payload.get("created_at"))
        record["updated_at"] = _parse_datetime(payload.get("updated_at"))
        record["delivered_at"] = _parse_datetime(payload.get("delivered_at"))
        return record

    def _serialize_upload(self, upload: dict[str, Any]) -> dict[str, Any]:
        return {
            **{k: v for k, v in upload.items() if k != "created_at"},
            "created_at": _isoformat(upload.get("created_at")),
        }

    def _deserialize_upload(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload)
        record["created_at"] = _parse_datetime(payload.get("created_at"))
        return record

    def _serialize_restore_job(self, job: dict[str, Any]) -> dict[str, Any]:
        filtered = {k: v for k, v in job.items() if k not in {"created_at", "updated_at", "_password", "upload_path"}}
        filtered["created_at"] = _isoformat(job.get("created_at"))
        filtered["updated_at"] = _isoformat(job.get("updated_at"))
        return filtered

    def _deserialize_restore_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload)
        record["created_at"] = _parse_datetime(payload.get("created_at"))
        record["updated_at"] = _parse_datetime(payload.get("updated_at"))
        if "upload_path" not in record:
            upload_id = record.get("upload_id")
            normalized_upload_id = upload_id if isinstance(upload_id, str) else None
            upload = self._uploads.get(normalized_upload_id) if normalized_upload_id is not None else None
            record["upload_path"] = upload.get("stored_path") if upload else None
        return record

    def _snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(payload)
        for hidden in ("artifact_path", "upload_path", "_password", "cancel_requested"):
            snapshot.pop(hidden, None)
        for key in ("created_at", "updated_at", "delivered_at"):
            if isinstance(snapshot.get(key), datetime):
                snapshot[key] = _isoformat(snapshot[key])
        return snapshot

    def _latest_active_job(
        self,
        jobs: dict[str, dict[str, Any]],
        active_statuses: set[str],
    ) -> dict[str, Any] | None:
        active_jobs = [job for job in jobs.values() if job.get("status") in active_statuses]
        if not active_jobs:
            return None
        return max(active_jobs, key=lambda job: job.get("updated_at") or job.get("created_at") or datetime.min.replace(tzinfo=timezone.utc))

    def _job_progress_callback(self, job_id: str, *, backup: bool) -> Callable[[dict[str, Any]], None]:
        loop = asyncio.get_running_loop()
        loop_thread_id = threading.get_ident()

        def _callback(event: dict[str, Any]) -> None:
            if threading.get_ident() == loop_thread_id:
                self._apply_job_progress(job_id, event, backup=backup)
                return
            loop.call_soon_threadsafe(self._apply_job_progress, job_id, event, backup)

        return _callback

    def _apply_job_progress(self, job_id: str, event: dict[str, Any], backup: bool) -> None:
        jobs = self._backup_jobs if backup else self._restore_jobs
        sidecar_path = self._backup_job_sidecar_path(job_id) if backup else self._restore_job_sidecar_path(job_id)
        job = jobs.get(job_id)
        if job is None:
            return
        current_progress = int(job.get("progress") or 0)
        next_progress = event.get("progress")
        if isinstance(next_progress, (int, float)) and int(next_progress) < current_progress:
            return
        phase = event.get("phase")
        if isinstance(phase, str) and phase:
            job["phase"] = phase
        if isinstance(next_progress, (int, float)):
            job["progress"] = int(next_progress)
        message = event.get("message")
        if isinstance(message, str) and message:
            job["message"] = message
        details = self._sanitize_progress_details(event)
        if details:
            job["details"] = details
        else:
            job.pop("details", None)
        job["updated_at"] = _utcnow()
        self._write_json(sidecar_path, self._serialize_backup_job(job) if backup else self._serialize_restore_job(job))

    def _sanitize_progress_details(self, event: dict[str, Any]) -> dict[str, Any]:
        details = {key: event[key] for key in _PROGRESS_DETAIL_ALLOWLIST if key in event}
        current_item = details.get("current_item")
        if not isinstance(current_item, str) or self._is_absolute_progress_item(current_item):
            details.pop("current_item", None)
        return details

    def _is_absolute_progress_item(self, value: str) -> bool:
        return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()

    def _snapshot_upload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(payload)
        snapshot.pop("stored_path", None)
        if isinstance(snapshot.get("created_at"), datetime):
            snapshot["created_at"] = _isoformat(snapshot["created_at"])
        return snapshot

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, sort_keys=True))
        with suppress(OSError):
            temp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        temp_path.replace(path)
        with suppress(OSError):
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def _ensure_free_space(self, announced_size: int) -> None:
        usage = shutil.disk_usage(self._root_dir)
        free_bytes = usage.free if hasattr(usage, "free") else usage[2]
        remaining_after_write = free_bytes - announced_size
        if remaining_after_write < self._min_free_bytes:
            raise HTTPException(status_code=507, detail="Not enough free space to stage the restore upload")

    def _extract_content_length(self, file: UploadFile) -> int | None:
        headers = getattr(file, "headers", None)
        if headers is None:
            return None
        raw_value = headers.get("content-length")
        if raw_value is None:
            return None
        try:
            return max(int(raw_value), 0)
        except ValueError:
            return None

    def _password_fd(self, password: str | None):
        outer_service = self

        class _PasswordFD:
            def __enter__(self) -> int | None:
                if password is None:
                    self.handle = None
                    self.dup_fd = None
                    return None
                encoded = password.encode("utf-8")
                if len(encoded) > outer_service._max_password_bytes:
                    raise HTTPException(status_code=400, detail="Password is too long")
                handle = TemporaryFile()
                handle.write(encoded)
                handle.seek(0)
                self.handle = handle
                self.dup_fd = os.dup(handle.fileno())
                return self.dup_fd

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                dup_fd = getattr(self, "dup_fd", None)
                if dup_fd is not None:
                    with suppress(OSError):
                        os.close(dup_fd)
                handle = getattr(self, "handle", None)
                if handle is not None:
                    handle.close()

        return _PasswordFD()

    def _runtime_lease_id(self, job_id: str, *, operation_kind: str = "restore") -> str:
        return f"server-{operation_kind}-{job_id}"

    def _backup_job_sidecar_path(self, job_id: str) -> Path:
        return self._backup_dir / f"{job_id}.json"

    def _restore_job_sidecar_path(self, job_id: str) -> Path:
        return self._restore_dir / f"{job_id}.json"

    def _upload_sidecar_path(self, upload_id: str) -> Path:
        return self._upload_dir / f"{upload_id}.json"


server_backup_service = ServerBackupService()
