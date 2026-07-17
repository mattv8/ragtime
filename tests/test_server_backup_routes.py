import asyncio
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from starlette.datastructures import Headers

from ragtime.core.server_backup import BackupManifest, BackupOptions, BackupScope, RestoreOptions, _confirmation_phrase
from ragtime.indexer import server_backup_routes
from ragtime.indexer.server_backup_service import ServerBackupService
from ragtime.main import _build_server_restart_signaler
from ragtime.main import app as main_app


def _admin() -> SimpleNamespace:
    return SimpleNamespace(id="admin-1", role="admin")


def _manifest(*, scope: str = "full", encrypted: bool = True, legacy: bool = False) -> dict[str, object]:
    return {
        "format": "ragtime-server-backup",
        "version": 1,
        "created_at": "2026-07-16T12:00:00Z",
        "scope": scope,
        "ragtime_version": "test",
        "schema_version": "test",
        "encrypted": encrypted,
        "includes_managed_key": encrypted,
        "requires_legacy_key_acknowledgement": legacy,
    }


def _upload_file(filename: str, chunks: list[bytes], *, content_length: int | None = None) -> UploadFile:
    header_items: dict[str, str] = {}
    if content_length is not None:
        header_items["content-length"] = str(content_length)
    return UploadFile(
        file=io.BytesIO(b"".join(chunks)),
        filename=filename,
        headers=Headers(header_items),
    )


class ServerBackupRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_require_admin_auth(self) -> None:
        app = FastAPI()
        app.include_router(server_backup_routes.router)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
            response = await client.post("/indexes/server-backups/jobs", json={"scope": "files", "encrypt": False})
        self.assertEqual(response.status_code, 401)

    async def test_backup_routes_delegate_to_service(self) -> None:
        app = FastAPI()
        app.include_router(server_backup_routes.router)
        app.dependency_overrides[server_backup_routes.require_admin] = lambda: _admin()
        service = mock.AsyncMock()
        service.create_backup_job.return_value = {"id": "job-1", "status": "pending"}
        service.get_backup_job.return_value = {"id": "job-1", "status": "running"}
        service.cancel_backup_job.return_value = {"id": "job-1", "status": "cancelled"}
        service.get_backup_download_response.return_value = SimpleNamespace(status_code=200)

        try:
            with mock.patch.object(server_backup_routes, "server_backup_service", service):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                    response = await client.post(
                        "/indexes/server-backups/jobs",
                        json={"scope": "files", "encrypt": True, "password": "secret"},
                    )
                    self.assertEqual(response.status_code, 202)
                    self.assertEqual(response.json()["id"], "job-1")

                    response = await client.get("/indexes/server-backups/jobs/job-1")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["status"], "running")

                    response = await client.post("/indexes/server-backups/jobs/job-1/cancel")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["status"], "cancelled")

            service.create_backup_job.assert_awaited_once()
            create_call = service.create_backup_job.await_args
            self.assertEqual(create_call.kwargs["scope"], "files")
            self.assertTrue(create_call.kwargs["encrypt"])
            self.assertEqual(create_call.kwargs["password"], "secret")
            service.get_backup_job.assert_awaited_once_with("job-1")
            service.cancel_backup_job.assert_awaited_once_with("job-1")
        finally:
            app.dependency_overrides.clear()

    async def test_restore_routes_delegate_to_service(self) -> None:
        app = FastAPI()
        app.include_router(server_backup_routes.router)
        app.dependency_overrides[server_backup_routes.require_admin] = lambda: _admin()
        service = mock.AsyncMock()
        service.stage_restore_upload.return_value = {"upload_id": "upload-1", "filename": "backup.ragbak", "size_bytes": 12}
        service.create_restore_job.return_value = {"id": "restore-1", "status": "validating"}
        service.get_restore_job.return_value = {"id": "restore-1", "status": "ready_for_commit"}
        service.commit_restore_job.return_value = {"id": "restore-1", "status": "restoring"}

        try:
            with mock.patch.object(server_backup_routes, "server_backup_service", service):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                    response = await client.post(
                        "/indexes/server-backups/uploads",
                        files={"file": ("backup.ragbak", b"archive-bytes", "application/octet-stream")},
                    )
                    self.assertEqual(response.status_code, 202)
                    self.assertEqual(response.json()["upload_id"], "upload-1")

                    response = await client.post(
                        "/indexes/server-backups/restore-jobs",
                        json={
                            "upload_id": "upload-1",
                            "password": "secret",
                            "scope_override": "database",
                            "replace_data": True,
                            "mirror_local_admin_access": True,
                            "mirror_local_admin_from": "backup",
                            "local_admin_username": "local:admin",
                        },
                    )
                    self.assertEqual(response.status_code, 202)
                    self.assertEqual(response.json()["id"], "restore-1")

                    response = await client.get("/indexes/server-backups/restore-jobs/restore-1")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["status"], "ready_for_commit")

                    response = await client.post(
                        "/indexes/server-backups/restore-jobs/restore-1/commit",
                        json={"confirmation_text": "RESTORE restore-1", "acknowledge_legacy_key": True},
                    )
                    self.assertEqual(response.status_code, 202)
                    self.assertEqual(response.json()["status"], "restoring")

            service.stage_restore_upload.assert_awaited_once()
            service.create_restore_job.assert_awaited_once()
            create_call = service.create_restore_job.await_args
            self.assertEqual(create_call.kwargs["upload_id"], "upload-1")
            self.assertEqual(create_call.kwargs["password"], "secret")
            self.assertEqual(create_call.kwargs["scope_override"], "database")
            self.assertTrue(create_call.kwargs["replace_data"])
            self.assertTrue(create_call.kwargs["mirror_local_admin_access"])
            self.assertEqual(create_call.kwargs["mirror_local_admin_from"], "backup")
            self.assertEqual(create_call.kwargs["local_admin_username"], "local:admin")
            service.commit_restore_job.assert_awaited_once_with(
                "restore-1",
                confirmation_text="RESTORE restore-1",
                acknowledge_legacy_key=True,
            )
        finally:
            app.dependency_overrides.clear()


class ServerBackupServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_option_builders_use_task1_dataclasses_and_exact_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            with mock.patch.dict(os.environ, {"POSTGRES_DB": "ragtime"}, clear=False):
                backup_options = service._build_backup_options(
                    BackupOptions,
                    BackupScope.FILES,
                    Path(temp_dir) / "backup.ragbak",
                    True,
                    7,
                )
                restore_options = service._build_restore_options(
                    RestoreOptions,
                    Path(temp_dir) / "backup.ragbak",
                    8,
                    {
                        "scope_override": BackupScope.DATABASE,
                        "skip_migrations": True,
                        "pg_data_only": True,
                        "replace_data": True,
                        "acknowledge_legacy_key": True,
                        "mirror_local_admin_access": True,
                        "mirror_local_admin_from": "backup",
                        "local_admin_username": "local:admin",
                    },
                )

            self.assertEqual(
                backup_options,
                BackupOptions(
                    scope=BackupScope.FILES,
                    output_path=Path(temp_dir) / "backup.ragbak",
                    encrypt=True,
                    password_fd=7,
                ),
            )
            self.assertEqual(
                restore_options,
                RestoreOptions(
                    archive_path=Path(temp_dir) / "backup.ragbak",
                    scope_override=BackupScope.DATABASE,
                    skip_migrations=True,
                    pg_data_only=True,
                    replace_data=True,
                    acknowledge_legacy_key=True,
                    restore_confirmation=_confirmation_phrase(),
                    mirror_local_admin_access=True,
                    mirror_local_admin_from="backup",
                    local_admin_username="local:admin",
                    password_fd=8,
                ),
            )

    async def test_manifest_normalization_uses_to_dict_and_legacy_embedded_key(self) -> None:
        service = ServerBackupService(root_dir=Path(tempfile.mkdtemp()))
        manifest = BackupManifest(
            format="ragtime-server-backup",
            version=1,
            created_at="2026-07-16T12:00:00Z",
            scope=BackupScope.FULL,
            ragtime_version="test",
            schema_version="schema",
            encrypted=True,
            includes_managed_key=True,
            legacy_embedded_key=True,
        )

        normalized = service._normalize_manifest(manifest)

        self.assertEqual(normalized["scope"], "full")
        self.assertTrue(normalized["legacy_embedded_key"])
        self.assertNotIn("requires_legacy_key_acknowledgement", normalized)

    async def test_restore_validation_sidecar_excludes_password_and_commit_signals_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            await service.startup()
            signaled_job_ids: list[str] = []
            service.set_restart_signaler(lambda job_id: signaled_job_ids.append(job_id))

            upload = await service.stage_restore_upload(
                _upload_file("backup.ragbak", [b"hello", b"world"]),
                filename="backup.ragbak",
            )
            release_events: list[str] = []

            async def _release_runtime(lease_id: str) -> None:
                release_events.append(f"runtime:{lease_id}")

            async def _release_maintenance(lease_id: str) -> None:
                release_events.append(f"maintenance:{lease_id}")

            with (
                mock.patch.object(service, "_inspect_backup_archive", mock.AsyncMock(return_value=_manifest(legacy=True))) as inspect_backup_archive,
                mock.patch.object(service, "_acquire_maintenance_lease", mock.AsyncMock(return_value="maintenance-lease")) as acquire_maintenance_lease,
                mock.patch.object(service, "_release_maintenance_lease", mock.AsyncMock(side_effect=_release_maintenance)),
                mock.patch.object(service, "_acquire_runtime_lease", mock.AsyncMock(return_value="runtime-lease")) as acquire_runtime_lease,
                mock.patch.object(service, "_release_runtime_lease", mock.AsyncMock(side_effect=_release_runtime)),
                mock.patch.object(service, "_restore_backup_archive", mock.AsyncMock(return_value=_manifest(legacy=True))),
            ):
                restore_job = await service.create_restore_job(upload_id=upload["upload_id"], password="super-secret")
                await service._restore_tasks[restore_job["id"]]

                sidecar_path = service._restore_job_sidecar_path(restore_job["id"])
                sidecar_payload = json.loads(sidecar_path.read_text())
                self.assertNotIn("password", json.dumps(sidecar_payload))

                latest = await service.get_restore_job(restore_job["id"])
                self.assertEqual(latest["status"], "ready_for_commit")

                committed = await service.commit_restore_job(
                    restore_job["id"],
                    confirmation_text=service._restore_jobs[restore_job["id"]]["required_confirmation"],
                    acknowledge_legacy_key=True,
                )
                await service._restore_tasks[committed["id"]]

                latest = await service.get_restore_job(restore_job["id"])
                self.assertEqual(latest["status"], "completed")
                self.assertTrue(latest["restart_required"])
                self.assertEqual(signaled_job_ids, [restore_job["id"]])
                acquire_maintenance_lease.assert_awaited_once()
                acquire_runtime_lease.assert_awaited_once()
                inspect_backup_archive.assert_awaited_once()
                self.assertEqual(release_events, ["runtime:runtime-lease", "maintenance:maintenance-lease"])
                self.assertNotIn("upload_path", latest)

            await service.shutdown()

    async def test_restart_signal_happens_after_persistence_and_lease_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            await service.startup()
            upload = await service.stage_restore_upload(
                _upload_file("backup.ragbak", [b"payload"]),
                filename="backup.ragbak",
            )
            order: list[str] = []

            async def _release_runtime(lease_id: str) -> None:
                order.append(f"release-runtime:{lease_id}")

            async def _release_maintenance(lease_id: str) -> None:
                order.append(f"release-maintenance:{lease_id}")

            def _restart(job_id: str) -> None:
                sidecar = json.loads(service._restore_job_sidecar_path(job_id).read_text())
                self.assertEqual(sidecar["status"], "completed")
                self.assertEqual(sidecar["message"], "Restore completed; restart requested")
                self.assertIn("release-runtime:runtime-lease", order)
                self.assertIn("release-maintenance:maintenance-lease", order)
                order.append(f"restart:{job_id}")

            original_write_json = service._write_json

            def _recording_write_json(path: Path, payload: dict[str, object]) -> None:
                original_write_json(path, payload)
                if path == service._restore_job_sidecar_path(restore_job["id"]) and payload.get("status") == "completed":
                    order.append("persist-completed")

            service.set_restart_signaler(_restart)
            with (
                mock.patch.object(service, "_inspect_backup_archive", mock.AsyncMock(return_value=_manifest())),
                mock.patch.object(service, "_acquire_maintenance_lease", mock.AsyncMock(return_value="maintenance-lease")),
                mock.patch.object(service, "_acquire_runtime_lease", mock.AsyncMock(return_value="runtime-lease")),
                mock.patch.object(service, "_restore_backup_archive", mock.AsyncMock(return_value=_manifest())),
                mock.patch.object(service, "_release_runtime_lease", mock.AsyncMock(side_effect=_release_runtime)),
                mock.patch.object(service, "_release_maintenance_lease", mock.AsyncMock(side_effect=_release_maintenance)),
            ):
                restore_job = await service.create_restore_job(upload_id=upload["upload_id"], password="secret")
                with mock.patch.object(service, "_write_json", side_effect=_recording_write_json):
                    await service._restore_tasks[restore_job["id"]]
                    await service.commit_restore_job(
                        restore_job["id"],
                        confirmation_text=service._restore_jobs[restore_job["id"]]["required_confirmation"],
                        acknowledge_legacy_key=False,
                    )
                    await service._restore_tasks[restore_job["id"]]

                self.assertEqual(
                    order,
                    [
                        "persist-completed",
                        "release-runtime:runtime-lease",
                        "release-maintenance:maintenance-lease",
                        "persist-completed",
                        f"restart:{restore_job['id']}",
                    ],
                )

            await service.shutdown()

    async def test_concurrent_backup_start_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            await service.startup()
            started = asyncio.Event()
            release = asyncio.Event()

            async def _create_artifact(*, output_path: Path, **_: object) -> dict[str, object]:
                started.set()
                await release.wait()
                output_path.write_bytes(b"backup")
                return _manifest(scope="files")

            with mock.patch.object(service, "_create_backup_archive", mock.AsyncMock(side_effect=_create_artifact)):
                first = await service.create_backup_job(scope="files", encrypt=False, password=None)
                await started.wait()

                with self.assertRaises(HTTPException) as blocked:
                    await service.create_backup_job(scope="database", encrypt=False, password=None)

                self.assertEqual(blocked.exception.status_code, 409)
                release.set()
                await service._backup_tasks[first["id"]]

            await service.shutdown()

    async def test_concurrent_restore_validation_and_commit_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            await service.startup()
            upload = await service.stage_restore_upload(_upload_file("backup.ragbak", [b"payload"]), filename="backup.ragbak")
            validation_started = asyncio.Event()
            validation_release = asyncio.Event()
            restore_release = asyncio.Event()

            async def _inspect_backup_archive(_path: Path, _password: str | None) -> dict[str, object]:
                validation_started.set()
                await validation_release.wait()
                return _manifest()

            async def _restore_backup_archive(_path: Path, *, password: str | None, options: dict[str, object]) -> dict[str, object]:
                await restore_release.wait()
                return _manifest()

            with (
                mock.patch.object(service, "_inspect_backup_archive", mock.AsyncMock(side_effect=_inspect_backup_archive)),
                mock.patch.object(service, "_restore_backup_archive", mock.AsyncMock(side_effect=_restore_backup_archive)),
                mock.patch.object(service, "_acquire_maintenance_lease", mock.AsyncMock(return_value="maintenance-lease")),
                mock.patch.object(service, "_acquire_runtime_lease", mock.AsyncMock(return_value="runtime-lease")),
                mock.patch.object(service, "_release_maintenance_lease", mock.AsyncMock()),
                mock.patch.object(service, "_release_runtime_lease", mock.AsyncMock()),
            ):
                first = await service.create_restore_job(upload_id=upload["upload_id"], password="secret")
                await validation_started.wait()

                with self.assertRaises(HTTPException) as blocked_validation:
                    await service.create_restore_job(upload_id=upload["upload_id"], password="secret")
                self.assertEqual(blocked_validation.exception.status_code, 409)

                validation_release.set()
                await service._restore_tasks[first["id"]]

                await service.commit_restore_job(
                    first["id"],
                    confirmation_text=service._restore_jobs[first["id"]]["required_confirmation"],
                    acknowledge_legacy_key=False,
                )
                with self.assertRaises(HTTPException) as blocked_commit:
                    await service.commit_restore_job(
                        first["id"],
                        confirmation_text=service._restore_jobs[first["id"]]["required_confirmation"],
                        acknowledge_legacy_key=False,
                    )
                self.assertEqual(blocked_commit.exception.status_code, 409)

                restore_release.set()
                await service._restore_tasks[first["id"]]

            await service.shutdown()

    async def test_password_fd_context_closes_duplicated_fd(self) -> None:
        service = ServerBackupService(root_dir=Path(tempfile.mkdtemp()))
        with service._password_fd("secret") as password_fd:
            self.assertIsInstance(password_fd, int)
            self.assertGreaterEqual(os.read(cast(int, password_fd), 6), b"secret")
        with self.assertRaises(OSError):
            os.close(cast(int, password_fd))

    async def test_runtime_lease_failed_acquire_still_releases_owned_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            await service.startup()
            upload = await service.stage_restore_upload(_upload_file("backup.ragbak", [b"payload"]), filename="backup.ragbak")
            release_events: list[str] = []

            async def _release_runtime(lease_id: str) -> None:
                release_events.append(lease_id)

            with (
                mock.patch.object(service, "_inspect_backup_archive", mock.AsyncMock(return_value=_manifest())),
                mock.patch.object(service, "_acquire_maintenance_lease", mock.AsyncMock(return_value="maintenance-lease")),
                mock.patch.object(service, "_release_maintenance_lease", mock.AsyncMock()),
                mock.patch.object(service, "_acquire_runtime_lease", mock.AsyncMock(side_effect=HTTPException(status_code=409, detail="not drained"))),
                mock.patch.object(service, "_release_runtime_lease", mock.AsyncMock(side_effect=_release_runtime)),
                mock.patch.object(service, "_restore_backup_archive", mock.AsyncMock()),
            ):
                restore_job = await service.create_restore_job(upload_id=upload["upload_id"], password="secret")
                await service._restore_tasks[restore_job["id"]]
                await service.commit_restore_job(
                    restore_job["id"],
                    confirmation_text=service._restore_jobs[restore_job["id"]]["required_confirmation"],
                    acknowledge_legacy_key=False,
                )
                await service._restore_tasks[restore_job["id"]]

            self.assertEqual(release_events, [f"server-restore-{restore_job['id']}"])
            latest = await service.get_restore_job(restore_job["id"])
            self.assertEqual(latest["status"], "failed")
            await service.shutdown()

    async def test_restore_heartbeat_renews_multiple_times_during_long_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            await service.startup()
            upload = await service.stage_restore_upload(_upload_file("backup.ragbak", [b"payload"]), filename="backup.ragbak")
            maintenance_renewals: list[str] = []
            runtime_renewals: list[str] = []
            allow_restore = asyncio.Event()

            async def _renew_local(lease_id: str) -> None:
                maintenance_renewals.append(lease_id)

            async def _renew_runtime(lease_id: str) -> None:
                runtime_renewals.append(lease_id)
                if len(runtime_renewals) >= 2:
                    allow_restore.set()

            async def _restore_backup_archive(_path: Path, *, password: str | None, options: dict[str, object]) -> dict[str, object]:
                await allow_restore.wait()
                return _manifest()

            with (
                mock.patch.object(service, "_inspect_backup_archive", mock.AsyncMock(return_value=_manifest())),
                mock.patch.object(service, "_acquire_maintenance_lease", mock.AsyncMock(return_value="maintenance-lease")),
                mock.patch.object(service, "_acquire_runtime_lease", mock.AsyncMock(return_value="runtime-lease")),
                mock.patch.object(service, "_release_maintenance_lease", mock.AsyncMock()),
                mock.patch.object(service, "_release_runtime_lease", mock.AsyncMock()),
                mock.patch.object(service, "_renew_process_maintenance_lease", mock.AsyncMock(side_effect=_renew_local)),
                mock.patch.object(service, "_renew_runtime_maintenance_lease", mock.AsyncMock(side_effect=_renew_runtime)),
                mock.patch.object(service, "_restore_backup_archive", mock.AsyncMock(side_effect=_restore_backup_archive)),
            ):
                restore_job = await service.create_restore_job(upload_id=upload["upload_id"], password="secret")
                await service._restore_tasks[restore_job["id"]]
                service._maintenance_heartbeat_interval_seconds = 0.01
                await service.commit_restore_job(
                    restore_job["id"],
                    confirmation_text=service._restore_jobs[restore_job["id"]]["required_confirmation"],
                    acknowledge_legacy_key=False,
                )
                await service._restore_tasks[restore_job["id"]]

            self.assertGreaterEqual(len(maintenance_renewals), 2)
            self.assertGreaterEqual(len(runtime_renewals), 2)
            await service.shutdown()

    async def test_runtime_lease_renew_sends_valid_json_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            captured: list[dict[str, object]] = []

            class _FakeClient:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    self.kwargs = kwargs

                async def __aenter__(self) -> "_FakeClient":
                    return self

                async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
                    return None

                async def put(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> SimpleNamespace:
                    captured.append({"url": url, "headers": headers, "json": json})
                    return SimpleNamespace(status_code=200)

            with (
                mock.patch("ragtime.indexer.server_backup_service.settings.userspace_runtime_auth_token", "runtime-token"),
                mock.patch("ragtime.indexer.server_backup_service.settings.userspace_runtime_manager_url", "http://runtime:8090"),
                mock.patch("ragtime.indexer.server_backup_service.httpx.AsyncClient", _FakeClient),
            ):
                await service._renew_runtime_maintenance_lease("lease-123")

            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["url"], "http://runtime:8090/maintenance/lease/lease-123")
            self.assertEqual(captured[0]["headers"], {"Authorization": "Bearer runtime-token"})
            self.assertEqual(captured[0]["json"], {})

    async def test_restore_failure_releases_leases_and_cancellation_releases_if_acquired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            await service.startup()
            release_events: list[str] = []
            upload = await service.stage_restore_upload(_upload_file("backup.ragbak", [b"payload"]), filename="backup.ragbak")

            with (
                mock.patch.object(service, "_inspect_backup_archive", mock.AsyncMock(return_value=_manifest())),
                mock.patch.object(service, "_acquire_maintenance_lease", mock.AsyncMock(return_value="maintenance-lease")),
                mock.patch.object(service, "_acquire_runtime_lease", mock.AsyncMock(return_value="runtime-lease")),
                mock.patch.object(service, "_release_runtime_lease", mock.AsyncMock(side_effect=lambda lease_id: release_events.append(f"runtime:{lease_id}"))),
                mock.patch.object(
                    service, "_release_maintenance_lease", mock.AsyncMock(side_effect=lambda lease_id: release_events.append(f"maintenance:{lease_id}"))
                ),
                mock.patch.object(service, "_restore_backup_archive", mock.AsyncMock(side_effect=RuntimeError("boom"))),
            ):
                restore_job = await service.create_restore_job(upload_id=upload["upload_id"], password="secret")
                await service._restore_tasks[restore_job["id"]]
                await service.commit_restore_job(
                    restore_job["id"],
                    confirmation_text=service._restore_jobs[restore_job["id"]]["required_confirmation"],
                    acknowledge_legacy_key=False,
                )
                await service._restore_tasks[restore_job["id"]]

                latest = await service.get_restore_job(restore_job["id"])
                self.assertEqual(latest["status"], "failed")
                self.assertEqual(release_events, ["runtime:runtime-lease", "maintenance:maintenance-lease"])

            await service.shutdown()

    async def test_active_job_conflicts_cleanup_ttl_and_orphan_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stale_service = ServerBackupService(root_dir=root)
            await stale_service.startup()
            stale_backup_id = "stale-backup"
            stale_restore_id = "stale-restore"
            stale_upload_id = "stale-upload"
            old_time = datetime.now(timezone.utc) - timedelta(days=2)

            stale_artifact = root / "backup_jobs" / f"{stale_backup_id}.ragbak"
            stale_artifact.write_bytes(b"backup")
            stale_service._write_json(
                stale_service._backup_job_sidecar_path(stale_backup_id),
                {
                    "id": stale_backup_id,
                    "status": "completed",
                    "progress": None,
                    "message": None,
                    "scope": "full",
                    "encrypt": True,
                    "artifact_path": str(stale_artifact),
                    "download_name": "backup.ragbak",
                    "manifest": _manifest(),
                    "error": None,
                    "delivered_at": None,
                    "cancel_requested": False,
                    "created_at": old_time.isoformat().replace("+00:00", "Z"),
                    "updated_at": old_time.isoformat().replace("+00:00", "Z"),
                },
            )
            stale_upload = root / "restore_uploads" / f"{stale_upload_id}-backup.ragbak"
            stale_upload.write_bytes(b"upload")
            stale_service._write_json(
                stale_service._upload_sidecar_path(stale_upload_id),
                {
                    "upload_id": stale_upload_id,
                    "filename": "backup.ragbak",
                    "stored_path": str(stale_upload),
                    "size_bytes": 6,
                    "created_at": old_time.isoformat().replace("+00:00", "Z"),
                },
            )
            stale_service._write_json(
                stale_service._restore_job_sidecar_path(stale_restore_id),
                {
                    "id": stale_restore_id,
                    "status": "restoring",
                    "progress": 55,
                    "message": "mid-flight",
                    "created_at": old_time.isoformat().replace("+00:00", "Z"),
                    "updated_at": old_time.isoformat().replace("+00:00", "Z"),
                    "upload_id": stale_upload_id,
                    "upload_filename": "backup.ragbak",
                    "upload_path": str(stale_upload),
                    "manifest": None,
                    "error": None,
                    "required_confirmation": "RESTORE RAGTIME",
                    "restart_required": False,
                    "requires_legacy_key_acknowledgement": False,
                    "scope_override": None,
                    "skip_migrations": False,
                    "pg_data_only": False,
                    "replace_data": False,
                    "mirror_local_admin_access": False,
                    "mirror_local_admin_from": "auto",
                    "local_admin_username": None,
                    "acknowledge_legacy_key": False,
                },
            )
            await stale_service.shutdown()

            service = ServerBackupService(root_dir=root, artifact_ttl_seconds=3600, upload_ttl_seconds=3600, job_ttl_seconds=3600)
            await service.startup()

            interrupted = await service.get_restore_job(stale_restore_id)
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertIn("restart", interrupted["message"])
            self.assertFalse(stale_artifact.exists())
            self.assertFalse(stale_upload.exists())

            blocker = asyncio.Event()

            async def _create_artifact(*, output_path: Path, **_: object) -> dict[str, object]:
                await blocker.wait()
                output_path.write_bytes(b"backup")
                return _manifest(scope="files")

            with mock.patch.object(service, "_create_backup_archive", mock.AsyncMock(side_effect=_create_artifact)):
                first_job = await service.create_backup_job(scope="files", encrypt=False, password=None)
                with self.assertRaises(HTTPException) as backup_conflict:
                    await service.create_backup_job(scope="full", encrypt=False, password=None)
                self.assertEqual(backup_conflict.exception.status_code, 409)
                with self.assertRaises(HTTPException) as restore_conflict:
                    await service.create_restore_job(upload_id="missing", password=None)
                self.assertEqual(restore_conflict.exception.status_code, 409)
                cancelled = await service.cancel_backup_job(first_job["id"])
                self.assertEqual(cancelled["status"], "cancelled")
                blocker.set()
                with self.assertRaises(asyncio.CancelledError):
                    await service._backup_tasks[first_job["id"]]

            await service.shutdown()

    async def test_upload_checks_content_length_free_space_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(
                root_dir=Path(temp_dir),
                upload_limit_bytes=4,
                min_free_bytes=1,
            )
            await service.startup()

            with mock.patch("shutil.disk_usage", return_value=(100, 10, 0)):
                with self.assertRaises(HTTPException) as no_space:
                    await service.stage_restore_upload(_upload_file("backup.ragbak", [b"ab"]), filename="backup.ragbak")
            self.assertEqual(no_space.exception.status_code, 507)

            with self.assertRaises(HTTPException) as too_large:
                await service.stage_restore_upload(
                    _upload_file("backup.ragbak", [b"ab"], content_length=10),
                    filename="backup.ragbak",
                )
            self.assertEqual(too_large.exception.status_code, 413)

            upload = await service.stage_restore_upload(_upload_file("backup.ragbak", [b"ab"]), filename="backup.ragbak")
            Path(service._uploads[upload["upload_id"]]["stored_path"]).unlink()

            with self.assertRaises(HTTPException) as missing_upload:
                await service.create_restore_job(upload_id=upload["upload_id"], password=None)
            self.assertEqual(missing_upload.exception.status_code, 404)

            await service.shutdown()

    async def test_download_cleanup_marks_job_delivered_and_deletes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ServerBackupService(root_dir=Path(temp_dir))
            await service.startup()

            async def _create_artifact(*, output_path: Path, **_: object) -> dict[str, object]:
                output_path.write_bytes(b"backup")
                return _manifest(scope="files")

            with mock.patch.object(service, "_create_backup_archive", mock.AsyncMock(side_effect=_create_artifact)):
                job = await service.create_backup_job(scope="files", encrypt=False, password=None)
                await service._backup_tasks[job["id"]]
                response = await service.get_backup_download_response(job["id"])
                artifact_path = Path(service._backup_jobs[job["id"]]["artifact_path"])
                self.assertTrue(str(artifact_path).endswith(".tar.gz"))
                self.assertTrue(artifact_path.exists())

                if response.background is not None:
                    await response.background()

                latest = await service.get_backup_job(job["id"])
                self.assertEqual(latest["status"], "delivered")
                self.assertFalse(artifact_path.exists())
                self.assertNotIn("artifact_path", latest)

            await service.shutdown()


class MainRegistrationTests(unittest.TestCase):
    def test_main_registers_server_backup_routes(self) -> None:
        self.assertIn("/indexes/server-backups/jobs", main_app.openapi()["paths"])

    def test_restart_signaler_records_job_and_schedules_pid1_sigterm(self) -> None:
        app = FastAPI()
        scheduled: list[tuple[Callable[..., None], tuple[object, ...]]] = []
        signaled: list[tuple[int, int]] = []

        def _schedule(callback: Callable[..., None], *args: object) -> None:
            scheduled.append((callback, args))

        def _kill(pid: int, sig: int) -> None:
            signaled.append((pid, sig))

        signaler = _build_server_restart_signaler(app, scheduler=_schedule, kill_fn=_kill)
        signaler("job-123")

        self.assertEqual(app.state.server_backup_restart_requested, "job-123")
        self.assertEqual(len(scheduled), 1)
        callback, args = scheduled[0]
        callback(*args)
        self.assertEqual(len(signaled), 1)
