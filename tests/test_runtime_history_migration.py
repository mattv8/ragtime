from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi import HTTPException

from runtime.worker.sqlite_history.coordinator import SqliteHistoryCoordinator
from runtime.worker.sqlite_history.migration import LegacyHistoryMigration, _digest
from runtime.worker.sqlite_history.models import RESTIC_IMAGE_PATH, ResticArtifact
from runtime.worker.sqlite_history.repository import ResticRepository
from runtime.worker.sqlite_history.service import RuntimeSqliteHistoryService


class _Repository:
    def __init__(self, root: Path) -> None:
        self.root = root

    async def ingest(self, image: Path, **kwargs: object) -> ResticArtifact:
        destination = self.root / "image.sqlite3"
        shutil.copyfile(image, destination)
        return ResticArtifact("a" * 64, "b" * 64, RESTIC_IMAGE_PATH, destination.stat().st_size, hashlib.sha256(destination.read_bytes()).hexdigest())

    async def materialize(self, artifact: ResticArtifact, destination: Path, **kwargs: object) -> None:
        shutil.copyfile(self.root / "image.sqlite3", destination)


class RuntimeHistoryMigrationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _snapshot_ids(repository: ResticRepository) -> list[str]:
        def list_snapshots(cancelled, pass_fds):
            return json.loads(repository._run("snapshots", "--json", cancelled=cancelled, pass_fds=pass_fds))

        return [entry["id"] for entry in await repository._blocking(list_snapshots)]

    async def test_legacy_import_switches_only_after_verified_materialization(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            files = root / "workspaces" / "workspace" / "files"
            (files / ".ragtime/db").mkdir(parents=True)
            service = RuntimeSqliteHistoryService(root, object(), _Repository(root / "repo"))
            repo = root / "repo"
            repo.mkdir()
            history = files.parent / "sqlite_backups"
            (history / "blobs").mkdir(parents=True)
            image = history / "blobs/image.sqlite3"
            image.write_bytes(b"immutable-image")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            service._save(
                history,
                {
                    "version": 1,
                    "workspace_id": "workspace",
                    "backups": [
                        {
                            "id": "one",
                            "workspace_id": "workspace",
                            "database_name": "app.sqlite3",
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "trigger": "manual",
                            "status": "ready",
                            "size_bytes": image.stat().st_size,
                            "sha256": digest,
                            "blob": "blobs/image.sqlite3",
                        },
                        {
                            "id": "two",
                            "workspace_id": "workspace",
                            "database_name": "app.sqlite3",
                            "created_at": "2026-01-02T00:00:00+00:00",
                            "trigger": "snapshot",
                            "status": "ready",
                            "size_bytes": image.stat().st_size,
                            "sha256": digest,
                            "blob": "blobs/image.sqlite3",
                        },
                    ],
                    "previews": {},
                    "operations": {},
                },
            )
            self.assertEqual(2, await service.migrate_legacy_backups("workspace"))
            rows = await service.list_backups("workspace")
            self.assertTrue(all(row["storage"]["kind"] == "restic" for row in rows))
            self.assertFalse(image.exists())

    async def test_http_failure_is_durably_terminal_and_replay_observes_it(self) -> None:
        class FailingHistory:
            async def migrate_legacy_backups(self, *_args: object, **_kwargs: object) -> int:
                raise HTTPException(status_code=409, detail="corrupt legacy catalog")

        from uuid import uuid4

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            coordinator = SqliteHistoryCoordinator(root, object(), history_factory=FailingHistory)
            coordinator.activate()
            migration = LegacyHistoryMigration(coordinator)
            operation_id = str(uuid4())
            with self.assertRaises(HTTPException) as failure:
                await migration.accept("workspace", {"operation_id": operation_id, "user_id": "operator"})
            self.assertEqual(409, failure.exception.status_code)
            terminal = await migration.get("workspace", operation_id)
            self.assertEqual("failed", terminal["phase"])
            self.assertEqual("corrupt legacy catalog", terminal["error"])
            replay = await migration.accept("workspace", {"operation_id": operation_id, "user_id": "operator"})
            self.assertEqual("failed", replay["phase"])

    async def test_cancel_is_observed_while_waiting_for_capture_admission(self) -> None:
        from uuid import uuid4

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            coordinator = SqliteHistoryCoordinator(root, object(), history_factory=lambda: object())
            coordinator.activate()
            migration = LegacyHistoryMigration(coordinator)
            operation_id = str(uuid4())
            payload = {"operation_id": operation_id, "user_id": "operator"}
            coordinator._store.accept(
                operation_id=operation_id,
                workspace_id="workspace",
                creator_id="operator",
                request_digest=_digest(payload),
                kind="legacy_migration",
                accepted_payload=payload,
            )

            async def cancelled() -> bool:
                return True

            with mock.patch.object(coordinator, "_try_global_capture_lock", return_value=None):
                result = await migration.run_with_parent_fds("workspace", operation_id, cancel_check=cancelled)
            self.assertEqual("cancelled", result["phase"])

    async def test_admission_replay_observes_live_liveness_holder_without_mutating_receipt(self) -> None:
        from uuid import uuid4

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            coordinator = SqliteHistoryCoordinator(root, object(), history_factory=lambda: object())
            coordinator.activate()
            migration = LegacyHistoryMigration(coordinator)
            operation_id = str(uuid4())
            payload = {"operation_id": operation_id, "user_id": "operator"}
            coordinator._store.accept(
                operation_id=operation_id,
                workspace_id="workspace",
                creator_id="operator",
                request_digest=_digest(payload),
                kind="legacy_migration",
                accepted_payload=payload,
            )
            with coordinator._store.hold_liveness(operation_id):
                observed = await migration.run_with_parent_fds("workspace", operation_id)
            self.assertEqual("accepted", observed["phase"])
            self.assertEqual("accepted", (await migration.get("workspace", operation_id))["phase"])

    async def test_task_cancellation_during_migration_terminalizes_without_cancel_check(self) -> None:
        class BlockingHistory:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def migrate_legacy_backups(self, *_args: object, **_kwargs: object) -> int:
                self.started.set()
                await asyncio.Event().wait()
                return 0

        from uuid import uuid4

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = BlockingHistory()
            coordinator = SqliteHistoryCoordinator(root, object(), history_factory=lambda: history)
            coordinator.activate()
            migration = LegacyHistoryMigration(coordinator)
            operation_id = str(uuid4())
            task = asyncio.create_task(migration.accept("workspace", {"operation_id": operation_id, "user_id": "operator"}))
            await history.started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual("cancelled", (await migration.get("workspace", operation_id))["phase"])

    async def test_restic_capture_materializes_a_verified_ready_row(self) -> None:
        binary = Path(os.environ.get("RESTIC_BINARY", "/opt/ragtime-backup/bin/restic"))
        if not binary.is_file():
            self.skipTest("actual Restic binary is unavailable")
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            files = root / "workspaces" / "workspace" / "files"
            database_dir = files / ".ragtime/db"
            database_dir.mkdir(parents=True)
            source = database_dir / "app.sqlite3"
            connection = sqlite3.connect(source)
            connection.execute("create table data (value text)")
            connection.execute("insert into data values ('verified')")
            connection.commit()
            connection.close()

            class Worker:
                async def acquire_sqlite_workspace_access(self, workspace_id, lease_id, *, maintenance=False):
                    return {"authoritative_root": str(files)}

                async def release_sqlite_workspace_access(self, workspace_id, lease_id):
                    return None

            repository = ResticRepository(
                repository_path=root / "_sqlite_history/restic",
                cache_path=root / "_sqlite_history/cache",
                password_path=root / "_sqlite_history/secrets/password",
                binary=binary,
            )
            service = RuntimeSqliteHistoryService(root, Worker(), repository)

            def capture(_files, _name, blob_dir, destination_name):
                destination = blob_dir / destination_name
                incoming = sqlite3.connect(source)
                outgoing = sqlite3.connect(destination)
                incoming.backup(outgoing)
                outgoing.close()
                incoming.close()
                return {"size_bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "source_token": "token"}

            service._capture_confined = capture  # type: ignore[method-assign]
            results = await service.capture_workspace_databases(
                "workspace", trigger="manual", database_names={"app.sqlite3"}, capture_job_id="parent", capture_operation_id="operation-1"
            )
            self.assertEqual("restic", results[0].get("storage", {}).get("kind"), results)
            with mock.patch.object(service, "_probe_confined", return_value="token"):
                alias = await service.capture_workspace_databases(
                    "workspace", trigger="snapshot", database_names={"app.sqlite3"}, capture_job_id="parent-2", capture_operation_id="operation-2"
                )
            self.assertEqual(results[0]["storage"], alias[0]["storage"])
            restored = root / "restored.sqlite3"
            await service.download_to_path("workspace", results[0]["id"], restored)
            check = sqlite3.connect(restored)
            self.assertEqual("verified", check.execute("select value from data").fetchone()[0])
            check.close()

    async def test_pre_restore_identical_image_aliases_verified_restic_snapshot(self) -> None:
        binary = Path(os.environ.get("RESTIC_BINARY", "/opt/ragtime-backup/bin/restic"))
        if not binary.is_file():
            self.skipTest("actual Restic binary is unavailable")
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            files = root / "workspaces" / "workspace" / "files"
            database_dir = files / ".ragtime/db"
            database_dir.mkdir(parents=True)
            source = database_dir / "app.sqlite3"
            with sqlite3.connect(source) as connection:
                connection.execute("create table data (value text)")
                connection.execute("insert into data values ('verified')")

            class Worker:
                async def acquire_sqlite_workspace_access(self, workspace_id, lease_id, *, maintenance=False):
                    return {"authoritative_root": str(files)}

                async def release_sqlite_workspace_access(self, workspace_id, lease_id):
                    return None

            repository = ResticRepository(
                repository_path=root / "_sqlite_history/restic",
                cache_path=root / "_sqlite_history/cache",
                password_path=root / "_sqlite_history/secrets/password",
                binary=binary,
            )
            service = RuntimeSqliteHistoryService(root, Worker(), repository)
            capture_calls = 0
            captured_images: list[bytes] = []

            def capture(_files, _name, blob_dir, destination_name):
                nonlocal capture_calls
                capture_calls += 1
                destination = blob_dir / destination_name
                with sqlite3.connect(source) as incoming, sqlite3.connect(destination) as outgoing:
                    incoming.backup(outgoing)
                captured_images.append(destination.read_bytes())
                return {"size_bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "source_token": "initial"}

            service._capture_confined = capture  # type: ignore[method-assign]
            with (
                mock.patch("runtime.worker.sqlite_history.service.assert_sqlite_workspace_maintenance_held", new=mock.AsyncMock()),
                mock.patch.object(repository, "ingest", wraps=repository.ingest) as ingest,
            ):
                initial = await service.capture_workspace_databases(
                    "workspace", trigger="manual", database_names={"app.sqlite3"}, capture_operation_id="initial"
                )
                safety = await service.capture_workspace_databases(
                    "workspace", trigger="pre_restore", database_names={"app.sqlite3"}, capture_operation_id="pre-restore"
                )
                self.assertEqual(2, capture_calls)
                self.assertEqual(1, ingest.await_count)
            self.assertEqual(initial[0]["storage"], safety[0]["storage"])
            self.assertEqual(1, len(await self._snapshot_ids(repository)))
            self.assertEqual([], await service.forget_unreferenced_snapshots([initial[0]["storage"]["snapshot_id"]]))
            await repository.prune(max_repack_size=1024 * 1024)
            restored = root / "restored.sqlite3"
            await service.download_to_path("workspace", safety[0]["id"], restored)
            self.assertEqual(captured_images[0], restored.read_bytes())

    async def test_changed_source_token_with_identical_output_does_not_ingest_orphan(self) -> None:
        binary = Path(os.environ.get("RESTIC_BINARY", "/opt/ragtime-backup/bin/restic"))
        if not binary.is_file():
            self.skipTest("actual Restic binary is unavailable")
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            files = root / "workspaces" / "workspace" / "files"
            database_dir = files / ".ragtime/db"
            database_dir.mkdir(parents=True)
            source = database_dir / "app.sqlite3"
            with sqlite3.connect(source) as connection:
                connection.execute("create table data (value text)")
                connection.execute("insert into data values ('initial')")

            class Worker:
                async def acquire_sqlite_workspace_access(self, workspace_id, lease_id, *, maintenance=False):
                    return {"authoritative_root": str(files)}

                async def release_sqlite_workspace_access(self, workspace_id, lease_id):
                    return None

            repository = ResticRepository(
                repository_path=root / "_sqlite_history/restic",
                cache_path=root / "_sqlite_history/cache",
                password_path=root / "_sqlite_history/secrets/password",
                binary=binary,
            )
            service = RuntimeSqliteHistoryService(root, Worker(), repository)
            token = "first"

            def capture(_files, _name, blob_dir, destination_name):
                destination = blob_dir / destination_name
                with sqlite3.connect(source) as incoming, sqlite3.connect(destination) as outgoing:
                    incoming.backup(outgoing)
                return {"size_bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "source_token": token}

            service._capture_confined = capture  # type: ignore[method-assign]
            with (
                mock.patch.object(service, "_probe_confined", side_effect=lambda files_dir, database_name: token),
                mock.patch.object(repository, "ingest", wraps=repository.ingest) as ingest,
            ):
                initial = await service.capture_workspace_databases(
                    "workspace", trigger="manual", database_names={"app.sqlite3"}, capture_operation_id="initial"
                )
                token = "changed-but-identical"
                alias = await service.capture_workspace_databases("workspace", trigger="manual", database_names={"app.sqlite3"}, capture_operation_id="alias")
                self.assertEqual(initial[0]["storage"], alias[0]["storage"])
                self.assertEqual(1, ingest.await_count)
                with sqlite3.connect(source) as connection:
                    connection.execute("insert into data values ('changed')")
                token = "changed-output"
                changed = await service.capture_workspace_databases(
                    "workspace", trigger="manual", database_names={"app.sqlite3"}, capture_operation_id="changed"
                )
                self.assertNotEqual(initial[0]["storage"], changed[0]["storage"])
                self.assertEqual(2, ingest.await_count)
            self.assertEqual(2, len(await self._snapshot_ids(repository)))
