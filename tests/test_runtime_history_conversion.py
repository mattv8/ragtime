from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi import HTTPException

from runtime.worker.sqlite_history.conversion import LegacyHistoryConverter
from runtime.worker.sqlite_history.models import RESTIC_IMAGE_PATH, ResticArtifact
from runtime.worker.sqlite_history.repository import ResticRepository
from runtime.worker.sqlite_history.service import RuntimeSqliteHistoryService


class _Repository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ingests = 0
        self.snapshots: dict[str, ResticArtifact] = {}

    async def initialize(self, **_kwargs: object) -> str:
        return "a" * 64

    async def ingest(self, image: Path, **kwargs: object) -> ResticArtifact:
        self.ingests += 1
        snapshot = f"{self.ingests:x}".zfill(64)
        destination = self.root / snapshot
        shutil.copyfile(image, destination)
        artifact = ResticArtifact("a" * 64, snapshot, RESTIC_IMAGE_PATH, destination.stat().st_size, hashlib.sha256(destination.read_bytes()).hexdigest())
        self.snapshots[str(kwargs["operation_id"])] = artifact
        return artifact

    async def list_operation_snapshot_ids(self, *, operation_id: str, **_kwargs: object) -> list[str]:
        artifact = self.snapshots.get(operation_id)
        return [artifact.snapshot_id] if artifact else []

    async def verify(self, artifact: ResticArtifact, **_kwargs: object) -> None:
        image = self.root / artifact.snapshot_id
        if not image.is_file() or image.stat().st_size != artifact.size_bytes or hashlib.sha256(image.read_bytes()).hexdigest() != artifact.sha256:
            raise RuntimeError("bad restic bytes")


class RuntimeHistoryConversionTests(unittest.IsolatedAsyncioTestCase):
    def _service(self, root: Path) -> tuple[RuntimeSqliteHistoryService, Path, _Repository]:
        files = root / "workspaces" / "workspace" / "files"
        (files / ".ragtime/db").mkdir(parents=True)
        repository = _Repository(root / "repo")
        repository.root.mkdir()
        return RuntimeSqliteHistoryService(root, object(), repository), files.parent / "sqlite_backups", repository

    @staticmethod
    def _manifest(service: RuntimeSqliteHistoryService, history: Path, image: Path) -> None:
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
                        "blob": "blobs/legacy.sqlite3",
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
                        "storage": {"kind": "legacy_file", "blob": "blobs/legacy.sqlite3"},
                    },
                ],
                "previews": {},
                "operations": {},
            },
        )

    async def test_resume_after_catalog_save_adopts_same_tag_without_second_ingest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            service, history, repository = self._service(root)
            (history / "blobs").mkdir(parents=True)
            image = history / "blobs/legacy.sqlite3"
            image.write_bytes(b"immutable")
            self._manifest(service, history, image)
            converter = LegacyHistoryConverter(service, "workspace", pass_fds=())
            original = converter._save_ledger
            failed = False

            def fail_after_catalog(ledger: dict[str, object]) -> None:
                nonlocal failed
                if ledger.get("stage") == "catalog_published" and not failed:
                    failed = True
                    raise OSError("injected after catalog save")
                original(ledger)

            converter._save_ledger = fail_after_catalog  # type: ignore[method-assign]
            with self.assertRaises(OSError):
                await converter.migrate()
            self.assertEqual(1, repository.ingests)
            self.assertTrue(image.exists())
            self.assertEqual(0, await service.migrate_legacy_backups("workspace"))
            self.assertEqual(1, repository.ingests)
            self.assertFalse(image.exists())
            journal = next((history / "conversion-ledger").glob("*.json"))
            self.assertNotIn("legacy.sqlite3", journal.name)

    async def test_fresh_conversion_verifies_once_but_resumed_stage_reverifies(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            service, history, repository = self._service(root)
            (history / "blobs").mkdir(parents=True)
            image = history / "blobs/legacy.sqlite3"
            image.write_bytes(b"immutable")
            self._manifest(service, history, image)
            converter = LegacyHistoryConverter(service, "workspace", pass_fds=())
            with mock.patch.object(repository, "verify", wraps=repository.verify) as verify:
                self.assertEqual(2, await converter.migrate())
                self.assertEqual(1, verify.await_count)

            image.write_bytes(b"resumed")
            self._manifest(service, history, image)
            converter = LegacyHistoryConverter(service, "workspace", pass_fds=())
            original = converter._save_ledger

            def stop_after_repository_verification(ledger: dict[str, object]) -> None:
                original(ledger)
                if ledger.get("stage") == "repository_verified":
                    raise OSError("simulated controller loss")

            converter._save_ledger = stop_after_repository_verification  # type: ignore[method-assign]
            with mock.patch.object(repository, "verify", wraps=repository.verify) as verify:
                with self.assertRaises(OSError):
                    await converter.migrate()
                self.assertEqual(1, verify.await_count)
            with mock.patch.object(repository, "verify", wraps=repository.verify) as verify:
                self.assertEqual(2, await service.migrate_legacy_backups("workspace"))
                self.assertEqual(1, verify.await_count)

    async def test_low_space_blocks_before_repository_write(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            service, history, repository = self._service(root)
            (history / "blobs").mkdir(parents=True)
            image = history / "blobs/legacy.sqlite3"
            image.write_bytes(b"immutable")
            self._manifest(service, history, image)
            with mock.patch("runtime.worker.sqlite_history.conversion.os.statvfs", return_value=mock.Mock(f_bavail=0, f_frsize=4096)):
                with self.assertRaises(HTTPException) as failure:
                    await service.migrate_legacy_backups("workspace")
            self.assertEqual(507, failure.exception.status_code)
            self.assertEqual(0, repository.ingests)
            self.assertTrue(image.exists())

    async def test_resume_after_source_unlink_before_journal_advance(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            service, history, repository = self._service(root)
            (history / "blobs").mkdir(parents=True)
            image = history / "blobs/legacy.sqlite3"
            image.write_bytes(b"immutable")
            self._manifest(service, history, image)
            converter = LegacyHistoryConverter(service, "workspace", pass_fds=())
            original = converter._save_ledger
            failed = False

            def fail_after_unlink(ledger: dict[str, object]) -> None:
                nonlocal failed
                if ledger.get("stage") == "source_removed" and not failed:
                    failed = True
                    raise OSError("injected after source unlink")
                original(ledger)

            converter._save_ledger = fail_after_unlink  # type: ignore[method-assign]
            with self.assertRaises(OSError):
                await converter.migrate()
            self.assertFalse(image.exists())
            self.assertEqual(0, await service.migrate_legacy_backups("workspace"))
            self.assertEqual(1, repository.ingests)

    async def test_malformed_journal_stage_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            service, history, _repository = self._service(root)
            (history / "blobs").mkdir(parents=True)
            image = history / "blobs/legacy.sqlite3"
            image.write_bytes(b"immutable")
            self._manifest(service, history, image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            operation = LegacyHistoryConverter._operation_id("workspace", "blobs/legacy.sqlite3", digest)
            directory = history / "conversion-ledger"
            directory.mkdir()
            (directory / f"{hashlib.sha256(operation.encode()).hexdigest()}.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "workspace_id": "workspace",
                        "legacy_blob": "blobs/legacy.sqlite3",
                        "sha256": digest,
                        "size_bytes": image.stat().st_size,
                        "operation_id": operation,
                        "backup_ids": ["one"],
                        "stage": "not-a-stage",
                    }
                )
            )
            with self.assertRaises(HTTPException) as failure:
                await service.migrate_legacy_backups("workspace")
            self.assertEqual(409, failure.exception.status_code)
            self.assertTrue(image.exists())

    async def test_delete_waits_for_conversion_publication_barrier(self) -> None:
        class SlowRepository(_Repository):
            def __init__(self, root: Path) -> None:
                super().__init__(root)
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def verify(self, artifact: ResticArtifact, **kwargs: object) -> None:
                self.entered.set()
                await self.release.wait()
                await super().verify(artifact, **kwargs)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = root / "workspaces" / "workspace" / "files"
            (files / ".ragtime/db").mkdir(parents=True)
            repository = SlowRepository(root / "repo")
            repository.root.mkdir()
            service = RuntimeSqliteHistoryService(root, object(), repository)
            history = files.parent / "sqlite_backups"
            (history / "blobs").mkdir(parents=True)
            image = history / "blobs/legacy.sqlite3"
            image.write_bytes(b"immutable")
            self._manifest(service, history, image)
            migration = asyncio.create_task(service.migrate_legacy_backups("workspace"))
            await asyncio.wait_for(repository.entered.wait(), timeout=2)
            deletion = asyncio.create_task(service.delete("workspace", "one"))
            await asyncio.sleep(0)
            self.assertFalse(deletion.done())
            repository.release.set()
            self.assertEqual(2, await migration)
            await deletion
            self.assertFalse(image.exists())

    async def test_actual_restic_lost_ingest_response_is_adopted_without_duplicate_snapshot(self) -> None:
        binary = Path(os.environ.get("RESTIC_BINARY", "/opt/ragtime-backup/bin/restic"))
        if not binary.is_file():
            self.skipTest("actual Restic binary is unavailable")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = root / "workspaces" / "workspace" / "files"
            (files / ".ragtime/db").mkdir(parents=True)
            history = files.parent / "sqlite_backups"
            (history / "blobs").mkdir(parents=True)
            image = history / "blobs/legacy.sqlite3"
            image.write_bytes(b"actual-restic-immutable")
            repository = ResticRepository(
                repository_path=root / "_sqlite_history/restic",
                cache_path=root / "_sqlite_history/cache",
                password_path=root / "_sqlite_history/secrets/password",
                binary=binary,
            )
            service = RuntimeSqliteHistoryService(root, object(), repository)
            self._manifest(service, history, image)
            original = repository.ingest

            async def lose_response(
                image: Path,
                *,
                workspace_id: str,
                operation_id: str,
                sha256: str,
                size_bytes: int,
                pass_fds: tuple[int, ...] = (),
            ) -> ResticArtifact:
                await original(
                    image,
                    workspace_id=workspace_id,
                    operation_id=operation_id,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    pass_fds=pass_fds,
                )
                raise RuntimeError("injected lost response after repository commit")

            repository.ingest = lose_response  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError):
                await service.migrate_legacy_backups("workspace")
            repository.ingest = original  # type: ignore[method-assign]
            self.assertEqual(1, await RuntimeHistoryMigrationTests._snapshot_count(repository))
            # A newly constructed service proves journal/tag recovery does not
            # depend on in-memory result state.
            resumed = RuntimeSqliteHistoryService(root, object(), repository)
            self.assertEqual(2, await resumed.migrate_legacy_backups("workspace"))
            self.assertEqual(1, await RuntimeHistoryMigrationTests._snapshot_count(repository))
            self.assertFalse(image.exists())


class RuntimeHistoryMigrationTests:
    @staticmethod
    async def _snapshot_count(repository: ResticRepository) -> int:
        def snapshots(cancelled: object, pass_fds: tuple[int, ...]) -> str:
            return repository._run("snapshots", "--json", cancelled=cancelled, pass_fds=pass_fds)  # type: ignore[arg-type]

        import json

        return len(json.loads(await repository._blocking(snapshots)))
