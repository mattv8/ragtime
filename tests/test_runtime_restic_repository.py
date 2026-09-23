"""Actual-binary contract tests for the runtime-owned Restic adapter."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime.worker.sqlite_history.models import RESTIC_IMAGE_PATH
from runtime.worker.sqlite_history.repository import ResticRepository, ResticRepositoryError


class ResticRepositoryContractTests(unittest.IsolatedAsyncioTestCase):
    """Run against the pinned binary, never a mocked successful Restic process."""

    def setUp(self) -> None:
        self._temporary = TemporaryDirectory()
        # macOS exposes its temporary directory through /var -> /private/var;
        # configuration passed to the adapter is canonicalized before use.
        self.root = Path(self._temporary.name).resolve()
        self.binary = Path(os.environ.get("RESTIC_BINARY", "/opt/ragtime-backup/bin/restic"))
        if not self.binary.is_file():
            self.fail(f"actual Restic 0.19.1 binary required at {self.binary}; set RESTIC_BINARY for host tests")
        self.repository = self._new_repository()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _new_repository(self) -> ResticRepository:
        return ResticRepository(
            repository_path=self.root / "repository",
            cache_path=self.root / "cache",
            password_path=self.root / "secrets" / "repository-password",
            scratch_path=self.root / "scratch",
            binary=self.binary,
        )

    def _image(self) -> tuple[Path, str, int]:
        image = self.root / "fixture.sqlite3"
        with sqlite3.connect(image) as connection:
            connection.execute("create table records (id integer primary key, payload blob not null)")
            connection.execute("insert into records(payload) values (?)", (b"restic-contract" * 4096,))
        return image, hashlib.sha256(image.read_bytes()).hexdigest(), image.stat().st_size

    async def test_initialize_creates_a_reopenable_repository(self) -> None:
        repository_id = await self.repository.initialize()
        reopened = self._new_repository()
        self.assertEqual(await reopened.initialize(), repository_id)
        self.assertRegex(repository_id, r"^[0-9a-f]{64}$")

    async def test_cancel_drains_worker_even_before_any_child_exists(self) -> None:
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def worker(cancelled: threading.Event, pass_fds: tuple[int, ...]) -> None:
            started.set()
            release.wait(timeout=5)
            finished.set()

        task = asyncio.create_task(self.repository._blocking(worker))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set())

    async def test_existing_repository_rejects_a_missing_key(self) -> None:
        await self.repository.initialize()
        (self.root / "secrets" / "repository-password").unlink()
        with self.assertRaisesRegex(ResticRepositoryError, "key is unavailable"):
            await self._new_repository().initialize()

    async def test_ingest_ls_dump_and_forget_shared_content_are_byte_exact(self) -> None:
        image, digest, size = self._image()
        first = await self.repository.ingest(image, workspace_id="w1", operation_id="a" * 32, sha256=digest, size_bytes=size)
        second = await self.repository.ingest(image, workspace_id="w2", operation_id="b" * 32, sha256=digest, size_bytes=size)

        self.assertEqual(first.path, RESTIC_IMAGE_PATH)
        self.assertRegex(first.snapshot_id, r"^[0-9a-f]{64}$")
        self.assertEqual(
            await self.repository.list_operation_snapshot_ids(workspace_id="w1", operation_id="a" * 32),
            [first.snapshot_id],
        )
        await self.repository.forget([first.snapshot_id])
        restored = self.root / "restored.sqlite3"
        await self.repository.materialize(second, restored)
        self.assertEqual(restored.read_bytes(), image.read_bytes())
        await self.repository.check(read_data=True)

    async def test_artifact_repository_identity_is_checked_before_dump(self) -> None:
        image, digest, size = self._image()
        artifact = await self.repository.ingest(image, workspace_id="w1", operation_id="c" * 32, sha256=digest, size_bytes=size)
        forged = artifact.__class__("0" * 64, artifact.snapshot_id, artifact.path, artifact.size_bytes, artifact.sha256)
        with self.assertRaisesRegex(ResticRepositoryError, "another repository"):
            await self.repository.materialize(forged, self.root / "forged.sqlite3")

    async def test_streamed_verification_checks_bytes_without_creating_an_image(self) -> None:
        image, digest, size = self._image()
        artifact = await self.repository.ingest(image, workspace_id="w1", operation_id="e" * 32, sha256=digest, size_bytes=size)
        await self.repository.verify(artifact)
        self.assertFalse(list(self.root.rglob("restic-materialize-*")))
        with self.assertRaisesRegex(ResticRepositoryError, "verification failed"):
            await self.repository.verify(replace(artifact, sha256="0" * 64))
        with self.assertRaisesRegex(ResticRepositoryError, "output exceeded limit"):
            await self.repository.verify(replace(artifact, size_bytes=size - 1))

    async def test_operation_lookup_requires_both_workspace_and_operation_tags(self) -> None:
        image, digest, size = self._image()
        first = await self.repository.ingest(image, workspace_id="w1", operation_id="f" * 32, sha256=digest, size_bytes=size)
        await self.repository.ingest(image, workspace_id="w1", operation_id="a" * 32, sha256=digest, size_bytes=size)
        self.assertEqual([first.snapshot_id], await self.repository.list_operation_snapshot_ids(workspace_id="w1", operation_id="f" * 32))

    async def test_rejects_untrusted_tags_and_bad_image_digest(self) -> None:
        image, digest, size = self._image()
        with self.assertRaisesRegex(ResticRepositoryError, "operation metadata"):
            await self.repository.ingest(image, workspace_id="w1 --option", operation_id="d" * 32, sha256=digest, size_bytes=size)
        with self.assertRaisesRegex(ResticRepositoryError, "verification failed"):
            await self.repository.ingest(image, workspace_id="w1", operation_id="d" * 32, sha256="0" * 64, size_bytes=size)

    async def test_rejects_a_symlink_in_the_runtime_storage_parent_chain(self) -> None:
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(self.root, target_is_directory=True)
        repository = ResticRepository(
            repository_path=linked_parent / "repository",
            cache_path=self.root / "cache",
            password_path=self.root / "secrets" / "repository-password",
            scratch_path=self.root / "scratch",
            binary=self.binary,
        )
        with self.assertRaisesRegex(ResticRepositoryError, "storage path is invalid"):
            await repository.initialize()
