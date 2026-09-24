"""Comprehensive tests for runtime history transfer (export/import) functionality.

Tests cover the full roundtrip with meaningful actual SQLite databases, Restic binary,
and real transfer validation. Focuses on defect scenarios, boundary conditions, and
recovery behavior. Runs in Docker with real Restic binary and proper file permissions.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import sqlite3
import stat
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from runtime.worker.sqlite_history.coordinator import SqliteHistoryCoordinator
from runtime.worker.sqlite_history.export import RuntimeHistoryExporter
from runtime.worker.sqlite_history.repository import ResticRepository
from runtime.worker.sqlite_history.storage import AsyncRepositoryGate
from runtime.worker.sqlite_history.transfer import RuntimeHistoryTransfers

RESTIC_BINARY = Path("/opt/ragtime-backup/bin/restic")


class MockCoordinator:
    """Mock coordinator for transfer testing."""

    def __init__(self, runtime_root: Path) -> None:
        self._runtime_root = runtime_root

    def _service(self) -> object:
        return self

    @property
    def _runtime(self) -> object:
        class Runtime:
            def __init__(self, runtime_root: Path) -> None:
                self._runtime_root = runtime_root

            @property
            def root(self) -> Path:
                return self._runtime_root

        return Runtime(self._runtime_root)


class RuntimeHistoryTransferTests(unittest.IsolatedAsyncioTestCase):
    """Transfer validation with real repository and strict key permissions."""

    async def asyncSetUp(self) -> None:
        """Set up temporary directories and directory structure."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.coordinator = MockCoordinator(self.root)
        self.transfers = RuntimeHistoryTransfers(self.coordinator)

        # Create directory structure with proper permissions (no repo init needed for transfer tests)
        repo_root = self.root / "_sqlite_history" / "restic"
        cache = self.root / "_sqlite_history" / "cache"
        scratch = self.root / "_sqlite_history" / "scratch"
        secret = self.root / "_sqlite_history" / "secrets"

        repo_root.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        scratch.mkdir(parents=True, exist_ok=True)
        secret.mkdir(mode=0o700, parents=True, exist_ok=True)

        # Create password file with 0600 permissions (validates permission checks work)
        self.repo_password_file = secret / "repository-password"
        self.repo_password_file.write_text("test-password-for-testing-only")
        os.chmod(self.repo_password_file, stat.S_IRUSR | stat.S_IWUSR)  # 0600

    async def asyncTearDown(self) -> None:
        """Clean up temporary directories."""
        self.tmpdir.cleanup()

    async def test_export_creates_receipt(self) -> None:
        """Export acceptance creates durable receipt immediately."""
        export_receipt = await self.transfers.accept_export(include_repository_key=True)
        self.assertEqual(export_receipt["status"], "accepted")
        export_id = export_receipt["export_id"]

        # Receipt file must exist immediately
        receipt_path = self.transfers._receipt_path("exports", export_id)
        self.assertTrue(receipt_path.exists())

        # Can retrieve receipt
        retrieved = self.transfers.export_receipt(export_id)
        self.assertEqual(retrieved["status"], "accepted")

    async def test_safe_extract_rejects_duplicates(self) -> None:
        """Tar extract rejects duplicate member names."""
        tar_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        dest = Path(tempfile.mkdtemp(prefix="extract-dest-"))

        try:
            # Create tar with duplicate entries
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="duplicate.txt")
                info.size = 5
                archive.addfile(info, fileobj=__import__("io").BytesIO(b"first"))
                info = tarfile.TarInfo(name="duplicate.txt")
                info.size = 6
                archive.addfile(info, fileobj=__import__("io").BytesIO(b"second"))

            with self.assertRaises(RuntimeError):
                self.transfers._safe_extract(tar_path, dest)
        finally:
            tar_path.unlink(missing_ok=True)
            shutil.rmtree(dest, ignore_errors=True)

    async def test_safe_extract_rejects_absolute_paths(self) -> None:
        """Tar extract rejects absolute paths."""
        tar_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        dest = Path(tempfile.mkdtemp(prefix="extract-dest-"))

        try:
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="/etc/passwd")
                info.size = 0
                archive.addfile(info)

            with self.assertRaises(RuntimeError):
                self.transfers._safe_extract(tar_path, dest)
        finally:
            tar_path.unlink(missing_ok=True)
            shutil.rmtree(dest, ignore_errors=True)

    async def test_safe_extract_rejects_traversal(self) -> None:
        """Tar extract rejects path traversal (..)."""
        tar_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        dest = Path(tempfile.mkdtemp(prefix="extract-dest-"))

        try:
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="../../etc/passwd")
                info.size = 0
                archive.addfile(info)

            with self.assertRaises(RuntimeError):
                self.transfers._safe_extract(tar_path, dest)
        finally:
            tar_path.unlink(missing_ok=True)
            shutil.rmtree(dest, ignore_errors=True)

    async def test_safe_extract_rejects_symlinks(self) -> None:
        """Tar extract rejects symlinks."""
        tar_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        dest = Path(tempfile.mkdtemp(prefix="extract-dest-"))

        try:
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="link.txt")
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                archive.addfile(info)

            with self.assertRaises(RuntimeError):
                self.transfers._safe_extract(tar_path, dest)
        finally:
            tar_path.unlink(missing_ok=True)
            shutil.rmtree(dest, ignore_errors=True)

    async def test_safe_extract_rejects_hardlinks(self) -> None:
        """Tar extract rejects hard links."""
        tar_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        dest = Path(tempfile.mkdtemp(prefix="extract-dest-"))

        try:
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="hardlink")
                info.type = tarfile.LNKTYPE
                info.linkname = "/etc/passwd"
                archive.addfile(info)

            with self.assertRaises(RuntimeError):
                self.transfers._safe_extract(tar_path, dest)
        finally:
            tar_path.unlink(missing_ok=True)
            shutil.rmtree(dest, ignore_errors=True)

    async def test_safe_extract_preserves_owner_permission_bits(self) -> None:
        """Extraction preserves 0600 on the repository key and drops set-id bits."""
        tar_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        dest = Path(tempfile.mkdtemp(prefix="extract-dest-"))
        try:
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="secrets")
                info.type = tarfile.DIRTYPE
                info.mode = 0o700
                archive.addfile(info)
                info = tarfile.TarInfo(name="secrets/repository-password")
                info.size = 3
                info.mode = 0o4600  # setuid bit must be stripped
                archive.addfile(info, fileobj=__import__("io").BytesIO(b"key"))
            self.transfers._safe_extract(tar_path, dest)
            self.assertEqual(stat.S_IMODE((dest / "secrets").stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((dest / "secrets" / "repository-password").stat().st_mode), 0o600)
        finally:
            tar_path.unlink(missing_ok=True)
            shutil.rmtree(dest, ignore_errors=True)

    async def test_receipt_path_validates_uuid(self) -> None:
        """Receipt path validation rejects non-UUID transfer IDs."""
        with self.assertRaises(Exception):
            self.transfers._receipt_path("exports", "not-a-uuid")

        with self.assertRaises(Exception):
            self.transfers._receipt_path("imports", "not-valid-uuid")

        # Valid UUID should succeed
        valid_id = str(uuid4())
        path = self.transfers._receipt_path("exports", valid_id)
        self.assertIn(valid_id, str(path))

    # -- durable install / journal / recovery ------------------------------

    def _seed_destination(self) -> None:
        live = self.root / "_sqlite_history"
        (live / "restic" / "data").mkdir(parents=True, exist_ok=True)
        (live / "restic" / "config").write_text("old-config")
        (live / "restic" / "data" / "pack").write_text("old-pack")
        (live / "operations").mkdir(exist_ok=True)
        (live / "operations" / "old.json").write_text("{}")
        (live / "activation-v1.json").write_text('{"version":2,"active":true}')
        catalog = self.root / "workspaces" / "ws1" / "sqlite_backups"
        catalog.mkdir(parents=True, exist_ok=True)
        (catalog / "manifest-v1.json").write_text("old-manifest")

    def _seed_unpacked(self, unpacked: Path, include_key: bool = True) -> None:
        incoming = unpacked / "_sqlite_history"
        (incoming / "restic" / "data").mkdir(parents=True)
        (incoming / "restic" / "config").write_text("new-config")
        (incoming / "restic" / "data" / "pack").write_text("new-pack")
        (incoming / "operations").mkdir()
        (incoming / "operations" / "new.json").write_text("{}")
        (incoming / "activation-v1.json").write_text('{"version":2,"active":true}')
        catalog = unpacked / "workspaces" / "ws1" / "sqlite_backups"
        catalog.mkdir(parents=True)
        (catalog / "manifest-v1.json").write_text("new-manifest")
        if include_key:
            secrets = unpacked / "secrets"
            secrets.mkdir(mode=0o700)
            (secrets / "repository-password").write_text("new-key")

    def _assert_destination_is_original(self) -> None:
        live = self.root / "_sqlite_history"
        self.assertEqual((live / "restic" / "config").read_text(), "old-config")
        self.assertEqual((live / "restic" / "data" / "pack").read_text(), "old-pack")
        self.assertTrue((live / "operations" / "old.json").exists())
        self.assertFalse((live / "operations" / "new.json").exists())
        self.assertEqual(
            (self.root / "workspaces" / "ws1" / "sqlite_backups" / "manifest-v1.json").read_text(),
            "old-manifest",
        )
        self.assertEqual((live / "secrets" / "repository-password").read_text(), "test-password-for-testing-only")

    def _catalog_failing_replace(self, catalog_source: Path):
        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if Path(src) == catalog_source:
                raise OSError("simulated mid-install crash")
            return real_replace(src, dst, *args, **kwargs)

        return failing_replace

    async def test_midinstall_failure_rolls_back_nonempty_directory_targets(self) -> None:
        """A failure after directories were swapped restores every original.

        Rolling a backup over an installed nonempty directory cannot use a
        bare os.replace; this deterministically fails installs that try.
        """
        self._seed_destination()
        unpacked = self.root / "unpacked"
        self._seed_unpacked(unpacked)
        transfer_id = str(uuid4())
        catalog_source = unpacked / "workspaces" / "ws1" / "sqlite_backups"
        with mock.patch("os.replace", new=self._catalog_failing_replace(catalog_source)):
            with self.assertRaises(OSError):
                self.transfers._install_verified(self.root, unpacked, transfer_id)
        self._assert_destination_is_original()
        journal = self.root / "_sqlite_history" / "transfers" / "import-journal.json"
        self.assertFalse(journal.exists())
        self.assertEqual(list(self.root.rglob(f"*before-import-{transfer_id}*")), [])

    async def test_restart_recovery_rolls_back_incomplete_install_from_journal(self) -> None:
        """A crash mid-install leaves a write-ahead journal that a fresh
        process rolls back on start, restoring the original state."""
        self._seed_destination()
        unpacked = self.root / "unpacked"
        self._seed_unpacked(unpacked)
        transfer_id = str(uuid4())
        catalog_source = unpacked / "workspaces" / "ws1" / "sqlite_backups"
        # Simulate a hard crash: the failing rename happens and in-process
        # rollback never completes, so only the durable journal remains.
        with mock.patch.object(RuntimeHistoryTransfers, "_rollback_actions", return_value=False):
            with mock.patch("os.replace", new=self._catalog_failing_replace(catalog_source)):
                with self.assertRaises(OSError):
                    self.transfers._install_verified(self.root, unpacked, transfer_id)
        journal = self.root / "_sqlite_history" / "transfers" / "import-journal.json"
        self.assertTrue(journal.exists())
        # Journal must be root-relative, never absolute paths.
        payload = json.loads(journal.read_text(encoding="utf-8"))
        for action in payload["actions"]:
            self.assertFalse(action["target"].startswith("/"))
            self.assertFalse(action["backup"].startswith("/"))
        # Destination is torn at this point.
        self.assertEqual((self.root / "_sqlite_history" / "restic" / "config").read_text(), "new-config")
        self.assertFalse((self.root / "workspaces" / "ws1" / "sqlite_backups").exists())

        fresh = RuntimeHistoryTransfers(MockCoordinator(self.root))
        await fresh.start()

        self.assertFalse(journal.exists())
        self._assert_destination_is_original()
        self.assertEqual(list(self.root.rglob(f"*before-import-{transfer_id}*")), [])

    async def test_successful_install_cleans_own_backups_and_preserves_unrelated(self) -> None:
        """Commit removes only this transfer's backups, everywhere they live."""
        self._seed_destination()
        unpacked = self.root / "unpacked"
        self._seed_unpacked(unpacked)
        stray = self.root / "_sqlite_history" / f".restic.before-import-{uuid4()}"
        stray.mkdir()
        (stray / "keep").write_text("unrelated backup content")
        transfer_id = str(uuid4())

        self.transfers._install_verified(self.root, unpacked, transfer_id)

        live = self.root / "_sqlite_history"
        self.assertEqual((live / "restic" / "config").read_text(), "new-config")
        self.assertTrue((live / "operations" / "new.json").exists())
        self.assertEqual(
            (self.root / "workspaces" / "ws1" / "sqlite_backups" / "manifest-v1.json").read_text(),
            "new-manifest",
        )
        self.assertEqual((live / "secrets" / "repository-password").read_text(), "new-key")
        # Unrelated backups from other transfers must never be deleted.
        self.assertTrue((stray / "keep").exists())
        # This transfer's backups are gone everywhere (history root, secrets,
        # and workspace catalog parents), and the journal is cleared.
        self.assertEqual(list(self.root.rglob(f"*before-import-{transfer_id}*")), [])
        self.assertFalse((live / "transfers" / "import-journal.json").exists())

    async def test_start_marks_orphaned_accepted_receipts_failed(self) -> None:
        """Accepted receipts with no live owner fail on startup recovery."""
        export_id = str(uuid4())
        import_id = str(uuid4())
        self.transfers._write_receipt(
            "exports", export_id, {"export_id": export_id, "status": "accepted", "export_version": 1, "includes_repository_key": False}
        )
        self.transfers._write_receipt("imports", import_id, {"import_id": import_id, "status": "accepted"})

        await self.transfers.start()

        self.assertEqual(self.transfers.export_receipt(export_id)["status"], "failed")
        self.assertEqual(self.transfers.import_receipt(import_id)["status"], "failed")

    async def test_start_leaves_live_transfers_untouched(self) -> None:
        """Startup recovery never fails a transfer whose liveness flock is held."""
        export_id = str(uuid4())
        lock_fd = self.transfers._acquire_transfer_lock("exports", export_id)
        try:
            self.transfers._write_receipt(
                "exports", export_id, {"export_id": export_id, "status": "accepted", "export_version": 1, "includes_repository_key": False}
            )
            other = RuntimeHistoryTransfers(MockCoordinator(self.root))
            await other.start()
            self.assertEqual(self.transfers.export_receipt(export_id)["status"], "accepted")
        finally:
            self.transfers._release_transfer_lock(lock_fd)

    async def test_start_preserves_terminal_receipts(self) -> None:
        """Completed and failed receipts are untouched by recovery."""
        completed_id = str(uuid4())
        self.transfers._write_receipt(
            "exports",
            completed_id,
            {"export_id": completed_id, "status": "completed", "export_version": 1, "repository_id": "0" * 64, "includes_repository_key": False},
        )
        await self.transfers.start()
        self.assertEqual(self.transfers.export_receipt(completed_id)["status"], "completed")

    async def test_cleanup_keeps_aged_bundle_while_download_lifetime_is_live(self) -> None:
        export_id = str(uuid4())
        receipt_dir = self.transfers._receipt_path("exports", export_id).parent
        receipt_dir.mkdir(parents=True)
        bundle = receipt_dir / "export.bundle"
        bundle.write_bytes(b"bundle")
        self.transfers._write_receipt(
            "exports",
            export_id,
            {
                "export_id": export_id,
                "status": "completed",
                "completed_at": (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=2)).isoformat(),
            },
        )
        async with self.transfers.export_download_lifetime(export_id):
            await self.transfers.cleanup()
            self.assertTrue(bundle.exists())
        await self.transfers.cleanup()
        self.assertFalse(bundle.exists())

    async def test_start_finishes_committed_journal_cleanup(self) -> None:
        """A committed journal only removes this transfer's backups on start."""
        self._seed_destination()
        transfer_id = str(uuid4())
        live = self.root / "_sqlite_history"
        backup = live / f".restic.before-import-{transfer_id}"
        backup.mkdir()
        (backup / "old").write_text("backup content")
        stray = live / f".restic.before-import-{uuid4()}"
        stray.mkdir()
        (stray / "keep").write_text("unrelated")
        journal = {
            "version": 1,
            "transfer_id": transfer_id,
            "phase": "committed",
            "actions": [
                {
                    "target": "_sqlite_history/restic",
                    "backup": f"_sqlite_history/.restic.before-import-{transfer_id}",
                    "had_existing": True,
                }
            ],
        }
        journal_path = live / "transfers" / "import-journal.json"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(json.dumps(journal), encoding="utf-8")

        await self.transfers.start()

        self.assertFalse(backup.exists())
        self.assertTrue((stray / "keep").exists())
        self.assertFalse(journal_path.exists())
        # The live tree is untouched by committed cleanup.
        self.assertEqual((live / "restic" / "config").read_text(), "old-config")

    async def test_journal_recovery_rejects_unsafe_paths(self) -> None:
        """Absolute or traversal paths in a journal must never be acted on."""
        live = self.root / "_sqlite_history"
        journal_path = live / "transfers" / "import-journal.json"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal = {
            "version": 1,
            "transfer_id": str(uuid4()),
            "phase": "installing",
            "actions": [{"target": "../outside", "backup": "/etc/passwd", "had_existing": False}],
        }
        journal_path.write_text(json.dumps(journal), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.transfers._recover_import_journal()
        # The invalid journal is retained for inspection, never silently dropped.
        self.assertTrue(journal_path.exists())

    async def test_shutdown_drains_inflight_tasks(self) -> None:
        """Shutdown waits for accepted background transfer work to finish."""
        started = asyncio.Event()
        release = asyncio.Event()
        finished: list[str] = []

        async def slow_task() -> None:
            started.set()
            await release.wait()
            finished.append("done")

        task = asyncio.create_task(slow_task())
        self.transfers._tasks["fake"] = task
        task.add_done_callback(lambda _task: self.transfers._tasks.pop("fake", None))
        await started.wait()
        shutdown = asyncio.create_task(self.transfers.shutdown())
        await asyncio.sleep(0)
        self.assertFalse(shutdown.done())
        release.set()
        await shutdown
        self.assertEqual(finished, ["done"])

    @staticmethod
    def _get_file_sha256(path: Path) -> str:
        """Compute SHA256 of a file."""
        import hashlib

        sha = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                sha.update(chunk)
        return sha.hexdigest()


class _ServiceCoordinator:
    """Coordinator+service stand-in exposing a real Restic repository."""

    def __init__(self, root: Path) -> None:
        history = root / "_sqlite_history"
        self.repository = ResticRepository(
            repository_path=history / "restic",
            cache_path=history / "cache",
            password_path=history / "secrets" / "repository-password",
            scratch_path=history / "scratch",
            binary=RESTIC_BINARY,
        )
        self._runtime = SimpleNamespace(root=root)

    def _service(self) -> "_ServiceCoordinator":
        return self


@unittest.skipUnless(RESTIC_BINARY.is_file(), f"actual Restic binary required at {RESTIC_BINARY}")
class RuntimeHistoryTransferRoundtripTests(unittest.IsolatedAsyncioTestCase):
    """Full export -> import -> materialize roundtrips against actual Restic."""

    async def asyncSetUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name).resolve()
        self._coordinators: list[SqliteHistoryCoordinator] = []

    async def asyncTearDown(self) -> None:
        for coordinator in reversed(self._coordinators):
            await coordinator.shutdown()
        self._temporary.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _fixture_database(self) -> tuple[Path, str, int]:
        image = self.base / "fixture.sqlite3"
        with sqlite3.connect(image) as connection:
            connection.execute("create table records (id integer primary key, payload blob not null)")
            connection.execute("insert into records(payload) values (?)", (b"transfer-roundtrip" * 4096,))
        return image, self._sha256(image), image.stat().st_size

    async def _stage_source(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        coordinator = _ServiceCoordinator(root)
        transfers = RuntimeHistoryTransfers(coordinator)
        image, digest, size = self._fixture_database()
        artifact = await coordinator.repository.ingest(image, workspace_id="ws-roundtrip", operation_id="e" * 32, sha256=digest, size_bytes=size)
        catalog = root / "workspaces" / "ws-roundtrip" / "sqlite_backups"
        catalog.mkdir(parents=True)
        (catalog / "manifest-v1.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "workspace_id": "ws-roundtrip",
                    "backups": [
                        {
                            "id": "backup-1",
                            "status": "ready",
                            "size_bytes": size,
                            "sha256": digest,
                            "storage": {"kind": "restic", "repository_id": artifact.repository_id, "snapshot_id": artifact.snapshot_id, "path": artifact.path},
                        }
                    ],
                    "previews": {},
                    "operations": {},
                }
            ),
            encoding="utf-8",
        )
        return transfers, artifact, digest, image

    async def _stage_active_coordinator_source(self, root: Path):
        """Create a real active coordinator and a keyed portable source bundle."""
        root.mkdir(parents=True, exist_ok=True)
        coordinator = SqliteHistoryCoordinator(root, object())
        self._coordinators.append(coordinator)
        await coordinator.start()
        self.assertFalse(coordinator.capability())
        coordinator.activate()
        repository = coordinator._service().repository
        image, digest, size = self._fixture_database()
        artifact = await repository.ingest(
            image,
            workspace_id="ws-roundtrip",
            operation_id="e" * 32,
            sha256=digest,
            size_bytes=size,
        )
        catalog = root / "workspaces" / "ws-roundtrip" / "sqlite_backups"
        catalog.mkdir(parents=True)
        (catalog / "manifest-v1.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "workspace_id": "ws-roundtrip",
                    "backups": [
                        {
                            "id": "backup-1",
                            "status": "ready",
                            "size_bytes": size,
                            "sha256": digest,
                            "storage": {
                                "kind": "restic",
                                "repository_id": artifact.repository_id,
                                "snapshot_id": artifact.snapshot_id,
                                "path": artifact.path,
                            },
                        }
                    ],
                    "previews": {},
                    "operations": {},
                }
            ),
            encoding="utf-8",
        )
        return coordinator, artifact, digest, image

    async def _run_export(self, transfers: RuntimeHistoryTransfers, include_key: bool):
        receipt = await transfers.accept_export(include_repository_key=include_key)
        export_id = receipt["export_id"]
        task = transfers._tasks.get(export_id)
        if task is not None:
            await task
        completed = transfers.export_receipt(export_id)
        self.assertEqual(completed["status"], "completed")
        return completed, transfers.export_bundle(export_id)

    async def _run_import(self, transfers: RuntimeHistoryTransfers, bundle: Path, repository_id: str) -> str:
        upload = self.base / f"upload-{uuid4()}.bundle"
        shutil.copy2(bundle, upload)
        receipt = await transfers.accept_import(upload, {"repository_id": repository_id})
        import_id = receipt["import_id"]
        task = transfers._tasks.get(import_id)
        if task is not None:
            await task
        return import_id

    async def test_full_export_import_materialize_roundtrip_with_key(self) -> None:
        """Repository, key, and catalogs arrive together and images restore byte-exact."""
        source_root = self.base / "source"
        transfers_a, artifact, digest, image = await self._stage_source(source_root)
        completed, bundle = await self._run_export(transfers_a, include_key=True)
        self.assertEqual(completed["repository_id"], artifact.repository_id)

        destination_root = self.base / "destination"
        destination_root.mkdir()
        coordinator_b = _ServiceCoordinator(destination_root)
        transfers_b = RuntimeHistoryTransfers(coordinator_b)
        import_id = await self._run_import(transfers_b, bundle, artifact.repository_id)
        self.assertEqual(transfers_b.import_receipt(import_id)["status"], "completed")

        # Catalog, key, and repository must arrive together.
        manifest = json.loads((destination_root / "workspaces" / "ws-roundtrip" / "sqlite_backups" / "manifest-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["backups"][0]["storage"]["snapshot_id"], artifact.snapshot_id)
        source_key = (source_root / "_sqlite_history" / "secrets" / "repository-password").read_bytes()
        destination_key = destination_root / "_sqlite_history" / "secrets" / "repository-password"
        self.assertEqual(destination_key.read_bytes(), source_key)
        self.assertEqual(stat.S_IMODE(destination_key.stat().st_mode), 0o600)
        # Journal cleared and no backup staging left behind.
        self.assertFalse((destination_root / "_sqlite_history" / "transfers" / "import-journal.json").exists())
        self.assertEqual(list(destination_root.rglob("*before-import-*")), [])

        restored = self.base / "restored-keyed.sqlite3"
        await coordinator_b.repository.materialize(artifact, restored)
        self.assertEqual(restored.read_bytes(), image.read_bytes())
        self.assertEqual(self._sha256(restored), digest)

    async def test_keyless_import_uses_existing_destination_key(self) -> None:
        """A keyless bundle installs against the destination's existing key."""
        source_root = self.base / "source"
        transfers_a, artifact, digest, image = await self._stage_source(source_root)
        completed, bundle = await self._run_export(transfers_a, include_key=False)
        self.assertFalse(completed["includes_repository_key"])
        with tarfile.open(bundle, "r:*") as archive:
            self.assertNotIn("secrets/repository-password", archive.getnames())

        destination_root = self.base / "destination"
        secrets = destination_root / "_sqlite_history" / "secrets"
        secrets.mkdir(mode=0o700, parents=True)
        shutil.copy2(source_root / "_sqlite_history" / "secrets" / "repository-password", secrets / "repository-password")
        coordinator_b = _ServiceCoordinator(destination_root)
        transfers_b = RuntimeHistoryTransfers(coordinator_b)
        import_id = await self._run_import(transfers_b, bundle, artifact.repository_id)
        self.assertEqual(transfers_b.import_receipt(import_id)["status"], "completed")

        restored = self.base / "restored-keyless.sqlite3"
        await coordinator_b.repository.materialize(artifact, restored)
        self.assertEqual(self._sha256(restored), digest)

    async def test_keyless_import_without_destination_key_fails_and_preserves_destination(self) -> None:
        """Validation failure must leave the destination untouched."""
        source_root = self.base / "source"
        transfers_a, artifact, _digest, _image = await self._stage_source(source_root)
        _completed, bundle = await self._run_export(transfers_a, include_key=False)

        destination_root = self.base / "destination"
        destination_root.mkdir()
        marker = destination_root / "_sqlite_history" / "restic"
        marker.mkdir(parents=True)
        (marker / "config").write_text("pre-existing destination state")
        coordinator_b = _ServiceCoordinator(destination_root)
        transfers_b = RuntimeHistoryTransfers(coordinator_b)
        import_id = await self._run_import(transfers_b, bundle, artifact.repository_id)
        self.assertEqual(transfers_b.import_receipt(import_id)["status"], "failed")
        self.assertEqual((marker / "config").read_text(), "pre-existing destination state")
        self.assertFalse((destination_root / "workspaces").exists())

    async def test_import_rejects_same_repository_destination_only_snapshot_absent_from_export(self) -> None:
        """An older export cannot replace a repository containing a newer catalog snapshot."""
        source_root = self.base / "source"
        transfers, artifact, digest, image = await self._stage_source(source_root)
        _completed, bundle = await self._run_export(transfers, include_key=True)

        post_export = self.base / "post-export.sqlite3"
        with sqlite3.connect(post_export) as connection:
            connection.execute("create table records (id integer primary key, payload blob not null)")
            connection.execute("insert into records(payload) values (?)", (b"post-export" * 4096,))
        post_digest = self._sha256(post_export)
        post_artifact = await transfers._service().repository.ingest(
            post_export,
            workspace_id="ws-destination-only",
            operation_id="f" * 32,
            sha256=post_digest,
            size_bytes=post_export.stat().st_size,
        )
        catalog = source_root / "workspaces" / "ws-destination-only" / "sqlite_backups"
        catalog.mkdir(parents=True)
        manifest = catalog / "manifest-v1.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "workspace_id": "ws-destination-only",
                    "backups": [
                        {
                            "id": "post-export",
                            "status": "ready",
                            "size_bytes": post_export.stat().st_size,
                            "sha256": post_digest,
                            "storage": {
                                "kind": "restic",
                                "repository_id": post_artifact.repository_id,
                                "snapshot_id": post_artifact.snapshot_id,
                                "path": post_artifact.path,
                            },
                        }
                    ],
                    "previews": {},
                    "operations": {},
                }
            ),
            encoding="utf-8",
        )
        original_manifest = manifest.read_bytes()
        original_config = (source_root / "_sqlite_history" / "restic" / "config").read_bytes()

        import_id = await self._run_import(transfers, bundle, artifact.repository_id)

        self.assertEqual(transfers.import_receipt(import_id)["status"], "failed")
        self.assertEqual(manifest.read_bytes(), original_manifest)
        self.assertEqual((source_root / "_sqlite_history" / "restic" / "config").read_bytes(), original_config)
        restored = self.base / "restored-post-export.sqlite3"
        await transfers._service().repository.materialize(post_artifact, restored)
        self.assertEqual(restored.read_bytes(), post_export.read_bytes())
        self.assertEqual(self._sha256(image), digest)

    async def test_import_allows_destination_only_catalog_without_restic_reference(self) -> None:
        """Legacy local-only destination catalogs do not depend on the replaced repository."""
        source_root = self.base / "source"
        transfers, artifact, _digest, _image = await self._stage_source(source_root)
        _completed, bundle = await self._run_export(transfers, include_key=True)
        catalog = source_root / "workspaces" / "ws-destination-only" / "sqlite_backups"
        catalog.mkdir(parents=True)
        manifest = catalog / "manifest-v1.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "workspace_id": "ws-destination-only",
                    "backups": [{"id": "legacy", "status": "ready", "size_bytes": 1, "sha256": "0" * 64}],
                    "previews": {},
                    "operations": {},
                },
            ),
            encoding="utf-8",
        )

        import_id = await self._run_import(transfers, bundle, artifact.repository_id)

        self.assertEqual(transfers.import_receipt(import_id)["status"], "completed")
        self.assertTrue(manifest.exists())

    async def test_keyed_import_activates_a_started_inactive_coordinator(self) -> None:
        """A normal keyed transfer activates an already-started fresh coordinator."""
        source, artifact, digest, image = await self._stage_active_coordinator_source(self.base / "source-coordinator")
        _completed, bundle = await self._run_export(source.get_history_transfers(), include_key=True)

        destination_root = self.base / "destination-coordinator"
        destination_root.mkdir()
        destination = SqliteHistoryCoordinator(destination_root, object())
        self._coordinators.append(destination)
        await destination.start()
        self.assertFalse(destination.capability())

        upload = self.base / "keyed-coordinator.bundle"
        shutil.copy2(bundle, upload)
        transfers = destination.get_history_transfers()
        accepted = await transfers.accept_import(upload, {"repository_id": artifact.repository_id})
        task = transfers._tasks[accepted["import_id"]]
        await task

        self.assertEqual(transfers.import_receipt(accepted["import_id"])["status"], "completed")
        self.assertTrue(destination.capability())
        self.assertEqual((await transfers.status())["repository_id"], artifact.repository_id)
        restored = self.base / "restored-started-inactive.sqlite3"
        await destination._service().repository.materialize(artifact, restored)
        self.assertEqual(self._sha256(restored), digest)
        self.assertEqual(restored.read_bytes(), image.read_bytes())

    async def test_start_recovers_activation_rename_crash_before_activation_is_read(self) -> None:
        """Startup rolls back a journal left after activation was moved aside."""
        source, artifact, _digest, _image = await self._stage_active_coordinator_source(self.base / "source-crash")
        _completed, bundle = await self._run_export(source.get_history_transfers(), include_key=True)

        destination_root = self.base / "destination-crash"
        destination_root.mkdir()
        destination = SqliteHistoryCoordinator(destination_root, object())
        destination.activate()
        transfers = destination.get_history_transfers()
        upload = self.base / "crash.bundle"
        shutil.copy2(bundle, upload)
        activation = destination_root / "_sqlite_history" / "activation-v1.json"
        real_replace = os.replace

        def crash_activation_install(src, dst, *args, **kwargs):
            if Path(dst) == activation:
                raise OSError("simulated process crash during activation rename")
            return real_replace(src, dst, *args, **kwargs)

        with mock.patch("os.replace", new=crash_activation_install):
            accepted = await transfers.accept_import(upload, {"repository_id": artifact.repository_id})
            await transfers._tasks[accepted["import_id"]]

        self.assertEqual(transfers.import_receipt(accepted["import_id"])["status"], "failed")
        self.assertFalse(activation.exists(), "old activation was renamed to its journal backup")
        journal = destination_root / "_sqlite_history" / "transfers" / "import-journal.json"
        self.assertTrue(journal.exists())

        recovered = SqliteHistoryCoordinator(destination_root, object())
        self._coordinators.append(recovered)
        await recovered.start()

        self.assertFalse(journal.exists())
        self.assertTrue(recovered.capability())
        self.assertEqual(
            json.loads(activation.read_text(encoding="utf-8")),
            {"version": 2, "active": True},
        )


class RuntimeHistoryTransferRouterTests(unittest.TestCase):
    """Test transfer router configuration."""

    def test_router_accepts_coordinator_getter_callable(self) -> None:
        """Router factory accepts coordinator_getter callable."""
        from runtime.worker.sqlite_history.transfer_api import create_transfer_router

        call_count = [0]

        def mock_getter() -> object:
            call_count[0] += 1
            return object()

        def mock_dependency() -> None:
            pass

        # Should not raise
        router = create_transfer_router(mock_getter, mock_dependency, prefix="/test")
        self.assertIsNotNone(router)
        # Coordinator getter should not be called during router creation
        self.assertEqual(call_count[0], 0)

    def test_router_prefix_parameter_works(self) -> None:
        """Router accepts prefix parameter."""
        from runtime.worker.sqlite_history.transfer_api import create_transfer_router

        def mock_getter() -> object:
            return object()

        def mock_dependency() -> None:
            pass

        router1 = create_transfer_router(mock_getter, mock_dependency, prefix="")
        router2 = create_transfer_router(mock_getter, mock_dependency, prefix="/worker")

        self.assertEqual(router1.prefix, "")
        self.assertEqual(router2.prefix, "/worker")

    def test_router_delegates_to_coordinator_cached_transfers(self) -> None:
        """Routes must reuse the coordinator's cached transfers instance."""
        try:
            from fastapi import Depends, FastAPI
            from fastapi.testclient import TestClient
        except ImportError:  # pragma: no cover - available in the verification image
            self.skipTest("fastapi test client unavailable")
        from runtime.worker.sqlite_history.transfer_api import create_transfer_router

        class FakeTransfers:
            def __init__(self) -> None:
                self.status_calls = 0

            async def status(self) -> dict:
                self.status_calls += 1
                return {"active": True, "repository_id": "0" * 64, "export_version": 1}

        class FakeCoordinator:
            def __init__(self) -> None:
                self.transfers = FakeTransfers()
                self.getter_calls = 0

            def get_history_transfers(self) -> FakeTransfers:
                self.getter_calls += 1
                return self.transfers

        coordinator = FakeCoordinator()
        application = FastAPI()
        application.include_router(create_transfer_router(lambda: coordinator, Depends(lambda: None)))
        client = TestClient(application)
        first = client.get("/sqlite-history/exports/status")
        second = client.get("/sqlite-history/exports/status")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        # Both requests used the coordinator's cached instance, not fresh ones.
        self.assertEqual(coordinator.getter_calls, 2)
        self.assertEqual(coordinator.transfers.status_calls, 2)


if __name__ == "__main__":
    unittest.main()
