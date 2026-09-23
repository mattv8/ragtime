import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragtime.userspace.sqlite_history import SqliteHistoryService


class SqliteBackupQueueCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.files = Path(self.temp.name) / "workspace" / "files"
        self.database_dir = self.files / ".ragtime" / "db"
        self.database_dir.mkdir(parents=True)
        self.service = SqliteHistoryService(lambda _: self.files)
        runtime_active_patch = mock.patch.object(
            SqliteHistoryService,
            "runtime_history_active",
            new=mock.AsyncMock(return_value=False),
        )
        runtime_active_patch.start()
        self.addCleanup(runtime_active_patch.stop)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _capture(blob_dir: Path, destination_name: str) -> dict[str, object]:
        destination = blob_dir / destination_name
        destination.write_bytes(b"sqlite backup")
        return {
            "size_bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_token": "stable",
        }

    def test_capture_job_tags_snapshot_rows_and_reuses_same_database_result(self) -> None:
        (self.database_dir / "app.sqlite3").write_bytes(b"source")

        async def capture() -> list[dict[str, object]]:
            with mock.patch.object(
                self.service,
                "_capture_confined",
                side_effect=lambda _files, _name, blob_dir, destination: self._capture(blob_dir, destination),
            ) as confined:
                first = await self.service.capture_workspace_databases(
                    "workspace",
                    trigger="snapshot",
                    snapshot_id="snapshot-1",
                    snapshot_git_commit_hash="commit-1",
                    capture_job_id="job-1",
                    files_dir=self.files,
                )
                second = await self.service.capture_workspace_databases(
                    "workspace",
                    trigger="snapshot",
                    snapshot_id="snapshot-1",
                    snapshot_git_commit_hash="commit-1",
                    capture_job_id="job-1",
                    files_dir=self.files,
                )
                self.assertEqual(1, confined.call_count)
            self.assertEqual(first[0]["id"], second[0]["id"])
            return first

        rows = asyncio.run(capture())
        self.assertEqual("job-1", rows[0]["capture_job_id"])
        self.assertEqual("snapshot-1", rows[0]["snapshot_id"])
        self.assertEqual("commit-1", rows[0]["snapshot_git_commit_hash"])

    def test_capture_stops_at_cancel_boundary_and_reports_partial_progress(self) -> None:
        (self.database_dir / "a.sqlite3").write_bytes(b"a")
        (self.database_dir / "b.sqlite3").write_bytes(b"b")
        calls: list[str] = []
        progress: list[tuple[int, int]] = []

        async def cancelled() -> bool:
            return len(calls) >= 1

        async def reported(completed: int, total: int) -> None:
            progress.append((completed, total))

        def capture_one(*args: object) -> dict[str, object]:
            calls.append(str(args[3]))
            return {"id": "backup-1", "status": "ready", "database_name": args[3], "capture_job_id": args[-1]}

        with mock.patch.object(self.service, "_capture_one", side_effect=capture_one):
            rows = asyncio.run(
                self.service.capture_workspace_databases(
                    "workspace",
                    trigger="manual",
                    capture_job_id="job-2",
                    cancel_check=cancelled,
                    progress_callback=reported,
                    files_dir=self.files,
                )
            )
        self.assertEqual(["a.sqlite3"], calls)
        self.assertEqual([("backup-1")], [str(row["id"]) for row in rows])
        self.assertEqual([(1, 2)], progress)

    def test_progress_callback_failure_propagates_after_processed_database(self) -> None:
        (self.database_dir / "app.sqlite3").write_bytes(b"source")

        async def failed_progress(_completed: int, _total: int) -> None:
            raise RuntimeError("queue unavailable")

        with mock.patch.object(self.service, "_capture_one", return_value={"id": "backup-1", "status": "ready"}):
            with self.assertRaisesRegex(RuntimeError, "queue unavailable"):
                asyncio.run(
                    self.service.capture_workspace_databases(
                        "workspace",
                        trigger="manual",
                        capture_job_id="job-3",
                        progress_callback=failed_progress,
                        files_dir=self.files,
                    )
                )

    def test_failed_capture_row_keeps_queue_job_association(self) -> None:
        (self.database_dir / "app.sqlite3").write_bytes(b"source")
        with mock.patch.object(self.service, "_capture_one", side_effect=RuntimeError("capture failed")):
            rows = asyncio.run(
                self.service.capture_workspace_databases(
                    "workspace",
                    trigger="manual",
                    capture_job_id="job-4",
                    files_dir=self.files,
                )
            )
        self.assertEqual("failed", rows[0]["status"])
        self.assertEqual("job-4", rows[0]["capture_job_id"])

    def test_pre_restore_stays_inline_and_ignores_queue_cancellation(self) -> None:
        (self.database_dir / "app.sqlite3").write_bytes(b"source")
        captured = mock.Mock(return_value={"id": "safety", "status": "ready"})
        cancelled = mock.AsyncMock(return_value=True)
        with (
            mock.patch("ragtime.userspace.sqlite_history.assert_sqlite_workspace_maintenance_held", new_callable=mock.AsyncMock),
            mock.patch.object(self.service, "_capture_one", captured),
        ):
            rows = asyncio.run(
                self.service.capture_workspace_databases(
                    "workspace",
                    trigger="pre_restore",
                    mandatory=True,
                    cancel_check=cancelled,
                    files_dir=self.files,
                )
            )
        self.assertEqual(["safety"], [row["id"] for row in rows])
        captured.assert_called_once()
        cancelled.assert_not_awaited()
