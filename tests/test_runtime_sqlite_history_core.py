"""Focused runtime-only history extraction contracts."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class RuntimeHistoryCoreTests(unittest.TestCase):
    def test_capture_child_imports_without_ragtime_application(self) -> None:
        import runtime.worker.sqlite_history.capture_child as child

        self.assertEqual("app.sqlite3", child._name("app.sqlite3"))

    def test_workspace_state_uses_legacy_operation_lock_name(self) -> None:
        from runtime.core.sqlite_workspace_state import OPERATION_LOCK_NAME, workspace_operation

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with workspace_operation(root, "workspace-a", exclusive=False):
                pass
            self.assertTrue((root / "workspaces" / "workspace-a" / "sqlite_backups" / OPERATION_LOCK_NAME).is_file())

    def test_runtime_service_keeps_unchanged_alias_and_scheduled_skip_policy(self) -> None:
        from unittest import mock

        from runtime.worker.sqlite_history.service import RuntimeSqliteHistoryService

        class Worker:
            pass

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = root / "workspaces" / "workspace" / "files"
            (files / ".ragtime" / "db").mkdir(parents=True)
            (files / ".ragtime" / "db" / "app.sqlite3").write_bytes(b"live")
            service = RuntimeSqliteHistoryService(root, Worker())

            def capture(_files, _name, blob_dir, destination):
                target = blob_dir / destination
                target.write_bytes(b"image")
                return {"size_bytes": 5, "sha256": "6105d6cc76af400325e94d588ce511be5bfdbb73b437dc51eca43917d7a43e3d", "source_token": "same"}

            with (
                mock.patch.object(service, "_probe_confined", return_value="same"),
                mock.patch.object(service, "_capture_confined", side_effect=capture) as captured,
            ):
                first = service._capture_one("workspace", files.parent / "sqlite_backups", files, "app.sqlite3", "manual", None, None)
                second = service._capture_one("workspace", files.parent / "sqlite_backups", files, "app.sqlite3", "snapshot", None, None)
                skipped = service._capture_one("workspace", files.parent / "sqlite_backups", files, "app.sqlite3", "scheduled", None, None)
            self.assertEqual(first["blob"], second["blob"])
            self.assertEqual("skipped_unchanged", skipped["outcome"])
            self.assertEqual(1, captured.call_count)

    def test_pre_restore_never_reuses_source_token(self) -> None:
        from unittest import mock

        from runtime.worker.sqlite_history.service import RuntimeSqliteHistoryService

        class Worker:
            pass

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = root / "workspaces" / "workspace" / "files"
            (files / ".ragtime" / "db").mkdir(parents=True)
            (files / ".ragtime" / "db" / "app.sqlite3").write_bytes(b"live")
            service = RuntimeSqliteHistoryService(root, Worker())

            def capture(_files, _name, blob_dir, destination):
                target = blob_dir / destination
                target.write_bytes(b"image")
                return {"size_bytes": 5, "sha256": "6105d6cc76af400325e94d588ce511be5bfdbb73b437dc51eca43917d7a43e3d", "source_token": "same"}

            with (
                mock.patch.object(service, "_probe_confined", return_value="same"),
                mock.patch.object(service, "_capture_confined", side_effect=capture) as captured,
            ):
                service._capture_one("workspace", files.parent / "sqlite_backups", files, "app.sqlite3", "manual", None, None)
                safety = service._capture_one("workspace", files.parent / "sqlite_backups", files, "app.sqlite3", "pre_restore", None, None)
            self.assertEqual("pre_restore", safety["trigger"])
            self.assertEqual(2, captured.call_count)
