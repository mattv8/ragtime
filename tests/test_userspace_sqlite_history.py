from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.sqlite_history import SqliteHistoryService


class SqliteHistoryCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.files = Path(self.temp.name) / "workspace" / "files"
        database_dir = self.files / ".ragtime" / "db"
        database_dir.mkdir(parents=True)
        self.database = database_dir / "app.sqlite3"
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item(value) VALUES ('before')")
        self.history = SqliteHistoryService(lambda workspace_id: self.files)

        @asynccontextmanager
        async def recovery(workspace_id: str, lease_id: str):
            yield

        self._recovery_patch = mock.patch("ragtime.userspace.sqlite_history.sqlite_workspace_recovery", recovery)
        self._recovery_patch.start()

    async def asyncTearDown(self) -> None:
        self._recovery_patch.stop()
        self.temp.cleanup()

    def _runtime_boundary(self):
        @asynccontextmanager
        async def access(workspace_id: str, *, maintenance: bool = False):
            marker = self.files.parent / "sqlite_backups" / "sqlite-maintenance-intent.json"
            if maintenance:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"lease_id": "test-maintenance-lease"}), encoding="utf-8")
            try:
                yield self.files
            finally:
                if maintenance:
                    marker.unlink(missing_ok=True)

        return access

    async def test_capture_records_exact_snapshot_association_and_lists_after_live_delete(self) -> None:
        with mock.patch("ragtime.userspace.sqlite_history.sqlite_workspace_access", self._runtime_boundary()):
            backups = await self.history.capture_workspace_databases(
                "workspace", trigger="snapshot", snapshot_id="snapshot-1", snapshot_git_commit_hash="abc123"
            )
            self.database.unlink()
            listed = await self.history.list_backups("workspace", snapshot_id="snapshot-1")

        self.assertEqual(1, len(backups))
        self.assertEqual("snapshot-1", listed[0]["snapshot_id"])
        self.assertEqual("abc123", listed[0]["snapshot_git_commit_hash"])
        self.assertTrue(listed[0]["can_restore"])

    async def test_capture_rejects_database_parent_symlink_without_reading_victim(self) -> None:
        """The confined child must not follow `.ragtime` into another workspace."""
        victim_dir = Path(self.temp.name) / "victim" / "db"
        victim_dir.mkdir(parents=True)
        victim = victim_dir / "app.sqlite3"
        victim.write_bytes(b"victim bytes must remain private")
        ragtime_dir = self.files / ".ragtime"
        for child in ragtime_dir.iterdir():
            if child.is_dir():
                for nested in child.iterdir():
                    nested.unlink()
                child.rmdir()
        ragtime_dir.rmdir()
        ragtime_dir.symlink_to(victim_dir.parent, target_is_directory=True)

        with mock.patch("ragtime.userspace.sqlite_history.sqlite_workspace_access", self._runtime_boundary()):
            backups = await self.history.capture_workspace_databases("workspace", trigger="manual")

        self.assertEqual([], backups)
        self.assertEqual(b"victim bytes must remain private", victim.read_bytes())

    async def test_preview_and_apply_recheck_candidate_with_runtime_boundary(self) -> None:
        with mock.patch("ragtime.userspace.sqlite_history.sqlite_workspace_access", self._runtime_boundary()):
            backup = (await self.history.capture_workspace_databases("workspace", trigger="manual"))[0]
            with sqlite3.connect(self.database) as connection:
                connection.execute("UPDATE item SET value = 'after'")
            preview = await self.history.preview(
                "workspace", backup["id"], mode="overwrite", conflict_policy="keep_current", table_policies=None, user_id="owner"
            )
            result = await self.history.apply("workspace", preview["preview_id"], user_id="owner")

        with sqlite3.connect(self.database) as connection:
            restored = connection.execute("SELECT value FROM item").fetchone()[0]
        self.assertEqual("completed", result["status"])
        self.assertEqual("before", restored)

    async def test_missing_live_database_overwrite_has_no_safety_backup(self) -> None:
        with mock.patch("ragtime.userspace.sqlite_history.sqlite_workspace_access", self._runtime_boundary()):
            backup = (await self.history.capture_workspace_databases("workspace", trigger="manual"))[0]
            self.database.unlink()
            preview = await self.history.preview(
                "workspace", backup["id"], mode="overwrite", conflict_policy="keep_current", table_policies=None, user_id="owner"
            )
            result = await self.history.apply("workspace", preview["preview_id"], user_id="owner")

        self.assertIsNone(result["safety_backup_id"])
        self.assertTrue(self.database.exists())

    async def test_expired_preview_candidate_is_retained_by_restore_intent(self) -> None:
        root = self.files.parent / "sqlite_backups"
        candidate = root / "candidates" / "candidate.sqlite3"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"candidate")
        manifest = {
            "version": 1,
            "workspace_id": "workspace",
            "backups": [],
            "previews": {"preview": {"candidate": "candidates/candidate.sqlite3", "expires_at": "2000-01-01T00:00:00+00:00"}},
            "operations": {"operation": {"preview_id": "preview", "candidate": "candidates/candidate.sqlite3", "status": "intent"}},
            "last_scheduled_at": None,
        }
        self.history._save(root, manifest)
        await asyncio.to_thread(self.history._cleanup_and_due_sync, root, "workspace")
        self.assertTrue(candidate.exists())
        self.assertIn("preview", self.history._load(root, "workspace")["previews"])

    async def test_marker_without_operation_is_safe_abort_only(self) -> None:
        root = self.files.parent / "sqlite_backups"
        root.mkdir(parents=True)
        (root / "sqlite-maintenance-intent.json").write_text(json.dumps({"lease_id": "lease-1"}), encoding="utf-8")
        state = await self.history.interrupted_maintenance("workspace")
        assert state is not None
        self.assertEqual("lease-1", state["operation_id"])
        self.assertTrue(state["can_abort"])
        self.assertFalse(state["can_complete"])

    async def test_pre_restore_with_direct_files_requires_held_maintenance(self) -> None:
        async def unheld(workspace_id: str) -> None:
            raise HTTPException(status_code=423, detail="SQLite maintenance is not held")

        with mock.patch(
            "ragtime.userspace.sqlite_history.assert_sqlite_workspace_maintenance_held",
            unheld,
        ):
            with self.assertRaises(HTTPException) as error:
                await self.history.capture_workspace_databases(
                    "workspace",
                    trigger="pre_restore",
                    files_dir=self.files,
                )

        self.assertEqual(423, error.exception.status_code)

    async def test_pre_restore_with_direct_files_checks_held_maintenance_before_capture(self) -> None:
        held = mock.AsyncMock()

        with mock.patch(
            "ragtime.userspace.sqlite_history.assert_sqlite_workspace_maintenance_held",
            held,
        ):
            backups = await self.history.capture_workspace_databases(
                "workspace",
                trigger="pre_restore",
                files_dir=self.files,
            )

        held.assert_awaited_once_with("workspace")
        self.assertEqual(1, len(backups))

    async def test_scheduled_maintenance_skips_symlinked_workspace_directories(self) -> None:
        workspace_root = Path(self.temp.name) / "scheduled-workspaces"
        workspace_root.mkdir()
        outside_workspace = Path(self.temp.name) / "outside-workspace"
        outside_workspace.mkdir()
        (workspace_root / "linked-workspace").symlink_to(outside_workspace, target_is_directory=True)

        with (
            mock.patch(
                "ragtime.userspace.service.userspace_service",
                SimpleNamespace(root_path=Path(self.temp.name) / "scheduled-root"),
            ),
            mock.patch.object(self.history, "capture_workspace_databases", new_callable=mock.AsyncMock) as capture,
        ):
            (Path(self.temp.name) / "scheduled-root" / "workspaces").parent.mkdir(parents=True, exist_ok=True)
            (Path(self.temp.name) / "scheduled-root" / "workspaces").symlink_to(workspace_root, target_is_directory=True)
            await self.history.run_maintenance_once()

        capture.assert_not_awaited()

    async def test_scheduled_maintenance_warns_with_workspace_and_exception_type(self) -> None:
        workspace_root = Path(self.temp.name) / "scheduled-root" / "workspaces"
        (workspace_root / "workspace").mkdir(parents=True)

        with (
            mock.patch(
                "ragtime.userspace.service.userspace_service",
                SimpleNamespace(root_path=workspace_root.parent),
            ),
            mock.patch.object(self.history, "_root", side_effect=OSError("storage unavailable")),
            mock.patch("ragtime.userspace.sqlite_history.logger.warning") as warning,
        ):
            await self.history.run_maintenance_once()

        warning.assert_called_once()
        self.assertIn("workspace", warning.call_args.args)
        self.assertIn("OSError", warning.call_args.args)

    async def test_quota_rejection_keeps_catalog_rows_and_blobs_when_eviction_is_insufficient(self) -> None:
        """An impossible reservation must not half-evict ready history."""
        root = self.files.parent / "sqlite_backups"
        blobs = root / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "old.sqlite3").write_bytes(b"old!")
        (blobs / "new.sqlite3").write_bytes(b"newest-is-protected")
        manifest = {
            "version": 1,
            "workspace_id": "workspace",
            "previews": {},
            "operations": {},
            "backups": [
                {
                    "id": "old",
                    "database_name": "app.sqlite3",
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "status": "ready",
                    "blob": "blobs/old.sqlite3",
                    "trigger": "manual",
                },
                {
                    "id": "new",
                    "database_name": "app.sqlite3",
                    "created_at": "2024-01-02T00:00:00+00:00",
                    "status": "ready",
                    "blob": "blobs/new.sqlite3",
                    "trigger": "manual",
                },
            ],
        }
        with mock.patch("ragtime.userspace.sqlite_history._MAX_WORKSPACE_BYTES", len(b"newest-is-protected")):
            with self.assertRaises(HTTPException) as error:
                await asyncio.to_thread(self.history._enforce_quota, root, manifest, 1)

        self.assertEqual(409, error.exception.status_code)
        backups = cast(list[dict[str, object]], manifest["backups"])
        self.assertEqual(["old", "new"], [row["id"] for row in backups])
        self.assertTrue((blobs / "old.sqlite3").exists())

    async def test_delete_failed_capture_without_blob_removes_only_diagnostic_row(self) -> None:
        root = self.files.parent / "sqlite_backups"
        self.history._save(
            root,
            {
                "version": 1,
                "workspace_id": "workspace",
                "previews": {},
                "operations": {},
                "backups": [{"id": "failed", "database_name": "app.sqlite3", "created_at": "2024-01-01T00:00:00+00:00", "status": "failed", "blob": None}],
            },
        )

        await self.history.delete("workspace", "failed")

        self.assertEqual([], self.history._load(root, "workspace")["backups"])

    async def test_apply_returns_creator_receipt_before_runtime_access_after_preview_removed(self) -> None:
        root = self.files.parent / "sqlite_backups"
        expected = {"operation_id": "done", "status": "completed", "runtime_stopped": True}
        self.history._save(
            root,
            {
                "version": 1,
                "workspace_id": "workspace",
                "backups": [],
                "previews": {},
                "operations": {"done": {"preview_id": "expired-preview", "status": "completed", "user_id": "owner", "result": expected}},
            },
        )

        @asynccontextmanager
        async def unexpected_access(*args, **kwargs):
            raise AssertionError("receipt retry must not acquire runtime access")
            yield self.files

        with mock.patch("ragtime.userspace.sqlite_history.sqlite_workspace_access", unexpected_access):
            result = await self.history.apply("workspace", "expired-preview", user_id="owner")

        self.assertEqual(expected, result)

    async def test_preview_reservation_rejects_before_creating_candidate(self) -> None:
        import hashlib

        root = self.files.parent / "sqlite_backups"
        blobs = root / "blobs"
        blobs.mkdir(parents=True)
        backup = blobs / "backup.sqlite3"
        backup.write_bytes(b"backup")
        self.history._save(
            root,
            {
                "version": 1,
                "workspace_id": "workspace",
                "operations": {},
                "previews": {},
                "backups": [
                    {
                        "id": "backup",
                        "database_name": "app.sqlite3",
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "status": "ready",
                        "blob": "blobs/backup.sqlite3",
                        "sha256": hashlib.sha256(b"backup").hexdigest(),
                        "trigger": "manual",
                    }
                ],
            },
        )
        with (
            mock.patch("ragtime.userspace.sqlite_history.sqlite_workspace_access", self._runtime_boundary()),
            mock.patch.object(self.history, "_database_size_estimate", return_value=8),
            mock.patch("ragtime.userspace.sqlite_history._MAX_WORKSPACE_BYTES", 10),
            mock.patch.object(self.history, "_preview_confined") as preview_child,
        ):
            with self.assertRaises(HTTPException) as error:
                await self.history.preview("workspace", "backup", mode="overwrite", conflict_policy="keep_current", table_policies=None, user_id="owner")

        self.assertEqual(409, error.exception.status_code)
        preview_child.assert_not_called()
        self.assertFalse((root / "candidates").exists())

    async def test_download_reservation_rejects_before_copy_growth(self) -> None:
        import hashlib

        root = self.files.parent / "sqlite_backups"
        blobs = root / "blobs"
        blobs.mkdir(parents=True)
        backup = blobs / "backup.sqlite3"
        backup.write_bytes(b"backup")
        self.history._save(
            root,
            {
                "version": 1,
                "workspace_id": "workspace",
                "operations": {},
                "previews": {},
                "backups": [
                    {
                        "id": "backup",
                        "database_name": "app.sqlite3",
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "status": "ready",
                        "blob": "blobs/backup.sqlite3",
                        "sha256": hashlib.sha256(b"backup").hexdigest(),
                        "trigger": "manual",
                    }
                ],
            },
        )

        with mock.patch("ragtime.userspace.sqlite_history._MAX_WORKSPACE_BYTES", 10):
            with self.assertRaises(HTTPException) as error:
                await self.history.download_path("workspace", "backup")

        self.assertEqual(409, error.exception.status_code)
        self.assertFalse((root / "downloads").exists())

    async def test_two_catalog_instances_claim_one_scheduled_run(self) -> None:
        root = self.files.parent / "sqlite_backups"
        other = SqliteHistoryService(lambda workspace_id: self.files)
        self.history._save(
            root,
            {
                "version": 1,
                "workspace_id": "workspace",
                "backups": [],
                "previews": {},
                "operations": {},
                "last_scheduled_at": None,
                "next_scheduled_at": "2000-01-01T00:00:00+00:00",
            },
        )
        claims = await asyncio.gather(
            asyncio.to_thread(self.history._claim_scheduled_due_sync, root, "workspace"),
            asyncio.to_thread(other._claim_scheduled_due_sync, root, "workspace"),
        )

        claim_ids = [claim for claim in claims if claim is not None]
        self.assertEqual(1, len(claim_ids))
        self.assertIsInstance(claim_ids[0], str)
        self.assertEqual(claim_ids[0], self.history._load(root, "workspace")["scheduled_claim"]["claim_id"])

    async def test_protected_history_rejects_symlink_and_special_leafs(self) -> None:
        root = self.files.parent / "sqlite_backups"
        blobs = root / "blobs"
        blobs.mkdir(parents=True)
        target = root / "outside.sqlite3"
        target.write_bytes(b"outside")
        (blobs / "linked.sqlite3").symlink_to(target)
        (blobs / "directory.sqlite3").mkdir()

        for name in ("linked.sqlite3", "directory.sqlite3"):
            with self.assertRaises(HTTPException) as error:
                self.history._protected_path(root, f"blobs/{name}", "blobs")
            self.assertEqual(409, error.exception.status_code)

    async def test_catalog_manifest_and_lock_reject_unsafe_leafs(self) -> None:
        root = self.files.parent / "sqlite_backups"
        root.mkdir()
        outside = self.files.parent / "outside-manifest.json"
        outside.write_text("{}", encoding="utf-8")
        (root / "manifest-v1.json").symlink_to(outside)

        with self.assertRaises(HTTPException) as manifest_error:
            await self.history.list_backups("workspace")
        self.assertEqual(409, manifest_error.exception.status_code)

        (root / "manifest-v1.json").unlink()
        (root / ".lock").unlink()
        (root / ".lock").mkdir()
        with self.assertRaises(HTTPException) as lock_error:
            await self.history.list_backups("workspace")
        self.assertEqual(409, lock_error.exception.status_code)

    async def test_pre_restore_capture_failure_retains_durable_marker_for_recovery(self) -> None:
        """When pre_restore capture fails, the marker remains for abort-only recovery."""
        # Simulate a durable maintenance marker left by incomplete apply.
        root = self.files.parent / "sqlite_backups"
        root.mkdir(parents=True)
        marker = root / "sqlite-maintenance-intent.json"
        lease_id = "lease-123"
        marker.write_text(json.dumps({"lease_id": lease_id}), encoding="utf-8")

        # Attempting to recover without an operation record should be abort-only.
        state = await self.history.interrupted_maintenance("workspace")
        assert state is not None
        self.assertEqual(lease_id, state["operation_id"])
        self.assertTrue(state["can_abort"])
        self.assertFalse(state["can_complete"])

        # Marker persists after recovery state check.
        self.assertTrue(marker.exists())

    async def test_pre_restore_capture_failure_allows_abort_recovery(self) -> None:
        """Abort action on marker-only recovery calls recovery maintenance."""
        root = self.files.parent / "sqlite_backups"
        root.mkdir(parents=True)
        marker = root / "sqlite-maintenance-intent.json"
        lease_id = "lease-456"
        marker.write_text(json.dumps({"lease_id": lease_id}), encoding="utf-8")

        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ) as recover_mock:
            result = await self.history.recover_operation("workspace", lease_id, action="abort")

        self.assertEqual("aborted", result["status"])
        self.assertEqual(lease_id, result["operation_id"])
        recover_mock.assert_awaited_once_with("workspace", lease_id, action="abort")

    def _seed_intent_operation(self, *, published: bool) -> tuple[Path, str, str]:
        """Create a marker + intent operation with a verifiable candidate.

        Returns (root, operation_id, lease_id).  When ``published`` the live
        database is already the restored candidate, matching the state left by a
        completed-but-not-finalized apply.
        """
        import hashlib
        import shutil as _shutil

        root = self.files.parent / "sqlite_backups"
        candidate_dir = root / "candidates"
        candidate_dir.mkdir(parents=True)
        candidate = candidate_dir / "candidate.sqlite3"
        with sqlite3.connect(candidate) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item(value) VALUES ('restored')")
        candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if published:
            _shutil.copyfile(candidate, self.database)
        lease_id = "lease-crash"
        operation_id = "op-crash"
        manifest = {
            "version": 1,
            "workspace_id": "workspace",
            "backups": [],
            "previews": {
                "preview": {
                    "backup_id": "backup-1",
                    "database_name": "app.sqlite3",
                    "candidate": "candidates/candidate.sqlite3",
                    "candidate_sha256": candidate_sha,
                }
            },
            "operations": {
                operation_id: {
                    "preview_id": "preview",
                    "status": "intent",
                    "candidate": "candidates/candidate.sqlite3",
                    "safety_backup_id": "safety-1",
                    "publication_state": "published" if published else "prepublication",
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            },
            "last_scheduled_at": None,
        }
        self.history._save(root, manifest)
        (root / "sqlite-maintenance-intent.json").write_text(json.dumps({"lease_id": lease_id}), encoding="utf-8")
        return root, operation_id, lease_id

    async def test_complete_persists_terminal_receipt_before_marker_release(self) -> None:
        """recover_operation(complete) must make the receipt terminal before release."""
        root, operation_id, lease_id = self._seed_intent_operation(published=True)
        observed_status: list[str] = []

        async def capture_status(workspace_id: str, marker_lease: str, *, action: str) -> None:
            # The catalog receipt must already be terminal when release runs.
            manifest = self.history._load(root, "workspace")
            observed_status.append(manifest["operations"][operation_id]["status"])

        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            side_effect=capture_status,
        ):
            result = await self.history.recover_operation("workspace", operation_id, action="complete")

        self.assertEqual("completed", result["status"])
        self.assertEqual(["completed"], observed_status)
        manifest = self.history._load(root, "workspace")
        self.assertEqual("completed", manifest["operations"][operation_id]["status"])

    async def test_complete_release_failure_leaves_terminal_receipt_and_retry_finishes(self) -> None:
        """A release failure keeps the receipt terminal; retry re-runs release only."""
        root, operation_id, lease_id = self._seed_intent_operation(published=True)
        live_before = self.database.read_bytes()

        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            side_effect=RuntimeError("release failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                await self.history.recover_operation("workspace", operation_id, action="complete")

        # Receipt is durably terminal despite the release failure.
        manifest = self.history._load(root, "workspace")
        self.assertEqual("completed", manifest["operations"][operation_id]["status"])
        # The marker still holds the fence, so the banner must surface a
        # release-retry of the SAME (complete) action rather than vanish or flip
        # to an abort affordance.
        state = await self.history.interrupted_maintenance("workspace")
        assert state is not None
        self.assertEqual(operation_id, state["operation_id"])
        self.assertTrue(state["can_complete"])
        self.assertFalse(state["can_abort"])

        # Retry must NOT republish (live db untouched) and must re-run release.
        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ) as recover_mock:
            result = await self.history.recover_operation("workspace", operation_id, action="complete")

        self.assertEqual("completed", result["status"])
        recover_mock.assert_awaited_once_with("workspace", lease_id, action="complete")
        self.assertEqual(live_before, self.database.read_bytes())

    async def test_completed_operation_rejects_opposite_abort_action(self) -> None:
        """A completed operation cannot be aborted on retry."""
        root, operation_id, lease_id = self._seed_intent_operation(published=True)
        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ):
            await self.history.recover_operation("workspace", operation_id, action="complete")

        # Re-seed the marker to simulate an unreleased fence for the retry.
        (root / "sqlite-maintenance-intent.json").write_text(json.dumps({"lease_id": lease_id}), encoding="utf-8")
        with self.assertRaises(HTTPException) as error:
            await self.history.recover_operation("workspace", operation_id, action="abort")
        self.assertEqual(409, error.exception.status_code)
        self.assertIn("opposite", error.exception.detail)

    async def test_abort_persists_terminal_receipt_before_release_and_retry_is_safe(self) -> None:
        """Abort persists a terminal receipt before release; retry re-runs release only."""
        root, operation_id, lease_id = self._seed_intent_operation(published=False)

        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            side_effect=RuntimeError("release failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                await self.history.recover_operation("workspace", operation_id, action="abort")

        manifest = self.history._load(root, "workspace")
        self.assertEqual("aborted", manifest["operations"][operation_id]["status"])
        # The fence marker still needs releasing, so the banner surfaces a
        # release-retry of the SAME (abort) action, not a stuck intent.
        state = await self.history.interrupted_maintenance("workspace")
        assert state is not None
        self.assertEqual(operation_id, state["operation_id"])
        self.assertTrue(state["can_abort"])
        self.assertFalse(state["can_complete"])

        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ) as recover_mock:
            result = await self.history.recover_operation("workspace", operation_id, action="abort")

        self.assertEqual("aborted", result["status"])
        recover_mock.assert_awaited_once_with("workspace", lease_id, action="abort")

        # Once the release succeeds the marker is gone and no banner remains.
        (root / "sqlite-maintenance-intent.json").unlink(missing_ok=True)
        self.assertIsNone(await self.history.interrupted_maintenance("workspace"))

    async def test_aborted_operation_rejects_opposite_complete_action(self) -> None:
        """An aborted operation cannot be completed on retry."""
        root, operation_id, lease_id = self._seed_intent_operation(published=False)
        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ):
            await self.history.recover_operation("workspace", operation_id, action="abort")

        (root / "sqlite-maintenance-intent.json").write_text(json.dumps({"lease_id": lease_id}), encoding="utf-8")
        with self.assertRaises(HTTPException) as error:
            await self.history.recover_operation("workspace", operation_id, action="complete")
        self.assertEqual(409, error.exception.status_code)
        self.assertIn("opposite", error.exception.detail)

    async def test_terminal_receipt_with_released_marker_retry_returns_result_without_release(self) -> None:
        """After a successful release, a redundant retry returns the receipt and skips release."""
        root, operation_id, lease_id = self._seed_intent_operation(published=True)
        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ):
            first = await self.history.recover_operation("workspace", operation_id, action="complete")
        # Simulate the crash-after-release state: marker gone, receipt terminal.
        marker = root / "sqlite-maintenance-intent.json"
        marker.unlink(missing_ok=True)

        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ) as recover_mock:
            second = await self.history.recover_operation("workspace", operation_id, action="complete")

        self.assertEqual(first["status"], second["status"])
        recover_mock.assert_not_awaited()

    async def test_terminal_retry_never_releases_a_foreign_maintenance_marker(self) -> None:
        """A stale retry must not tear down a fence held by an unrelated lease."""
        root, operation_id, lease_id = self._seed_intent_operation(published=True)
        # Drive the operation terminal, releasing its own fence.
        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ):
            await self.history.recover_operation("workspace", operation_id, action="complete")

        # A DIFFERENT maintenance holder now legitimately owns the fence.
        marker = root / "sqlite-maintenance-intent.json"
        marker.write_text(json.dumps({"lease_id": "foreign-lease"}), encoding="utf-8")

        # A replayed recover for the already-terminal operation must return its
        # receipt WITHOUT releasing the foreign holder's marker or lease.
        with mock.patch(
            "ragtime.userspace.sqlite_history.recover_sqlite_workspace_maintenance",
            new_callable=mock.AsyncMock,
        ) as recover_mock:
            result = await self.history.recover_operation("workspace", operation_id, action="complete")

        self.assertEqual("completed", result["status"])
        recover_mock.assert_not_awaited()
        # The foreign marker is untouched.
        self.assertEqual("foreign-lease", json.loads(marker.read_text())["lease_id"])

    async def test_interrupted_maintenance_surfaces_unsafe_symlink_marker(self) -> None:
        """A symlinked marker must surface as unrecoverable, not vanish from the UI."""
        root = self.files.parent / "sqlite_backups"
        root.mkdir(parents=True)
        target = root / "foreign-marker.json"
        target.write_text(json.dumps({"lease_id": "foreign"}), encoding="utf-8")
        (root / "sqlite-maintenance-intent.json").symlink_to(target)

        state = await self.history.interrupted_maintenance("workspace")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual("unknown", state["operation_id"])
        self.assertFalse(state["can_complete"])
        self.assertFalse(state["can_abort"])

    async def test_interrupted_maintenance_surfaces_malformed_marker(self) -> None:
        """A malformed marker must surface as unrecoverable rather than disappear."""
        root = self.files.parent / "sqlite_backups"
        root.mkdir(parents=True)
        (root / "sqlite-maintenance-intent.json").write_text("{ not json", encoding="utf-8")

        state = await self.history.interrupted_maintenance("workspace")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual("unknown", state["operation_id"])
        self.assertFalse(state["can_complete"])
        self.assertFalse(state["can_abort"])


class SqliteHistoryApplyFencePreIntentTests(unittest.IsolatedAsyncioTestCase):
    """Exercise apply() against the REAL offline runtime fence.

    These tests prove that a pre-intent validation failure (stale preview)
    releases the durable maintenance marker, while a publication failure raised
    after the restore intent is recorded retains it (fail-closed).
    """

    async def asyncSetUp(self) -> None:
        import hashlib

        self.temp = tempfile.TemporaryDirectory()
        data = Path(self.temp.name)
        # Layout must match sqlite_runtime._workspace_dir so the runtime marker
        # and SqliteHistoryService._root resolve to the SAME sqlite_backups dir.
        self.workspace_id = "workspace-1"
        workspace_dir = data / "_userspace" / "workspaces" / self.workspace_id
        self.files = workspace_dir / "files"
        db_dir = self.files / ".ragtime" / "db"
        db_dir.mkdir(parents=True)
        self.database = db_dir / "app.sqlite3"
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item(value) VALUES ('live')")

        self.root = workspace_dir / "sqlite_backups"
        self.marker = self.root / "sqlite-maintenance-intent.json"
        self.history = SqliteHistoryService(lambda workspace_id: self.files)

        # A candidate blob whose sha the preview will reference.
        candidate_dir = self.root / "candidates"
        candidate_dir.mkdir(parents=True)
        self.candidate = candidate_dir / "candidate.sqlite3"
        with sqlite3.connect(self.candidate) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item(value) VALUES ('restored')")
        candidate_sha = hashlib.sha256(self.candidate.read_bytes()).hexdigest()

        from runtime.core.sqlite_recovery import database_fingerprint, migration_fingerprint

        migrations = self.files / ".ragtime" / "db" / "migrations"
        self._preview_common = {
            "backup_id": "backup-1",
            "database_name": "app.sqlite3",
            "candidate": "candidates/candidate.sqlite3",
            "candidate_sha256": candidate_sha,
            "migration_fingerprint": migration_fingerprint(migrations),
            "mode": "overwrite",
            "conflict_policy": "keep_current",
            "table_policies": {},
            "user_id": "owner",
        }
        self._live_fingerprint = database_fingerprint(self.database)

        from ragtime.userspace import sqlite_runtime

        self._sqlite_runtime = sqlite_runtime
        self._settings_patch = mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data))
        self._manager_patch = mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=False)
        self._boundary_patch = mock.patch(
            "ragtime.userspace.sqlite_history.sqlite_workspace_access",
            sqlite_runtime.sqlite_workspace_access,
        )
        self._settings_patch.start()
        self._manager_patch.start()
        self._boundary_patch.start()

    async def asyncTearDown(self) -> None:
        self._boundary_patch.stop()
        self._manager_patch.stop()
        self._settings_patch.stop()
        self.temp.cleanup()

    def _seed_preview(self, *, current_fingerprint: str) -> str:
        preview_id = "preview-1"
        manifest = {
            "version": 1,
            "workspace_id": self.workspace_id,
            "backups": [],
            "previews": {preview_id: {**self._preview_common, "current_fingerprint": current_fingerprint, "expires_at": (_dt_now() + _dt_delta()).isoformat()}},
            "operations": {},
            "last_scheduled_at": None,
        }
        self.history._save(self.root, manifest)
        return preview_id

    async def test_stale_preview_returns_409_and_releases_fence(self) -> None:
        # A fingerprint that will not match the live database forces the
        # pre-intent "database changed" rejection.
        preview_id = self._seed_preview(current_fingerprint="stale-fingerprint")
        with self.assertRaises(HTTPException) as error:
            await self.history.apply(self.workspace_id, preview_id, user_id="owner")
        self.assertEqual(409, error.exception.status_code)
        # The fence marker must be gone: the offline context exited normally.
        self.assertFalse(self.marker.exists())
        self.assertFalse(os.path.lexists(self.marker))
        # No intent operation should have been recorded.
        self.assertEqual({}, self.history._load(self.root, self.workspace_id)["operations"])

    async def test_publication_failure_retains_fence(self) -> None:
        # A matching fingerprint passes pre-intent checks, records an intent,
        # then a publication copy failure must cross the maintenance boundary.
        preview_id = self._seed_preview(current_fingerprint=self._live_fingerprint)
        with mock.patch(
            "ragtime.userspace.sqlite_history.publish_regular_file",
            side_effect=OSError("publish failed"),
        ):
            with self.assertRaises(OSError):
                await self.history.apply(self.workspace_id, preview_id, user_id="owner")
        # Fail-closed: the durable marker must remain for operator recovery.
        self.assertTrue(os.path.lexists(self.marker))
        # An intent operation exists and is recoverable.
        operations = self.history._load(self.root, self.workspace_id)["operations"]
        self.assertEqual(1, len(operations))
        self.assertEqual("intent", next(iter(operations.values()))["status"])

    async def test_completed_apply_release_failure_surfaces_complete_retry(self) -> None:
        preview_id = self._seed_preview(current_fingerprint=self._live_fingerprint)

        with mock.patch.object(self._sqlite_runtime, "_remove_owned_marker", return_value=False):
            with self.assertRaises(HTTPException) as error:
                await self.history.apply(self.workspace_id, preview_id, user_id="owner")

        self.assertEqual(423, error.exception.status_code)
        operation_id, operation = next(iter(self.history._load(self.root, self.workspace_id)["operations"].items()))
        self.assertEqual("completed", operation["status"])
        state = await self.history.interrupted_maintenance(self.workspace_id)
        assert state is not None
        self.assertEqual(operation_id, state["operation_id"])
        self.assertTrue(state["can_complete"])
        self.assertFalse(state["can_abort"])

    async def test_crash_after_replace_is_publishing_abort_rejected_complete_clears_sidecars(self) -> None:
        # A crash AFTER os.replace overwrites the live database but BEFORE the
        # terminal save.  The durable state must be 'publishing' so the restore
        # can only be completed (never aborted), and completion must clear the
        # stale WAL/SHM sidecars from the previous database.
        preview_id = self._seed_preview(current_fingerprint=self._live_fingerprint)
        # Stale sidecars from the pre-restore database that completion must clear.
        wal = self.database.with_name(self.database.name + "-wal")
        shm = self.database.with_name(self.database.name + "-shm")
        wal.write_bytes(b"stale-wal")
        shm.write_bytes(b"stale-shm")

        from runtime.core.secure_files import publish_regular_file

        def crash_after_target_publish(*args, **kwargs):
            publish_regular_file(*args, **kwargs)
            raise RuntimeError("crash after replace before terminal save")

        with mock.patch("ragtime.userspace.sqlite_history.publish_regular_file", side_effect=crash_after_target_publish):
            with self.assertRaises(RuntimeError):
                await self.history.apply(self.workspace_id, preview_id, user_id="owner")

        # Durable state is 'publishing'; live DB already replaced; marker retained.
        operations = self.history._load(self.root, self.workspace_id)["operations"]
        self.assertEqual(1, len(operations))
        operation_id, operation = next(iter(operations.items()))
        self.assertEqual("intent", operation["status"])
        self.assertEqual("publishing", operation["publication_state"])
        self.assertTrue(os.path.lexists(self.marker))
        with sqlite3.connect(self.database) as connection:
            self.assertEqual("restored", connection.execute("SELECT value FROM item").fetchone()[0])

        # The UI must require completion, not offer abort.
        state = await self.history.interrupted_maintenance(self.workspace_id)
        assert state is not None
        self.assertEqual(operation_id, state["operation_id"])
        self.assertTrue(state["can_complete"])
        self.assertFalse(state["can_abort"])

        # Abort must be rejected for a publishing operation.
        with self.assertRaises(HTTPException) as rejected:
            await self.history.recover_operation(self.workspace_id, operation_id, action="abort")
        self.assertEqual(409, rejected.exception.status_code)

        # Completion finishes the restore and clears the stale sidecars.
        result = await self.history.recover_operation(self.workspace_id, operation_id, action="complete")
        self.assertEqual("completed", result["status"])
        self.assertFalse(wal.exists())
        self.assertFalse(shm.exists())
        self.assertFalse(os.path.lexists(self.marker))
        with sqlite3.connect(self.database) as connection:
            self.assertEqual("restored", connection.execute("SELECT value FROM item").fetchone()[0])

    async def test_recovery_rejects_live_operation_holder_before_receipt_publication(self) -> None:
        """The real exclusive flock gates all recovery mutation and publication."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.marker.write_text(json.dumps({"lease_id": "live-lease", "origin": "offline"}), encoding="utf-8")
        ready = asyncio.Event()
        release = asyncio.Event()

        async def hold_operation() -> None:
            async with self._sqlite_runtime._sqlite_workspace_operation(self.workspace_id, exclusive=True):
                ready.set()
                await release.wait()

        holder = asyncio.create_task(hold_operation())
        await ready.wait()
        try:
            with self.assertRaises(HTTPException) as error:
                await self.history.recover_operation(self.workspace_id, "live-lease", action="abort")
            self.assertEqual(423, error.exception.status_code)
            self.assertEqual({"lease_id": "live-lease", "origin": "offline"}, json.loads(self.marker.read_text()))
            self.assertEqual({}, self.history._load(self.root, self.workspace_id)["operations"])
        finally:
            release.set()
            await holder


def _dt_now():
    return datetime.now(timezone.utc)


def _dt_delta():
    return timedelta(minutes=15)
