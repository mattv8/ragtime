"""Linux integration coverage for the active runtime SQLite-history HTTP surface."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI

from runtime.worker.sqlite_history.api import history_router
from runtime.worker.sqlite_history.coordinator import SqliteHistoryCoordinator

_RESTIC = Path("/opt/ragtime-backup/bin/restic")
# Real Restic initialization/verification can exceed 10s on a loaded CI runner.
_CAPTURE_TERMINAL_TIMEOUT_SECONDS = 60


class _WorkerLeaseBoundary:
    """Select the real temp workspace tree while recording lease ownership."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.active: dict[tuple[str, str], bool] = {}

    async def acquire_sqlite_workspace_access(self, workspace_id: str, lease_id: str, *, maintenance: bool) -> dict[str, str]:
        key = (workspace_id, lease_id)
        if key in self.active:
            raise RuntimeError("duplicate worker lease")
        self.active[key] = maintenance
        files = self.root / "workspaces" / workspace_id / "files"
        return {"authoritative_root": str(files)}

    async def release_sqlite_workspace_access(self, workspace_id: str, lease_id: str) -> None:
        self.active.pop((workspace_id, lease_id), None)


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux Landlock and packaged Restic")
class RuntimeSqliteHistoryRouteIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.assertTrue(_RESTIC.is_file(), "runtime image must package Restic at the production path")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace_id = "workspace-e2e"
        self.files = self.root / "workspaces" / self.workspace_id / "files"
        self.database_dir = self.files / ".ragtime" / "db"
        self.database_dir.mkdir(parents=True)
        self.database = self.database_dir / "app.sqlite3"
        self._write_database("captured")
        self.worker = _WorkerLeaseBoundary(self.root)
        self.coordinator = SqliteHistoryCoordinator(self.root, self.worker)
        app = FastAPI()
        app.include_router(history_router("/worker", None, lambda: self.coordinator))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://runtime.test")
        activation = await self.client.post("/worker/sqlite-history/activation")
        self.assertEqual(201, activation.status_code, activation.text)

    async def asyncTearDown(self) -> None:
        await self.coordinator.shutdown()
        await self.client.aclose()
        self.temporary.cleanup()

    def _write_database(self, value: str) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS item (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("DELETE FROM item")
            connection.execute("INSERT INTO item(id, value) VALUES (1, ?)", (value,))

    def _rows(self, path: Path | str) -> list[tuple[int, str]]:
        with sqlite3.connect(path) as connection:
            return connection.execute("SELECT id, value FROM item ORDER BY id").fetchall()

    async def test_inspector_import_preserves_old_database_without_nested_fence(self) -> None:
        await self._capture("manual")
        upload = self.root / "upload.sqlite3"
        with sqlite3.connect(upload) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO item VALUES (1, 'uploaded')")
        response = await asyncio.wait_for(
            self.client.post(
                f"/worker/workspaces/{self.workspace_id}/sqlite-history/import-database/app.sqlite3",
                headers={"X-Ragtime-Creator-User": "local:owner", "Content-Type": "application/vnd.sqlite3"},
                content=upload.read_bytes(),
            ),
            timeout=60,
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual([(1, "uploaded")], self._rows(self.database))
        safety_id = response.json()["receipt"]["safety_backup_id"]
        self.assertTrue(safety_id)
        safety = await self.client.get(f"/worker/workspaces/{self.workspace_id}/sqlite-history/backups/{safety_id}/download")
        self.assertEqual(200, safety.status_code, safety.text)
        restored = self.root / "pre-import.sqlite3"
        restored.write_bytes(safety.content)
        self.assertEqual([(1, "captured")], self._rows(restored))
        self.assertEqual({}, self.worker.active)

    async def _capture(self, trigger: str) -> dict:
        operation_id = str(uuid4())
        response = await self.client.post(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history/captures",
            json={
                "operation_id": operation_id,
                "creator_id": "local:owner",
                "trigger": trigger,
                "database_names": ["app.sqlite3"],
            },
        )
        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual("accepted", response.json()["phase"])
        last_receipt: dict | None = None
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        try:
            async with asyncio.timeout(_CAPTURE_TERMINAL_TIMEOUT_SECONDS):
                while True:
                    observed = await self.client.get(f"/worker/workspaces/{self.workspace_id}/sqlite-history/captures/{operation_id}")
                    self.assertEqual(200, observed.status_code, observed.text)
                    receipt = observed.json()
                    last_receipt = receipt
                    if receipt["phase"] in {"completed", "failed", "cancelled", "interrupted"}:
                        self.assertEqual("completed", receipt["phase"], receipt)
                        return receipt
                    await asyncio.sleep(0.1)
        except TimeoutError:
            elapsed = loop.time() - started_at
            self.fail(
                f"capture {operation_id} did not become terminal after {elapsed:.1f}s; "
                f"last phase={last_receipt.get('phase') if last_receipt else None}; "
                f"receipt={last_receipt}"
            )

    async def test_active_routes_capture_receipt_restic_download_and_safe_apply(self) -> None:
        manual = await self._capture("manual")
        manual_result = manual["database_outcomes"]["app.sqlite3"]["results"][0]
        self.assertEqual("ready", manual_result["status"])
        manifest = json.loads((self.root / "workspaces" / self.workspace_id / "sqlite_backups" / "manifest-v1.json").read_text())
        stored = next(row for row in manifest["backups"] if row["id"] == manual_result["id"])
        self.assertEqual("restic", stored["storage"]["kind"])
        self.assertEqual(64, len(stored["storage"]["snapshot_id"]))

        scheduled = await self._capture("scheduled")
        scheduled_result = scheduled["database_outcomes"]["app.sqlite3"]["results"][0]
        self.assertEqual("skipped_unchanged", scheduled_result["outcome"])

        history = await self.client.get(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history",
            params={"database_name": "app.sqlite3"},
        )
        self.assertEqual(200, history.status_code, history.text)
        backups = history.json()["backups"]
        self.assertEqual(1, len(backups))
        backup_id = backups[0]["id"]

        download = await self.client.get(f"/worker/workspaces/{self.workspace_id}/sqlite-history/backups/{backup_id}/download")
        self.assertEqual(200, download.status_code, download.text)
        downloaded = self.root / "downloaded.sqlite3"
        downloaded.write_bytes(download.content)
        self.assertEqual([(1, "captured")], self._rows(downloaded))

        self._write_database("mutated")
        migration_dir = self.database_dir / "migrations"
        migration_dir.mkdir()
        migration = migration_dir / "001_noop.sql"
        migration.write_text("CREATE TABLE IF NOT EXISTS migration_probe (id INTEGER PRIMARY KEY);\n", encoding="utf-8")

        preview = await self.client.post(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history/backups/{backup_id}/preview",
            json={
                "user_id": "local:owner",
                "mode": "overwrite",
                "conflict_policy": "keep_current",
                "table_policies": {},
            },
        )
        self.assertEqual(200, preview.status_code, preview.text)
        preview_payload = preview.json()
        self.assertTrue(preview_payload["can_apply"], preview_payload)
        preview_id = preview_payload["preview_id"]

        original_migration = migration.read_bytes()
        migration.write_text("CREATE TABLE changed_probe (id INTEGER PRIMARY KEY);\n", encoding="utf-8")
        rejected = await self.client.post(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history/previews/{preview_id}/apply",
            json={"user_id": "local:owner"},
        )
        self.assertEqual(409, rejected.status_code, rejected.text)
        self.assertEqual([(1, "mutated")], self._rows(self.database))
        marker = self.root / "workspaces" / self.workspace_id / "sqlite_backups" / "sqlite-maintenance-intent.json"
        self.assertFalse(marker.exists(), "pre-publication rejection must release its maintenance marker")

        migration.write_bytes(original_migration)
        applied = await self.client.post(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history/previews/{preview_id}/apply",
            json={"user_id": "local:owner"},
        )
        self.assertEqual(200, applied.status_code, applied.text)
        self.assertEqual("completed", applied.json()["status"])
        self.assertEqual([(1, "captured")], self._rows(self.database))

        activation_file = self.root / "_sqlite_history" / "activation-v1.json"
        self.assertEqual(hashlib.sha256(b'{"version":2,"active":true}').hexdigest(), hashlib.sha256(activation_file.read_bytes()).hexdigest())
        restarted = SqliteHistoryCoordinator(self.root, self.worker)
        self.assertTrue(restarted.capability())

    async def test_guarded_restore_binds_creator_and_recovers_with_new_service(self) -> None:
        operation_id = str(uuid4())
        begin = await self.client.post(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history/guarded-code-restores/begin",
            json={"operation_id": operation_id, "user_id": "local:owner"},
        )
        self.assertEqual(200, begin.status_code, begin.text)
        self._write_database("git-mutated")

        finished = await self.client.post(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history/guarded-code-restores/{operation_id}/finish",
            json={"user_id": "local:owner", "git_error": "checkout failed"},
        )
        self.assertEqual(200, finished.status_code, finished.text)
        self.assertEqual([(1, "captured")], self._rows(self.database))

        recovery_id = str(uuid4())
        begun = await self.client.post(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history/guarded-code-restores/begin",
            json={"operation_id": recovery_id, "user_id": "local:owner"},
        )
        self.assertEqual(200, begun.status_code, begun.text)
        self._write_database("crash-mutated")

        old_service = self.coordinator._service()
        lifecycle = old_service._guarded_restores[recovery_id]["task"]
        lifecycle.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await lifecycle
        old_service._guarded_restores.clear()

        await self.client.aclose()
        self.coordinator = SqliteHistoryCoordinator(self.root, self.worker)
        app = FastAPI()
        app.include_router(history_router("/worker", None, lambda: self.coordinator))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://runtime.test")

        recovered = await self.client.post(
            f"/worker/workspaces/{self.workspace_id}/sqlite-history/recover/{recovery_id}",
            json={"action": "complete"},
        )
        self.assertEqual(200, recovered.status_code, recovered.text)
        self.assertTrue(recovered.json()["recovered"])
        self.assertEqual([(1, "captured")], self._rows(self.database))
        self.assertFalse(self.worker.active)
