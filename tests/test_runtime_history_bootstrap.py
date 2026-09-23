from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import queue
import tempfile
import unittest
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from fastapi.routing import APIRoute

from runtime.worker.sqlite_history.bootstrap import BootstrapManager
from runtime.worker.sqlite_history.bootstrap_api import bootstrap_router
from runtime.worker.sqlite_history.coordinator import SqliteHistoryCoordinator
from runtime.worker.sqlite_history.models import RESTIC_IMAGE_PATH
from runtime.worker.sqlite_history.repository import ResticRepository
from runtime.worker.sqlite_history.service import RuntimeSqliteHistoryService


class _Repository:
    def __init__(self) -> None:
        self.verified: list[str] = []

    async def verify(self, artifact: Any, *, pass_fds: tuple[int, ...] = ()) -> None:
        self.verified.append(artifact.snapshot_id)


class _History:
    def __init__(self, inventory: "_Inventory") -> None:
        self.inventory = inventory
        self.repository = _Repository()
        self.calls: list[str] = []
        self.failures: dict[str, int] = {}
        self.entered: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    async def migrate_legacy_backups(
        self,
        workspace_id: str,
        *,
        pass_fds: tuple[int, ...] = (),
        cancel_check: Any = None,
    ) -> int:
        self.calls.append(workspace_id)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if cancel_check is not None and await cancel_check():
            return 0
        remaining = self.failures.get(workspace_id, 0)
        if remaining:
            self.failures[workspace_id] = remaining - 1
            raise RuntimeError("bounded converter failure")
        self.inventory.converted.add(workspace_id)
        return 1

    @contextlib.asynccontextmanager
    async def _installed(self):
        yield

    async def list_backups(self, workspace_id: str, **_filters: Any) -> list[dict[str, Any]]:
        if workspace_id not in self.inventory.converted:
            return []
        return [
            {
                "id": f"backup-{workspace_id}",
                "status": "ready",
                "size_bytes": 7,
                "sha256": "a" * 64,
                "storage": {
                    "kind": "restic",
                    "repository_id": "b" * 64,
                    "snapshot_id": (workspace_id[0] * 64),
                    "path": RESTIC_IMAGE_PATH,
                },
            }
        ]


class _Inventory:
    def __init__(self, workspace_ids: list[str]) -> None:
        self.workspace_ids = list(workspace_ids)
        self.converted: set[str] = set()
        self.free_bytes = 1_000_000_000
        self.required_headroom_bytes = 1
        self.repository_physical_bytes = 0

    def report(self, workspace_ids: list[str] | None = None) -> dict[str, Any]:
        selected = sorted(self.workspace_ids if workspace_ids is None else workspace_ids)
        rows = []
        for workspace_id in selected:
            converted = workspace_id in self.converted
            rows.append(
                {
                    "workspace_id": workspace_id,
                    "ready_records": 1,
                    "legacy_records": 0 if converted else 1,
                    "unique_legacy_blobs": 0 if converted else 1,
                    "legacy_bytes": 0 if converted else 7,
                    "restic_records": 1 if converted else 0,
                    "ledger_pending": 0,
                    "issues": [],
                }
            )
        return {
            "version": 1,
            "active": True,
            "workspaces": rows,
            "totals": {
                "ready_records": len(rows),
                "legacy_records": sum(row["legacy_records"] for row in rows),
                "unique_legacy_blobs": sum(row["unique_legacy_blobs"] for row in rows),
                "legacy_bytes": sum(row["legacy_bytes"] for row in rows),
                "restic_records": sum(row["restic_records"] for row in rows),
                "ledger_pending": 0,
                "repository_physical_bytes": self.repository_physical_bytes,
            },
            "free_bytes": self.free_bytes,
            "required_headroom_bytes": self.required_headroom_bytes,
        }


class BootstrapManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inventory = _Inventory(["alpha", "beta"])
        self.history = _History(self.inventory)
        self.coordinator = SqliteHistoryCoordinator(self.root, object(), history_factory=lambda: self.history)
        self.coordinator.activate()
        self.manager = BootstrapManager(self.coordinator)
        self.manager.inventory = lambda workspace_ids=None, **_kwargs: self.inventory.report(workspace_ids)  # type: ignore[method-assign]

    async def asyncTearDown(self) -> None:
        await self.manager.shutdown()
        self.temporary.cleanup()

    async def _terminal(self, run_id: str, expected: set[str] | None = None) -> dict[str, Any]:
        expected = expected or {"completed", "failed", "blocked", "cancelled", "interrupted"}
        for _ in range(3000):
            result = await self.manager.get(run_id)
            if result["status"] in expected:
                return result
            await asyncio.sleep(0.01)
        self.fail(f"bootstrap did not become terminal: {await self.manager.get(run_id)}")

    async def test_accept_completes_and_exact_replays_do_not_rerun(self) -> None:
        run_id = str(uuid4())
        await self.manager.accept(run_id, "operator", ["beta", "alpha", "alpha"])
        completed = await self._terminal(run_id)
        replay = await self.manager.accept(run_id, "operator", ["alpha", "beta"])
        self.assertEqual("completed", completed["status"])
        self.assertEqual(completed, replay)
        self.assertEqual(["alpha", "beta"], completed["workspace_ids"])
        self.assertEqual(["alpha", "beta"], self.history.calls)
        self.assertEqual(2, completed["completed_workspaces"])

    async def test_null_scope_is_frozen_and_replay_never_rediscovers(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        run_id = str(uuid4())
        await self.manager.accept(run_id, "operator", None)
        await self._terminal(run_id)
        self.inventory.workspace_ids.append("later")
        replay = await self.manager.accept(run_id, "operator", None)
        self.assertEqual(["alpha"], replay["workspace_ids"])
        self.assertEqual(["alpha"], self.history.calls)

    async def test_low_space_blocks_then_resumes_same_run(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        self.inventory.free_bytes = 0
        run_id = str(uuid4())
        await self.manager.accept(run_id, "operator", None)
        blocked = await self._terminal(run_id)
        self.assertEqual("blocked", blocked["status"])
        self.assertEqual([], self.history.calls)
        self.inventory.free_bytes = 1_000_000_000
        resumed = await self.manager.resume(run_id)
        self.assertEqual(run_id, resumed["run_id"])
        completed = await self._terminal(run_id)
        self.assertEqual("completed", completed["status"])
        self.assertEqual(["alpha"], self.history.calls)

    async def test_failed_child_requires_retry_flag_and_gets_new_attempt_id(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        self.history.failures["alpha"] = 1
        run_id = str(uuid4())
        await self.manager.accept(run_id, "operator", ["alpha"])
        failed = await self._terminal(run_id)
        first_id = failed["workspaces"]["alpha"]["operation_id"]
        unchanged = await self.manager.resume(run_id, retry_failed=False)
        self.assertEqual("failed", unchanged["status"])
        await self.manager.resume(run_id, retry_failed=True)
        completed = await self._terminal(run_id)
        second_id = completed["workspaces"]["alpha"]["operation_id"]
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(2, len(completed["workspaces"]["alpha"]["attempts"]))

    async def test_cancellation_waits_for_current_child_boundary(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        self.history.entered = asyncio.Event()
        self.history.release = asyncio.Event()
        run_id = str(uuid4())
        await self.manager.accept(run_id, "operator", ["alpha"])
        await asyncio.wait_for(self.history.entered.wait(), timeout=2)
        cancelling = await self.manager.cancel(run_id)
        self.assertEqual("running", cancelling["status"])
        self.assertNotEqual("cancelled", (await self.manager.get(run_id))["status"])
        self.history.release.set()
        cancelled = await self._terminal(run_id)
        self.assertEqual("cancelled", cancelled["status"])

    async def test_duplicate_concurrent_accepts_have_one_runner(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        self.history.entered = asyncio.Event()
        self.history.release = asyncio.Event()
        run_id = str(uuid4())
        one, two = await asyncio.gather(
            self.manager.accept(run_id, "operator", ["alpha"]),
            self.manager.accept(run_id, "operator", ["alpha"]),
        )
        self.assertEqual(run_id, one["run_id"])
        self.assertEqual(run_id, two["run_id"])
        await asyncio.wait_for(self.history.entered.wait(), timeout=2)
        self.assertEqual(["alpha"], self.history.calls)
        self.history.release.set()
        await self._terminal(run_id)

    async def test_blocking_inventory_keeps_the_event_loop_responsive_through_bootstrap(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        inventory_started: queue.Queue[list[str] | None] = queue.Queue()
        inventory_release: queue.Queue[None] = queue.Queue()

        def blocking_inventory(workspace_ids: list[str] | None = None, **_kwargs: Any) -> dict[str, Any]:
            inventory_started.put(workspace_ids)
            inventory_release.get(timeout=2)
            return self.inventory.report(workspace_ids)

        async def release_one_inventory() -> None:
            await asyncio.wait_for(asyncio.to_thread(inventory_started.get, True, 2), timeout=2)
            progressed = asyncio.Event()
            asyncio.get_running_loop().call_soon(progressed.set)
            await asyncio.wait_for(progressed.wait(), timeout=0.1)
            inventory_release.put(None)

        self.manager.inventory = blocking_inventory  # type: ignore[method-assign]
        run_id = str(uuid4())
        accepting = asyncio.create_task(self.manager.accept(run_id, "operator", ["alpha"]))
        await release_one_inventory()
        await accepting
        await release_one_inventory()
        await release_one_inventory()

        completed = await self._terminal(run_id)
        self.assertEqual("completed", completed["status"])

    async def test_invalid_workspace_ids_remain_sync_errors_and_become_http_400_at_boundaries(self) -> None:
        direct = BootstrapManager(self.coordinator)
        with self.assertRaises(ValueError):
            direct.inventory(["invalid workspace id"])

        with self.assertRaises(HTTPException) as accept_error:
            await direct.accept(str(uuid4()), "operator", ["invalid workspace id"])
        self.assertEqual(400, accept_error.exception.status_code)

        router = bootstrap_router("", None, lambda: self.coordinator)
        inventory_endpoint = next(route.endpoint for route in router.routes if isinstance(route, APIRoute) and route.path.endswith("/inventory"))
        with self.assertRaises(HTTPException) as api_error:
            await inventory_endpoint(workspace_id=["invalid workspace id"], _auth=None)
        self.assertEqual(400, api_error.exception.status_code)

    async def test_existing_completed_child_is_adopted_without_rerun(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        self.inventory.converted.add("alpha")
        run_id = str(uuid4())
        child_id = self.manager.child_operation_id(run_id, "alpha", 1)
        canonical = {"run_id": run_id, "user_id": "operator", "workspace_ids": ["alpha"]}
        parent = self.manager.store.accept(
            operation_id=run_id,
            workspace_id="bootstrap",
            creator_id="operator",
            request_digest=self.manager.request_digest(canonical),
            kind="legacy_bootstrap",
            accepted_payload={
                "request": canonical,
                "run_id": run_id,
                "user_id": "operator",
                "raw_workspace_ids": ["alpha"],
                "workspace_ids": ["alpha"],
                "before": self.inventory.report(["alpha"]),
            },
        )
        self.manager.store.transition(run_id, "interrupted", error="worker restart")
        child_payload = {"operation_id": child_id, "user_id": "operator"}
        self.coordinator._store.accept(
            operation_id=child_id,
            workspace_id="alpha",
            creator_id="operator",
            request_digest=self.manager.request_digest(child_payload),
            kind="legacy_migration",
            accepted_payload=child_payload,
        )
        self.coordinator._store.transition(child_id, "running")
        outcomes = {"legacy_migration": {"migrated_backup_count": 1}}
        self.coordinator._store.transition(child_id, "repository_committed", database_outcomes=outcomes)
        self.coordinator._store.transition(child_id, "catalog_committed")
        self.coordinator._store.transition(child_id, "completed")
        self.manager.store.transition(
            run_id,
            "interrupted",
            database_outcomes={
                "workspaces": {"alpha": {"operation_id": child_id, "status": "running", "attempts": [{"operation_id": child_id, "status": "running"}]}}
            },
        )
        await self.manager.resume(parent["operation_id"])
        completed = await self._terminal(run_id)
        self.assertEqual("completed", completed["status"])
        self.assertEqual([], self.history.calls)

    async def test_completed_get_uses_durable_after_metrics(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        self.inventory.repository_physical_bytes = 42
        run_id = str(uuid4())
        await self.manager.accept(run_id, "operator", None)
        completed = await self._terminal(run_id)
        self.inventory.repository_physical_bytes = 999
        observed = await self.manager.get(run_id)
        self.assertEqual(42, completed["after"]["totals"]["repository_physical_bytes"])
        self.assertEqual(completed["after"], observed["after"])
        self.assertEqual(1, observed["verification"]["unique_artifacts_verified"])

    async def test_final_verification_uses_current_records_without_racy_inventory_count(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        self.inventory.converted.add("alpha")
        original = self.history.list_backups

        async def concurrent_list(workspace_id: str, **filters: Any) -> list[dict[str, Any]]:
            first = await original(workspace_id, **filters)
            second = dict(first[0])
            second["id"] = "backup-added-during-verification"
            second["storage"] = dict(first[0]["storage"])
            second["storage"]["snapshot_id"] = "c" * 64
            return [*first, second]

        self.history.list_backups = concurrent_list  # type: ignore[method-assign]
        after, verification = await self.manager._verify_completion(["alpha"], -1)
        self.assertEqual(1, after["totals"]["ready_records"])
        self.assertEqual(2, verification["ready_records_verified"])
        self.assertEqual(2, verification["unique_artifacts_verified"])

    async def test_dead_running_child_resumes_same_attempt(self) -> None:
        self.inventory.workspace_ids = ["alpha"]
        run_id = str(uuid4())
        child_id = self.manager.child_operation_id(run_id, "alpha", 1)
        canonical = {"run_id": run_id, "user_id": "operator", "workspace_ids": ["alpha"]}
        self.manager.store.accept(
            operation_id=run_id,
            workspace_id="bootstrap",
            creator_id="operator",
            request_digest=self.manager.request_digest(canonical),
            kind="legacy_bootstrap",
            accepted_payload={
                "request": canonical,
                "run_id": run_id,
                "user_id": "operator",
                "raw_workspace_ids": ["alpha"],
                "workspace_ids": ["alpha"],
                "before": self.inventory.report(["alpha"]),
            },
        )
        child_payload = {"operation_id": child_id, "user_id": "operator"}
        self.coordinator._store.accept(
            operation_id=child_id,
            workspace_id="alpha",
            creator_id="operator",
            request_digest=self.manager.request_digest(child_payload),
            kind="legacy_migration",
            accepted_payload=child_payload,
        )
        self.coordinator._store.transition(child_id, "running")
        self.manager.store.transition(
            run_id,
            "interrupted",
            database_outcomes={
                "workspaces": {
                    "alpha": {
                        "operation_id": child_id,
                        "status": "running",
                        "attempt_count": 1,
                        "attempts": [{"operation_id": child_id, "status": "running"}],
                    }
                }
            },
            error="worker restart",
        )
        await self.manager.resume(run_id)
        completed = await self._terminal(run_id)
        self.assertEqual("completed", completed["status"])
        self.assertEqual(child_id, completed["workspaces"]["alpha"]["operation_id"])
        self.assertEqual(["alpha"], self.history.calls)

    async def test_actual_restic_bootstrap_verifies_converted_bytes(self) -> None:
        binary = Path(os.environ.get("RESTIC_BINARY", "/opt/ragtime-backup/bin/restic"))
        if not binary.is_file():
            self.skipTest("actual Restic binary is unavailable")
        await self.manager.shutdown()
        files = self.root / "workspaces" / "actual" / "files"
        files.mkdir(parents=True)
        history_root = files.parent / "sqlite_backups"
        (history_root / "blobs").mkdir(parents=True)
        image = history_root / "blobs" / "image.sqlite3"
        image.write_bytes(b"byte-exact-bootstrap")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        repository = ResticRepository(
            repository_path=self.root / "_sqlite_history" / "restic",
            cache_path=self.root / "_sqlite_history" / "cache",
            password_path=self.root / "_sqlite_history" / "secrets" / "password",
            scratch_path=self.root / "_sqlite_history" / "scratch",
            binary=binary,
        )
        service = RuntimeSqliteHistoryService(self.root, object(), repository)
        service._save(
            history_root,
            {
                "version": 1,
                "workspace_id": "actual",
                "backups": [
                    {
                        "id": "preserved-id",
                        "workspace_id": "actual",
                        "database_name": "app.sqlite3",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "trigger": "manual",
                        "status": "ready",
                        "size_bytes": image.stat().st_size,
                        "sha256": digest,
                        "blob": "blobs/image.sqlite3",
                    }
                ],
                "previews": {},
                "operations": {},
            },
        )
        coordinator = SqliteHistoryCoordinator(self.root, object(), history_factory=lambda: service)
        coordinator.activate()
        manager = BootstrapManager(coordinator)
        self.manager = manager
        run_id = str(uuid4())
        await manager.accept(run_id, "operator", ["actual"])
        completed = await self._terminal(run_id)
        self.assertEqual("completed", completed["status"])
        self.assertEqual(1, completed["verification"]["unique_artifacts_verified"])
        rows = await service.list_backups("actual")
        self.assertEqual("preserved-id", rows[0]["id"])
        self.assertEqual("restic", rows[0]["storage"]["kind"])
        self.assertFalse(image.exists())


if __name__ == "__main__":
    unittest.main()
