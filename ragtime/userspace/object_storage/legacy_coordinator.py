"""Durable scheduling for legacy workspace object-storage imports."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ragtime.userspace.object_storage import control
from ragtime.userspace.object_storage.legacy_migration import (
    LegacyMigrationError,
    LegacyObjectStorageMigrator,
    workspace_gc_fence,
)

WorkspaceIds = Callable[[], Awaitable[list[str]]]
RuntimeActive = Callable[[str], Awaitable[bool]]
LegacyPayload = Callable[[str], dict[str, Any] | None]


class LegacyObjectStorageCoordinator:
    """Serialize staging while allowing durable jobs to be polled independently."""

    def __init__(
        self,
        migrator: LegacyObjectStorageMigrator,
        *,
        workspace_ids: WorkspaceIds,
        runtime_active: RuntimeActive,
        legacy_payload: LegacyPayload,
        config_path: Callable[[str], Path],
        workspaces_dir: Path,
        logger: Any,
    ) -> None:
        self._migrator = migrator
        self._workspace_ids = workspace_ids
        self._runtime_active = runtime_active
        self._legacy_payload = legacy_payload
        self._config_path = config_path
        self._workspaces_dir = workspaces_dir
        self._logger = logger
        self._loop_task: asyncio.Task[None] | None = None
        self._loop_lock = asyncio.Lock()
        self.workspace_tasks: dict[str, asyncio.Task[None]] = {}
        self._workspace_locks: dict[str, asyncio.Lock] = {}
        self._terminal_workspaces: set[str] = set()
        self._active_workspace: str | None = None
        self._reported_orphans: set[str] = set()

    def _receipts(self, workspace_id: str) -> list[dict[str, Any]]:
        """Use the public compact receipt view; terminal records have no manifest."""
        return self._migrator.load_receipts(workspace_id)

    @staticmethod
    def _unfinished(receipts: list[dict[str, Any]]) -> bool:
        return any(receipt.get("cleanup_state") != "consumed" for receipt in receipts)

    def needs_reconciliation(self, workspace_id: str, receipts: list[dict[str, Any]] | None = None) -> bool:
        receipts = self._receipts(workspace_id) if receipts is None else receipts
        if any(receipt.get("cleanup_state") == "consumed" for receipt in receipts):
            return False
        if self._unfinished(receipts):
            return True
        if workspace_id in self._terminal_workspaces:
            return False
        if self._migrator.source_buckets(workspace_id).exists():
            return True
        return self._config_path(workspace_id).is_file()

    async def reconcile(self, workspace_id: str) -> bool:
        """Advance one admitted workspace; a pending job retains admission."""
        try:
            if self._active_workspace not in {None, workspace_id}:
                return False
            self._active_workspace = workspace_id
            lock = self._workspace_locks.setdefault(workspace_id, asyncio.Lock())
            async with lock, workspace_gc_fence(workspace_id):
                result = await self._migrator.reconcile(workspace_id, runtime_active=await self._runtime_active(workspace_id))
            if not await asyncio.to_thread(self.needs_reconciliation, workspace_id):
                self._active_workspace = None
            return result
        except (LegacyMigrationError, HTTPException) as exc:
            # A durable receipt reserves the staging budget; a failed first stage does not.
            receipts = await asyncio.to_thread(self._receipts, workspace_id)
            if not self._unfinished(receipts):
                self._active_workspace = None
            self._logger.warning("Legacy object-storage migration deferred for workspace=%s: %s", workspace_id, exc)
            raise HTTPException(status_code=503, detail="Object storage migration in progress") from exc

    def enqueue(self, workspace_id: str) -> None:
        task = self.workspace_tasks.get(workspace_id)
        if task is not None and not task.done():
            return

        async def run() -> None:
            try:
                await self.process(workspace_id)
            except HTTPException:
                pass

        self.workspace_tasks[workspace_id] = asyncio.create_task(run(), name=f"legacy-object-storage-{workspace_id}")

    async def process(self, workspace_id: str, receipts: list[dict[str, Any]] | None = None) -> bool:
        """Background advancement. HTTP reads only enqueue this work."""
        if receipts is None:
            receipts = await asyncio.to_thread(self._receipts, workspace_id)
        if not self.needs_reconciliation(workspace_id, receipts):
            return True
        unfinished = self._unfinished(receipts)
        try:
            config = await control.get_workspace(workspace_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            legacy = await asyncio.to_thread(self._legacy_payload, workspace_id)
            if legacy is None:
                raise HTTPException(status_code=503, detail="Object storage migration in progress") from exc
            await control.ensure_workspace(workspace_id, legacy)
            return await self.reconcile(workspace_id)

        legacy_state = str(config.get("legacy_import_state") or "")
        if legacy_state == "completed":
            if unfinished:
                return await self.reconcile(workspace_id)
            self._logger.warning("Retaining legacy object-storage source without bound receipt workspace=%s", workspace_id)
            self._terminal_workspaces.add(workspace_id)
            return True
        if config.get("state") == "ready" and not legacy_state:
            self._logger.warning("Retaining ambiguous legacy object-storage source workspace=%s", workspace_id)
            self._terminal_workspaces.add(workspace_id)
            return True
        if not unfinished and await asyncio.to_thread(self._legacy_payload, workspace_id) is None:
            raise HTTPException(status_code=503, detail="Object storage migration in progress")
        return await self.reconcile(workspace_id)

    async def ensure_managed(self, workspace_id: str) -> dict[str, Any]:
        """Serve ready storage while lazily scheduling unfinished durable work."""
        receipts = await asyncio.to_thread(self._receipts, workspace_id)
        needs_reconciliation = self.needs_reconciliation(workspace_id, receipts)
        try:
            config = await control.get_workspace(workspace_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            if not needs_reconciliation:
                return await control.ensure_workspace(workspace_id)
            self.enqueue(workspace_id)
            raise HTTPException(status_code=503, detail="Object storage migration in progress") from exc
        legacy_state = str(config.get("legacy_import_state") or "")
        if legacy_state in {"pending", "copying", "failed"}:
            self.enqueue(workspace_id)
            raise HTTPException(status_code=503, detail="Object storage migration in progress")
        if legacy_state == "completed" or (config.get("state") == "ready" and not legacy_state):
            if needs_reconciliation:
                self.enqueue(workspace_id)
            return config
        if not needs_reconciliation:
            return config
        self.enqueue(workspace_id)
        raise HTTPException(status_code=503, detail="Object storage migration in progress")

    def orphan_ids(self, live_workspace_ids: set[str]) -> set[str]:
        if not self._workspaces_dir.is_dir():
            return set()
        return {
            path.name for path in self._workspaces_dir.iterdir() if path.is_dir() and path.name not in live_workspace_ids and (path / "s3" / "buckets").exists()
        }

    async def start(self) -> None:
        async with self._loop_lock:
            if self._loop_task is None or self._loop_task.done():
                self._loop_task = asyncio.create_task(self._loop(), name="legacy-object-storage-reconciliation")

    async def shutdown(self) -> None:
        task, self._loop_task = self._loop_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        tasks = list(self.workspace_tasks.values())
        self.workspace_tasks.clear()
        for workspace_task in tasks:
            workspace_task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Durable receipt state is deliberately not cached across process restart.
        self._active_workspace = None
        self._workspace_locks.clear()

    async def _loop(self) -> None:
        delay = 2.0
        while True:
            try:
                workspace_ids = await self._workspace_ids()
                receipts_by_workspace = await asyncio.to_thread(lambda: {workspace_id: self._receipts(workspace_id) for workspace_id in workspace_ids})
                unfinished = {workspace_id for workspace_id, receipts in receipts_by_workspace.items() if self._unfinished(receipts)}
                active = self._active_workspace
                workspace_ids.sort(key=lambda item: (0 if item == active else 1 if item in unfinished else 2, item))
                for workspace_id in workspace_ids:
                    receipts = receipts_by_workspace[workspace_id]
                    if self.needs_reconciliation(workspace_id, receipts):
                        try:
                            await self.process(workspace_id, receipts)
                        except HTTPException:
                            continue
                orphans = await asyncio.to_thread(self.orphan_ids, set(workspace_ids))
                for workspace_id in sorted(orphans - self._reported_orphans):
                    self._logger.warning("Retaining orphan legacy object-storage directory workspace=%s", workspace_id)
                self._reported_orphans.update(orphans)
                delay = 2.0 if self._active_workspace else 30.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.warning("Legacy object-storage reconciliation loop failed: %s", exc)
                delay = min(max(delay, 30.0) * 2, 60.0)
            await asyncio.sleep(delay)
