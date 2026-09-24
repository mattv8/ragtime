"""Durable, sequential bootstrap orchestration for legacy history conversion."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from fastapi import HTTPException

from .bootstrap_store import BootstrapOperationStore
from .inventory import inventory_legacy_history
from .migration import LegacyHistoryMigration
from .operations import OperationConflict, OperationNotFound
from .storage import restic_artifact

logger = logging.getLogger(__name__)
_MAX_ATTEMPT_HISTORY = 20
_LOW_SPACE_ERROR = "Insufficient migration disk headroom"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BootstrapManager:
    """Coordinate one durable bootstrap run without weakening generic receipts."""

    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator
        self.store = BootstrapOperationStore(Path(coordinator._root) / "_sqlite_history" / "bootstrap")
        self.tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def request_digest(value: Any) -> str:
        return _digest(value)

    @staticmethod
    def child_operation_id(run_id: str, workspace_id: str, attempt: int) -> str:
        return str(uuid5(UUID(run_id), f"bootstrap:{workspace_id}:{attempt}"))

    def inventory(self, workspace_ids: list[str] | None = None, *, verify_integrity: bool = True) -> dict[str, Any]:
        report = inventory_legacy_history(self.coordinator._root, workspace_ids, verify_integrity=verify_integrity)
        report["active"] = self.coordinator._active()
        return report

    @staticmethod
    def _canonical_scope(workspace_ids: list[str] | None) -> list[str] | None:
        if workspace_ids is None:
            return None
        if not isinstance(workspace_ids, list) or any(not isinstance(item, str) for item in workspace_ids):
            raise HTTPException(status_code=400, detail="workspace_ids must be a list of workspace IDs or null")
        return sorted(set(workspace_ids))

    async def accept(self, run_id: str, user_id: str, workspace_ids: list[str] | None) -> dict[str, Any]:
        try:
            run_id = str(UUID(run_id))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="run_id must be a UUID") from exc
        user_id = user_id.strip()
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        if not self.coordinator._active():
            raise HTTPException(status_code=503, detail="Runtime SQLite history v2 is not activated")

        raw_scope = self._canonical_scope(workspace_ids)
        canonical = {"run_id": run_id, "user_id": user_id, "workspace_ids": raw_scope}
        request_digest = _digest(canonical)
        try:
            existing = await asyncio.to_thread(self.store.get, run_id)
        except OperationNotFound:
            existing = None
        if existing is not None:
            if existing.get("creator_id") != user_id or existing.get("request_digest") != request_digest:
                raise HTTPException(status_code=409, detail="Bootstrap run conflicts with an existing request")
            return self.view(existing)

        try:
            before = await asyncio.to_thread(self.inventory, raw_scope, verify_integrity=False)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid workspace ID") from exc
        frozen = [str(item["workspace_id"]) for item in before["workspaces"]]
        accepted_payload = {
            "request": canonical,
            "run_id": run_id,
            "user_id": user_id,
            "raw_workspace_ids": raw_scope,
            "workspace_ids": frozen,
            "before": before,
        }
        try:
            receipt = await asyncio.to_thread(
                self.store.accept,
                operation_id=run_id,
                workspace_id="bootstrap",
                creator_id=user_id,
                request_digest=request_digest,
                kind="legacy_bootstrap",
                accepted_payload=accepted_payload,
            )
        except (ValueError, OperationConflict) as exc:
            raise HTTPException(status_code=409, detail="Bootstrap run conflicts with an existing request") from exc
        if receipt["phase"] == "accepted":
            self._launch(run_id, retry_failed=False)
        return self.view(receipt)

    def _launch(self, run_id: str, *, retry_failed: bool) -> bool:
        existing = self.tasks.get(run_id)
        if existing is not None and not existing.done():
            return False
        liveness = self.store.hold_liveness(run_id)
        try:
            liveness_fd = liveness.__enter__()
        except OperationConflict:
            return False
        try:
            receipt = self.store.get(run_id)
            phase = receipt["phase"]
            if phase == "cancelling":
                self.store.transition(run_id, "cancelled")
                liveness.__exit__(None, None, None)
                return False
            if phase != "accepted":
                if phase != "reconciling":
                    self.store.transition(run_id, "reconciling", error=None)
                self.store.transition(run_id, "running", error=None)
            else:
                self.store.transition(run_id, "running")
        except Exception:
            liveness.__exit__(None, None, None)
            raise
        task = asyncio.create_task(
            self._run_claimed(run_id, liveness, liveness_fd, retry_failed=retry_failed),
            name=f"sqlite-history-bootstrap-{run_id}",
        )
        self.tasks[run_id] = task

        def finished(done: asyncio.Task[None]) -> None:
            if self.tasks.get(run_id) is done:
                self.tasks.pop(run_id, None)

        task.add_done_callback(finished)
        return True

    async def start(self) -> None:
        """Classify only kernel-proven dead active receipts; never auto-run them."""
        for receipt in await asyncio.to_thread(self.store.list_active):
            operation_id = receipt["operation_id"]
            liveness = self.store.hold_liveness(operation_id)
            try:
                liveness.__enter__()
            except OperationConflict:
                continue
            try:
                current = await asyncio.to_thread(self.store.get, operation_id)
                if current["phase"] != "reconciling":
                    await asyncio.to_thread(self.store.transition, operation_id, "reconciling")
            except (OperationConflict, OperationNotFound, ValueError):
                logger.exception("Bootstrap startup reconciliation failed run_id=%s", operation_id)
            finally:
                liveness.__exit__(None, None, None)

    async def shutdown(self) -> None:
        if self.tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in tuple(self.tasks.values())), return_exceptions=True)

    async def get(self, run_id: str) -> dict[str, Any]:
        try:
            return self.view(await asyncio.to_thread(self.store.get, run_id))
        except (ValueError, OperationNotFound) as exc:
            raise HTTPException(status_code=404, detail="Bootstrap run not found") from exc

    async def resume(self, run_id: str, retry_failed: bool = False) -> dict[str, Any]:
        try:
            receipt = await asyncio.to_thread(self.store.get, run_id)
        except (ValueError, OperationNotFound) as exc:
            raise HTTPException(status_code=404, detail="Bootstrap run not found") from exc
        if receipt["phase"] == "completed":
            return self.view(receipt)
        if receipt["phase"] == "failed" and not retry_failed:
            return self.view(receipt)
        # _launch acquires the nonblocking kernel liveness claim before the task
        # is allowed to alter a resume phase. A concurrent/live runner is merely
        # observed and never turned into a failure.
        self._launch(run_id, retry_failed=retry_failed)
        return self.view(await asyncio.to_thread(self.store.get, run_id))

    async def cancel(self, run_id: str) -> dict[str, Any]:
        try:
            return self.view(await asyncio.to_thread(self.store.request_cancel, run_id))
        except (ValueError, OperationNotFound) as exc:
            raise HTTPException(status_code=404, detail="Bootstrap run not found") from exc

    async def _run_claimed(
        self,
        run_id: str,
        liveness: Any,
        liveness_fd: int,
        *,
        retry_failed: bool,
    ) -> None:
        try:
            await self._run(run_id, liveness_fd, retry_failed=retry_failed)
        except Exception:
            logger.exception("SQLite history bootstrap failed run_id=%s", run_id)
            try:
                current = await asyncio.to_thread(self.store.get, run_id)
                if current["phase"] not in {"completed", "failed", "cancelled", "interrupted"}:
                    terminal = "cancelled" if current["phase"] == "cancelling" else "failed"
                    await asyncio.to_thread(
                        self.store.transition,
                        run_id,
                        terminal,
                        error="SQLite history bootstrap failed",
                    )
            except (OperationConflict, OperationNotFound, ValueError):
                logger.exception("Bootstrap terminal receipt update failed run_id=%s", run_id)
        finally:
            liveness.__exit__(None, None, None)

    async def _run(self, run_id: str, liveness_fd: int, *, retry_failed: bool) -> None:
        receipt = await asyncio.to_thread(self.store.get, run_id)
        phase = receipt["phase"]
        if phase == "completed":
            return
        if phase == "failed" and not retry_failed:
            return
        if phase == "cancelling":
            await asyncio.to_thread(self.store.transition, run_id, "cancelled")
            return
        # _launch established running only after acquiring this task's kernel
        # liveness claim. Reload so every resume starts from durable progress.
        receipt = await asyncio.to_thread(self.store.get, run_id)

        payload = receipt.get("accepted_payload") or {}
        frozen = list(payload.get("workspace_ids") or [])
        current_inventory = await asyncio.to_thread(self.inventory, frozen, verify_integrity=False)
        if any(item.get("issues") for item in current_inventory["workspaces"]):
            await asyncio.to_thread(
                self.store.transition,
                run_id,
                "interrupted",
                error="Legacy history inventory has unsafe records",
            )
            return
        free_bytes = current_inventory.get("free_bytes")
        if free_bytes is not None and free_bytes < current_inventory.get("required_headroom_bytes", 0):
            await asyncio.to_thread(self.store.transition, run_id, "interrupted", error=_LOW_SPACE_ERROR)
            return

        for workspace_id in frozen:
            if await self._cancel_requested(run_id):
                await asyncio.to_thread(self.store.transition, run_id, "cancelled")
                return
            current = await asyncio.to_thread(self.store.get, run_id)
            durable = dict(current.get("database_outcomes") or {})
            outcomes = dict(durable.get("workspaces") or {})
            outcome = dict(outcomes.get(workspace_id) or {})
            result = await self._run_workspace(
                run_id,
                workspace_id,
                str(payload["user_id"]),
                outcome,
                liveness_fd,
                retry_failed=retry_failed,
            )
            outcomes[workspace_id] = result
            durable["workspaces"] = outcomes
            await asyncio.to_thread(self.store.update_progress, run_id, database_outcomes=durable)
            if result["status"] == "cancelled" or await self._cancel_requested(run_id):
                await asyncio.to_thread(self.store.transition, run_id, "cancelled")
                return
            if result["status"] == "blocked":
                await asyncio.to_thread(
                    self.store.transition,
                    run_id,
                    "interrupted",
                    database_outcomes=durable,
                    error=_LOW_SPACE_ERROR,
                )
                return
            if result["status"] != "completed":
                await asyncio.to_thread(
                    self.store.transition,
                    run_id,
                    "failed",
                    error="A workspace migration failed",
                )
                return

        after, verification = await self._verify_completion(frozen, liveness_fd)
        current = await asyncio.to_thread(self.store.get, run_id)
        durable = dict(current.get("database_outcomes") or {})
        durable["summary"] = {"after": after, "verification": verification}
        await asyncio.to_thread(
            self.store.transition,
            run_id,
            "repository_committed",
            database_outcomes=durable,
        )
        await asyncio.to_thread(self.store.transition, run_id, "catalog_committed")
        await asyncio.to_thread(self.store.transition, run_id, "completed")

    async def _run_workspace(
        self,
        run_id: str,
        workspace_id: str,
        user_id: str,
        outcome: dict[str, Any],
        parent_fd: int,
        *,
        retry_failed: bool,
    ) -> dict[str, Any]:
        attempts = [dict(item) for item in outcome.get("attempts", []) if isinstance(item, dict)]
        attempt_count = int(outcome.get("attempt_count") or len(attempts))
        child_id = outcome.get("operation_id")
        child: dict[str, Any] | None = None
        if isinstance(child_id, str):
            with contextlib.suppress(OperationNotFound, ValueError):
                child = await asyncio.to_thread(self.coordinator._store.get_for_workspace, child_id, workspace_id)

        if child is not None and child["phase"] == "completed":
            return self._finished_outcome(child, attempts, attempt_count)
        if child is not None and child["phase"] == "failed" and not retry_failed and outcome.get("category") != "low_space":
            return self._failed_outcome(child, attempts, attempt_count)
        if child is None or child["phase"] in {"failed", "cancelled", "interrupted"}:
            attempt_count += 1
            child_id = self.child_operation_id(run_id, workspace_id, attempt_count)
            attempt = {"operation_id": child_id, "status": "accepted"}
            attempts = [*attempts, attempt][-_MAX_ATTEMPT_HISTORY:]
            outcome = {
                "operation_id": child_id,
                "status": "running",
                "attempt_count": attempt_count,
                "attempts": attempts,
            }
            await self._save_workspace_outcome(run_id, workspace_id, outcome)
            child_payload = {"operation_id": child_id, "user_id": user_id}
            child = await asyncio.to_thread(
                self.coordinator._store.accept,
                operation_id=child_id,
                workspace_id=workspace_id,
                creator_id=user_id,
                request_digest=_digest(child_payload),
                kind="legacy_migration",
                accepted_payload=child_payload,
            )

        assert child is not None and isinstance(child_id, str)
        child = await self._observe_or_run_child(run_id, workspace_id, child, parent_fd)
        if child["phase"] == "completed":
            return self._finished_outcome(child, attempts, attempt_count)
        if child["phase"] == "cancelled":
            return self._child_outcome(child, attempts, attempt_count, "cancelled")
        return self._failed_outcome(child, attempts, attempt_count)

    async def _observe_or_run_child(
        self,
        run_id: str,
        workspace_id: str,
        child: dict[str, Any],
        parent_fd: int,
    ) -> dict[str, Any]:
        child_id = child["operation_id"]
        migration = LegacyHistoryMigration(self.coordinator)
        while True:
            child = await asyncio.to_thread(self.coordinator._store.get_for_workspace, child_id, workspace_id)
            if child["phase"] in {"completed", "failed", "cancelled", "interrupted"}:
                return child
            try:
                live = await asyncio.to_thread(self.coordinator._store.is_live, child_id)
            except FileNotFoundError:
                live = False
            if live:
                if await self._cancel_requested(run_id):
                    await asyncio.to_thread(self.coordinator._store.request_cancel, child_id)
                await asyncio.sleep(0.1)
                continue
            if child["phase"] not in {"accepted", "reconciling"}:
                child = await asyncio.to_thread(self.coordinator._store.transition, child_id, "reconciling")
            try:
                child = await migration.run_with_parent_fds(
                    workspace_id,
                    child_id,
                    parent_pass_fds=(parent_fd,),
                    cancel_check=lambda: self._cancel_requested(run_id),
                )
            except HTTPException as exc:
                if exc.status_code == 507:
                    logger.warning(
                        "Bootstrap child blocked by disk headroom run_id=%s workspace_id=%s child_id=%s",
                        run_id,
                        workspace_id,
                        child_id,
                    )
                    child = await asyncio.to_thread(self.coordinator._store.get_for_workspace, child_id, workspace_id)
                    child = dict(child)
                    child["_bootstrap_low_space"] = True
                    return child
                logger.exception(
                    "Bootstrap child migration failed run_id=%s workspace_id=%s child_id=%s",
                    run_id,
                    workspace_id,
                    child_id,
                )
                return await asyncio.to_thread(self.coordinator._store.get_for_workspace, child_id, workspace_id)
            except Exception:
                logger.exception(
                    "Bootstrap child migration failed run_id=%s workspace_id=%s child_id=%s",
                    run_id,
                    workspace_id,
                    child_id,
                )
                return await asyncio.to_thread(self.coordinator._store.get_for_workspace, child_id, workspace_id)
            if child["phase"] == "accepted":
                await asyncio.sleep(0.1)
                continue
            return child

    async def _save_workspace_outcome(self, run_id: str, workspace_id: str, outcome: dict[str, Any]) -> None:
        current = await asyncio.to_thread(self.store.get, run_id)
        durable = dict(current.get("database_outcomes") or {})
        workspaces = dict(durable.get("workspaces") or {})
        workspaces[workspace_id] = outcome
        durable["workspaces"] = workspaces
        await asyncio.to_thread(self.store.update_progress, run_id, database_outcomes=durable)

    @staticmethod
    def _attempts_with_status(
        attempts: list[dict[str, Any]],
        child: dict[str, Any],
        status: str,
    ) -> list[dict[str, Any]]:
        result = [dict(item) for item in attempts]
        for item in reversed(result):
            if item.get("operation_id") == child["operation_id"]:
                item["status"] = status
                if child.get("error"):
                    item["error"] = child["error"]
                break
        return result[-_MAX_ATTEMPT_HISTORY:]

    def _child_outcome(
        self,
        child: dict[str, Any],
        attempts: list[dict[str, Any]],
        attempt_count: int,
        status: str,
    ) -> dict[str, Any]:
        result = {
            "operation_id": child["operation_id"],
            "status": status,
            "attempt_count": attempt_count,
            "attempts": self._attempts_with_status(attempts, child, status),
        }
        if child.get("error"):
            result["error"] = child["error"]
        return result

    def _finished_outcome(
        self,
        child: dict[str, Any],
        attempts: list[dict[str, Any]],
        attempt_count: int,
    ) -> dict[str, Any]:
        result = self._child_outcome(child, attempts, attempt_count, "completed")
        result["migrated_backup_count"] = int(((child.get("database_outcomes") or {}).get("legacy_migration") or {}).get("migrated_backup_count", 0))
        return result

    def _failed_outcome(
        self,
        child: dict[str, Any],
        attempts: list[dict[str, Any]],
        attempt_count: int,
    ) -> dict[str, Any]:
        if child.get("_bootstrap_low_space"):
            result = self._child_outcome(child, attempts, attempt_count, "blocked")
            result["category"] = "low_space"
            return result
        return self._child_outcome(child, attempts, attempt_count, "failed")

    async def _cancel_requested(self, run_id: str) -> bool:
        return (await asyncio.to_thread(self.store.get, run_id))["phase"] == "cancelling"

    async def _verify_completion(self, frozen: list[str], parent_fd: int) -> tuple[dict[str, Any], dict[str, int]]:
        after = await asyncio.to_thread(self.inventory, frozen)
        if after["totals"].get("legacy_records") or after["totals"].get("ledger_pending") or any(item.get("issues") for item in after["workspaces"]):
            raise RuntimeError("Legacy references, conversion journals, or inventory issues remain")
        service = self.coordinator._service()
        repository = getattr(service, "repository", None)
        if repository is None or not hasattr(repository, "verify"):
            raise RuntimeError("SQLite history repository verification is unavailable")
        unique: dict[tuple[str, str, str, int, str], Any] = {}
        ready_records = 0
        async with service._installed():
            from .storage import held_repository_fds

            verification_fds = tuple(dict.fromkeys((parent_fd, *held_repository_fds())))
            for workspace_id in frozen:
                rows = await service.list_backups(workspace_id)
                for row in rows:
                    if row.get("status") != "ready":
                        continue
                    ready_records += 1
                    storage = row.get("storage")
                    if not isinstance(storage, dict) or storage.get("kind") != "restic":
                        raise RuntimeError("A ready history record is not stored in Restic")
                    artifact = restic_artifact(
                        storage,
                        size_bytes=int(row["size_bytes"]),
                        sha256=str(row["sha256"]),
                    )
                    key = (
                        artifact.repository_id,
                        artifact.snapshot_id,
                        artifact.path,
                        artifact.size_bytes,
                        artifact.sha256,
                    )
                    unique[key] = artifact
            for artifact in unique.values():
                await repository.verify(artifact, pass_fds=verification_fds)
        return after, {
            "ready_records_verified": ready_records,
            "unique_artifacts_verified": len(unique),
        }

    def view(self, receipt: dict[str, Any]) -> dict[str, Any]:
        payload = receipt.get("accepted_payload") or {}
        durable = receipt.get("database_outcomes") or {}
        outcomes = durable.get("workspaces") or {}
        summary = durable.get("summary") or {}
        phase = receipt["phase"]
        status = {
            "accepted": "accepted",
            "running": "running",
            "cancelling": "running",
            "completed": "completed",
            "cancelled": "cancelled",
            "failed": "failed",
            "interrupted": "blocked" if receipt.get("error") == _LOW_SPACE_ERROR else "interrupted",
            "reconciling": "interrupted",
            "repository_committed": "running",
            "catalog_committed": "running",
        }[phase]
        return {
            "run_id": receipt["operation_id"],
            "status": status,
            "workspace_ids": payload.get("workspace_ids", []),
            "completed_workspaces": sum(1 for row in outcomes.values() if row.get("status") == "completed"),
            "total_workspaces": len(payload.get("workspace_ids", [])),
            "workspaces": outcomes,
            "before": payload.get("before"),
            "after": summary.get("after"),
            "verification": summary.get("verification"),
            "error": receipt.get("error"),
        }
