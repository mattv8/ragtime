"""Receipt-first legacy SQLite-history migration operations.

Legacy migration runs as a bounded maintenance request. The receipt, global
admission, and history gate remain live in Restic children until work drains.
Retries observe the durable receipt rather than starting a second import.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
from typing import Any, Awaitable, Callable
from uuid import UUID

from fastapi import HTTPException

from .operations import OperationConflict, OperationNotFound

LOW_SPACE_ERROR = "SQLite history conversion is blocked by insufficient disk space"


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class LegacyHistoryMigration:
    """Run the idempotent core migration behind a durable operation receipt."""

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator

    async def accept(self, workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._coordinator._service()  # Reject inactive storage before acceptance.
        try:
            operation_id = str(UUID(str(payload.get("operation_id", ""))))
        except (TypeError, ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail="operation_id must be a UUID") from exc
        user_id = str(payload.get("user_id", "")).strip()
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        accepted_payload = {"operation_id": operation_id, "user_id": user_id}
        try:
            receipt = await asyncio.to_thread(
                self._coordinator._store.accept,
                operation_id=operation_id,
                workspace_id=workspace_id,
                creator_id=user_id,
                request_digest=_digest(accepted_payload),
                kind="legacy_migration",
                accepted_payload=accepted_payload,
            )
        except (ValueError, OperationConflict) as exc:
            raise HTTPException(status_code=409, detail="Migration operation conflicts with an existing request") from exc

        # Exact replays observe the durable result and never run a second import.
        if receipt["phase"] != "accepted":
            return receipt
        return await self._run(workspace_id, operation_id)

    async def get(self, workspace_id: str, operation_id: str) -> dict[str, Any]:
        try:
            receipt = await asyncio.to_thread(self._coordinator._store.get_for_workspace, operation_id, workspace_id)
        except (ValueError, OperationNotFound) as exc:
            raise HTTPException(status_code=404, detail="SQLite history migration operation not found") from exc
        if receipt.get("kind") != "legacy_migration":
            raise HTTPException(status_code=404, detail="SQLite history migration operation not found")
        return receipt

    async def run_with_parent_fds(
        self,
        workspace_id: str,
        operation_id: str,
        *,
        parent_pass_fds: tuple[int, ...] = (),
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> dict[str, Any]:
        """Resume a known receipt while inheriting bootstrap liveness safely."""
        return await self._run(workspace_id, operation_id, parent_pass_fds=parent_pass_fds, cancel_check=cancel_check)

    async def _run(
        self,
        workspace_id: str,
        operation_id: str,
        *,
        parent_pass_fds: tuple[int, ...] = (),
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> dict[str, Any]:
        """Execute while liveness and global capture admission are held.

        A busy global admission leaves the receipt accepted so the same UUID can
        be retried; it is not misreported as a completed migration.
        """
        liveness = self._coordinator._store.hold_liveness(operation_id)
        try:
            liveness_fd = liveness.__enter__()
        except OperationConflict:
            # Only failure to acquire liveness means another producer is live.
            # Conflicts raised while processing must still terminalize below.
            return await asyncio.to_thread(self._coordinator._store.get, operation_id)
        try:
            try:
                capture_fd = self._coordinator._try_global_capture_lock()
                if capture_fd is None:
                    while capture_fd is None:
                        current = await asyncio.to_thread(self._coordinator._store.get, operation_id)
                        if current["phase"] == "cancelled":
                            return current
                        if current["phase"] == "cancelling" or (cancel_check is not None and await cancel_check()):
                            if current["phase"] in {"accepted", "cancelling"}:
                                return await asyncio.to_thread(self._coordinator._store.transition, operation_id, "cancelled")
                            return current
                        await asyncio.sleep(0.25)
                        capture_fd = self._coordinator._try_global_capture_lock()
                try:
                    await asyncio.to_thread(self._coordinator._store.transition, operation_id, "running")
                    pass_fds = tuple(dict.fromkeys((*parent_pass_fds, liveness_fd, capture_fd)))
                    migrated = await self._coordinator._service().migrate_legacy_backups(workspace_id, pass_fds=pass_fds, cancel_check=cancel_check)
                    if cancel_check is not None and await cancel_check():
                        current = await asyncio.to_thread(self._coordinator._store.get, operation_id)
                        if current["phase"] == "running":
                            await asyncio.to_thread(self._coordinator._store.transition, operation_id, "cancelling")
                        return await asyncio.to_thread(self._coordinator._store.transition, operation_id, "cancelled")
                    outcomes = {"legacy_migration": {"migrated_backup_count": migrated}}
                    await asyncio.to_thread(
                        self._coordinator._store.transition,
                        operation_id,
                        "repository_committed",
                        database_outcomes=outcomes,
                    )
                    await asyncio.to_thread(self._coordinator._store.transition, operation_id, "catalog_committed")
                    return await asyncio.to_thread(self._coordinator._store.transition, operation_id, "completed")
                finally:
                    fcntl.flock(capture_fd, fcntl.LOCK_UN)
                    os.close(capture_fd)
            except asyncio.CancelledError:
                await self._terminalize_cancel(operation_id)
                raise
        except HTTPException as exc:
            try:
                current = await asyncio.to_thread(self._coordinator._store.get, operation_id)
                if current["phase"] not in {"completed", "failed", "cancelled", "interrupted"}:
                    error = LOW_SPACE_ERROR if exc.status_code == 507 else str(exc.detail)
                    await asyncio.to_thread(self._coordinator._store.transition, operation_id, "failed", error=error)
            except (OperationConflict, OperationNotFound):
                pass
            raise
        except Exception:
            try:
                current = await asyncio.to_thread(self._coordinator._store.get, operation_id)
                if current["phase"] not in {"completed", "failed", "cancelled", "interrupted"}:
                    return await asyncio.to_thread(
                        self._coordinator._store.transition,
                        operation_id,
                        "failed",
                        error="SQLite history legacy migration failed",
                    )
            except (OperationConflict, OperationNotFound):
                pass
            raise
        finally:
            liveness.__exit__(None, None, None)

    async def _terminalize_cancel(self, operation_id: str) -> dict[str, Any]:
        """Make cancellation durable even when the caller supplied no probe."""
        try:
            current = await asyncio.to_thread(self._coordinator._store.get, operation_id)
            phase = current["phase"]
            if phase in {"completed", "failed", "cancelled", "interrupted"}:
                return current
            if phase == "catalog_committed":
                return await asyncio.to_thread(self._coordinator._store.transition, operation_id, "completed")
            if phase == "accepted":
                return await asyncio.to_thread(self._coordinator._store.transition, operation_id, "cancelled")
            if phase != "cancelling":
                await asyncio.to_thread(self._coordinator._store.transition, operation_id, "cancelling")
            return await asyncio.to_thread(self._coordinator._store.transition, operation_id, "cancelled")
        except (OperationConflict, OperationNotFound):
            return await asyncio.to_thread(self._coordinator._store.get, operation_id)
