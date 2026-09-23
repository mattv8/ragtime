"""Durable runtime coordination for SQLite history HTTP operations.

The coordinator deliberately owns only receipt admission and observation.  The
history service remains the sole owner of catalogs, workspace fences, and
SQLite/Restic work.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from fastapi import HTTPException

from .operations import OperationConflict, OperationNotFound, OperationStore

logger = logging.getLogger(__name__)


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SqliteHistoryCoordinator:
    """Receipt-first coordinator shared by manager and embedded worker routes."""

    def __init__(self, root: Path, worker: Any, history_factory: Callable[[], Any] | None = None) -> None:
        self._root = Path(root)
        self._worker = worker
        self._store = OperationStore(self._root / "_sqlite_history")
        self._history_factory = history_factory or self._default_history_factory
        self._history: Any | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._maintenance_task: asyncio.Task[None] | None = None
        self._maintenance: Any | None = None
        self._maintenance_error: str | None = None
        self._started = False
        self._transfers: Any | None = None
        self._bootstrap: Any | None = None

    def sqlite_history_bootstrap_manager(self) -> Any:
        """Return the separately stored bootstrap coordinator.

        Bootstrap receipts intentionally do not live in ``_store``: startup
        recovery of that store is capture-only and must never dispatch a legacy
        migration as though it were a capture.
        """
        if self._bootstrap is None:
            from .bootstrap import BootstrapManager

            self._bootstrap = BootstrapManager(self)
        return self._bootstrap

    def get_history_transfers(self) -> Any:
        if self._transfers is None:
            from .transfer import RuntimeHistoryTransfers

            self._transfers = RuntimeHistoryTransfers(self)
        return self._transfers

    @property
    def _activation_path(self) -> Path:
        return self._root / "_sqlite_history" / "activation-v1.json"

    def _active(self) -> bool:
        try:
            if self._activation_path.is_symlink():
                raise HTTPException(status_code=503, detail="Runtime SQLite history activation state is invalid")
            payload = json.loads(self._activation_path.read_text(encoding="utf-8"))
            if payload != {"version": 2, "active": True}:
                raise HTTPException(status_code=503, detail="Runtime SQLite history activation state is invalid")
            return True
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Runtime SQLite history activation state is unavailable") from exc

    def activate(self) -> dict[str, Any]:
        """Persist the v2 cutover; authentication is enforced by the route."""
        path = self._activation_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            json.dump({"version": 2, "active": True}, output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if self._started and self._maintenance_task is None:
            self._maintenance_task = asyncio.get_running_loop().create_task(self._maintenance_loop(), name="sqlite-history-maintenance")
        return {"version": 2, "active": True, "capability": self.capability()}

    def _default_history_factory(self) -> Any:
        from . import service as history_module

        cls = getattr(history_module, "RuntimeSqliteHistoryService", None)
        if cls is None:
            return None
        from .repository import ResticRepository

        history_root = self._root / "_sqlite_history"
        repository = ResticRepository(
            repository_path=history_root / "restic",
            cache_path=history_root / "cache",
            password_path=history_root / "secrets" / "repository-password",
            scratch_path=history_root / "scratch",
        )
        return cls(self._root, self._worker, repository=repository)

    def _service(self) -> Any:
        if not self._active():
            raise HTTPException(status_code=503, detail="Runtime SQLite history v2 is not activated")
        if self._history is None:
            self._history = self._history_factory()
        if self._history is None:
            raise HTTPException(status_code=503, detail="Runtime SQLite history capability is unavailable")
        return self._history

    def capability(self) -> bool:
        """Report only an explicitly activated and loadable implementation."""
        if not self._active():
            return False
        try:
            return self._service() is not None
        except Exception:
            return False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await self.sqlite_history_bootstrap_manager().start()
        if self._active():
            self._service()
            transfers = self.get_history_transfers()
            if hasattr(transfers, "start"):
                await transfers.start()
            self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="sqlite-history-maintenance")
        # Never infer death from a timer.  Only receipts whose inherited
        # liveness flock is free can be classified after a worker restart.
        for receipt in await asyncio.to_thread(self._store.list_active):
            operation_id = receipt["operation_id"]
            try:
                if self._active() and not await asyncio.to_thread(self._store.is_live, operation_id):
                    await asyncio.to_thread(self._store.transition, operation_id, "reconciling")
                    # Core reconciliation can adopt a committed repository
                    # snapshot or catalog boundary.  In its absence preserve
                    # the durable reconciling state; never manufacture an
                    # interruption merely because this process restarted.
                    reconcile = getattr(self._service(), "reconcile_capture_operation", None)
                    if reconcile is None:
                        await asyncio.to_thread(self._store.transition, operation_id, "interrupted", error="Runtime capture recovery is unavailable")
                    else:
                        outcome = await reconcile(receipt)
                        phase = str((outcome or {}).get("phase") or "interrupted")
                        if phase in self._store.PHASES:
                            await asyncio.to_thread(
                                self._store.transition,
                                operation_id,
                                phase,
                                **{key: value for key, value in (outcome or {}).items() if key in {"database_outcomes", "repository_refs", "error"}},
                            )
            except Exception:
                logger.exception("Runtime history reconciliation failed operation_id=%s", operation_id)

    async def shutdown(self) -> None:
        # Accepted jobs are intentionally not cancelled by a lost HTTP request
        # or normal service shutdown.  Their receipts reconcile on next start.
        self._started = False
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None
        if self._tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in tuple(self._tasks.values())), return_exceptions=True)
        if self._bootstrap is not None:
            await self._bootstrap.shutdown()
        if self._transfers is not None and hasattr(self._transfers, "shutdown"):
            await self._transfers.shutdown()

    async def _maintenance_loop(self) -> None:
        try:
            await self.get_history_transfers().start()
        except Exception:
            logger.exception("Runtime history transfer recovery failed")
            self._maintenance_error = "Runtime SQLite history transfer recovery failed"
            return
        while self._started:
            try:
                if self._maintenance is None:
                    from .maintenance import RuntimeHistoryMaintenance

                    self._maintenance = RuntimeHistoryMaintenance(self._service())
                await self._maintenance.run_once()
            except Exception:
                # A maintenance fault must not change capture admission or
                # release any durable receipt/maintenance fence.
                self._maintenance_error = "Runtime SQLite history maintenance failed"
                logger.exception("Runtime SQLite history maintenance failed")
            await asyncio.sleep(300)

    async def list_backups(self, workspace_id: str, **filters: Any) -> list[dict[str, Any]]:
        return await self._service().list_backups(workspace_id, **filters)

    async def interrupted_maintenance(self, workspace_id: str) -> dict[str, Any] | None:
        return await self._service().interrupted_maintenance(workspace_id)

    async def accept_capture(self, workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        service = self._service()  # capability check occurs before durable admission
        operation_id = str(UUID(str(payload.get("operation_id", ""))))
        creator_id = str(payload.get("creator_id", "")).strip()
        if not creator_id:
            raise HTTPException(status_code=400, detail="creator_id is required")
        identity_payload = {key: value for key, value in payload.items() if key not in {"request_digest"}}
        request_digest = _digest(identity_payload)
        supplied = payload.get("request_digest")
        if supplied is not None and supplied != request_digest:
            raise HTTPException(status_code=409, detail="Capture request digest conflicts with payload")
        try:
            receipt = await asyncio.to_thread(
                self._store.accept,
                operation_id=operation_id,
                workspace_id=workspace_id,
                creator_id=creator_id,
                request_digest=request_digest,
                kind="capture",
                accepted_payload=identity_payload,
            )
        except (ValueError, OperationConflict) as exc:
            raise HTTPException(status_code=409, detail="Capture operation conflicts with an existing request") from exc
        if receipt["phase"] == "accepted" and operation_id not in self._tasks:
            # The task is created strictly after the durable acceptance write.
            self._tasks[operation_id] = asyncio.create_task(self._run_capture(service, receipt), name=f"sqlite-history-{operation_id}")
            self._tasks[operation_id].add_done_callback(lambda _task: self._tasks.pop(operation_id, None))
        return receipt

    async def _run_capture(self, service: Any, receipt: dict[str, Any]) -> None:
        operation_id = receipt["operation_id"]
        payload = dict(receipt.get("accepted_payload") or {})
        outcomes_by_database: dict[str, Any] = {}
        total_databases: int | None = None

        def terminal_outcomes() -> dict[str, Any]:
            outcomes = dict(outcomes_by_database)
            if total_databases is not None:
                outcomes["progress"] = {"completed": len(outcomes_by_database), "total": total_databases}
            return outcomes

        try:
            with self._store.hold_liveness(operation_id) as liveness_fd:
                global_capture_fd = self._try_global_capture_lock()
                while global_capture_fd is None:
                    if (await asyncio.to_thread(self._store.get, operation_id)).get("phase") == "cancelling":
                        await asyncio.to_thread(self._store.transition, operation_id, "cancelled")
                        return
                    await asyncio.sleep(0.25)
                    global_capture_fd = self._try_global_capture_lock()
                try:
                    initial = await asyncio.to_thread(self._store.get, operation_id)
                    if initial["phase"] == "cancelled":
                        return
                    await asyncio.to_thread(self._store.transition, operation_id, "running")
                    repository = getattr(service, "repository", None)
                    if repository is not None:
                        await repository.initialize(pass_fds=(liveness_fd,))

                    async def cancelled() -> bool:
                        current = await asyncio.to_thread(self._store.get, operation_id)
                        return current["phase"] == "cancelling"

                    async def progress(done: int, total: int) -> None:
                        current = await asyncio.to_thread(self._store.get, operation_id)
                        outcomes = dict(current.get("database_outcomes") or {})
                        outcomes["progress"] = {"completed": done, "total": total}
                        await asyncio.to_thread(self._store.update_progress, operation_id, database_outcomes=outcomes)

                    kwargs = {
                        "trigger": payload.get("trigger", "manual"),
                        "snapshot_id": payload.get("snapshot_id"),
                        "snapshot_git_commit_hash": payload.get("snapshot_git_commit_hash"),
                        "mandatory": bool(payload.get("mandatory", False)),
                        "cancel_check": cancelled,
                        "pass_fds": (liveness_fd,),
                    }
                    # The core service must explicitly support inherited liveness.
                    # Falling back would permit a child to outlive its authority.
                    names = payload.get("database_names")
                    if names is not None and (not isinstance(names, list) or len({str(name) for name in names}) != len(names)):
                        raise HTTPException(status_code=400, detail="database_names must be a unique list or null")
                    if names is None:
                        lease_id = f"history-enumerate-{uuid4()}"
                        access = await self._worker.acquire_sqlite_workspace_access(receipt["workspace_id"], lease_id, maintenance=False)
                        try:
                            names = service._database_names(Path(str(access["authoritative_root"])))
                        finally:
                            await self._worker.release_sqlite_workspace_access(receipt["workspace_id"], lease_id)
                    total_databases = len(names)
                    # Per-database capture is deliberately sequential.  Each
                    # repository adoption identity is deterministic from the parent
                    # operation UUID rather than a request-order-specific token.
                    for index, name in enumerate(names, start=1):
                        if name is not None and (
                            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(name))
                            or not str(name).lower().endswith((".sqlite", ".sqlite3", ".db", ".db3"))
                        ):
                            raise HTTPException(status_code=400, detail="Invalid database name")
                        suboperation_id = self._store.suboperation_id(operation_id, str(name)) if name is not None else operation_id
                        capture_kwargs = {
                            **kwargs,
                            "database_names": {str(name)} if name is not None else None,
                            "capture_job_id": operation_id,
                            "capture_operation_id": suboperation_id,
                            "progress_callback": lambda _done, _total, index=index: progress(index, len(names)),
                        }
                        outcomes = await service.capture_workspace_databases(
                            receipt["workspace_id"],
                            **capture_kwargs,
                        )
                        outcomes_by_database[str(name) if name is not None else "auto"] = {"operation_id": suboperation_id, "results": outcomes}
                        if await cancelled():
                            break
                    current = await asyncio.to_thread(self._store.get, operation_id)
                    phase = current["phase"]
                    failed = any(row.get("status") == "failed" for outcome in outcomes_by_database.values() for row in outcome["results"])
                    final_outcomes = terminal_outcomes()
                    if phase == "cancelling":
                        await asyncio.to_thread(self._store.update_progress, operation_id, database_outcomes=final_outcomes)
                        await asyncio.to_thread(self._store.transition, operation_id, "cancelled")
                    elif failed:
                        await asyncio.to_thread(
                            self._store.transition, operation_id, "failed", database_outcomes=final_outcomes, error="One or more database captures failed"
                        )
                    else:
                        await asyncio.to_thread(self._store.transition, operation_id, "repository_committed", database_outcomes=final_outcomes)
                        await asyncio.to_thread(self._store.transition, operation_id, "catalog_committed")
                        await asyncio.to_thread(self._store.transition, operation_id, "completed")
                finally:
                    fcntl.flock(global_capture_fd, fcntl.LOCK_UN)
                    os.close(global_capture_fd)
        except Exception:
            logger.exception("Runtime history capture failed operation_id=%s", operation_id)
            with contextlib.suppress(OperationConflict, OperationNotFound):
                current = await asyncio.to_thread(self._store.get, operation_id)
                terminal = "cancelled" if current["phase"] == "cancelling" else "failed"
                terminal_kwargs: dict[str, Any] = {"error": "SQLite history capture failed"}
                if total_databases is not None:
                    terminal_kwargs["database_outcomes"] = terminal_outcomes()
                await asyncio.to_thread(self._store.transition, operation_id, terminal, **terminal_kwargs)

    def _try_global_capture_lock(self) -> int | None:
        path = self._root / "_sqlite_history" / "capture.lock"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise HTTPException(status_code=503, detail="Runtime capture admission is unavailable")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise

    async def get_operation(self, workspace_id: str, operation_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._store.get_for_workspace, operation_id, workspace_id)
        except (ValueError, OperationNotFound) as exc:
            raise HTTPException(status_code=404, detail="SQLite history operation not found") from exc

    async def cancel(self, workspace_id: str, operation_id: str) -> dict[str, Any]:
        await self.get_operation(workspace_id, operation_id)
        return await asyncio.to_thread(self._store.request_cancel, operation_id)

    async def acknowledge(self, workspace_id: str, operation_id: str) -> dict[str, Any]:
        await self.get_operation(workspace_id, operation_id)
        return await asyncio.to_thread(self._store.acknowledge, operation_id)

    async def preview(self, workspace_id: str, backup_id: str, **payload: Any) -> dict[str, Any]:
        return await self._service().preview(workspace_id, backup_id, **payload)

    async def apply(self, workspace_id: str, preview_id: str, user_id: str) -> dict[str, Any]:
        # Mutations remain capability and activation gated by _service().
        return await self._service().apply(workspace_id, preview_id, user_id=user_id)

    async def recover(self, workspace_id: str, operation_id: str, action: str) -> dict[str, Any]:
        return await self._service().recover_operation(workspace_id, operation_id, action=action)

    async def delete(self, workspace_id: str, backup_id: str) -> None:
        await self._service().delete(workspace_id, backup_id)

    async def download_path(self, workspace_id: str, backup_id: str) -> Path:
        return await self._service().download_path(workspace_id, backup_id)

    async def claim_due(self, workspace_ids: list[str]) -> list[dict[str, str]]:
        if len(workspace_ids) > 100:
            raise HTTPException(status_code=400, detail="Too many workspace IDs")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", workspace_id) for workspace_id in workspace_ids):
            raise HTTPException(status_code=400, detail="Invalid workspace ID")
        from .maintenance import RuntimeHistoryMaintenance

        return await RuntimeHistoryMaintenance(self._service()).claim_due(workspace_ids)

    async def ack_due(self, workspace_id: str, occurrence_id: str, job_id: str) -> bool:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", workspace_id):
            raise HTTPException(status_code=400, detail="Invalid workspace ID")
        from .maintenance import RuntimeHistoryMaintenance

        return await RuntimeHistoryMaintenance(self._service()).ack_due(workspace_id, occurrence_id, job_id)

    async def begin_guarded_code_restore(self, workspace_id: str, operation_id: str, user_id: str) -> dict[str, Any]:
        service = self._service()
        method = getattr(service, "begin_guarded_code_restore", None)
        if method is None:
            raise HTTPException(status_code=503, detail="Runtime guarded restore capability is unavailable")
        return await method(workspace_id, operation_id=operation_id, user_id=user_id)

    async def finish_guarded_code_restore(self, workspace_id: str, operation_id: str, user_id: str, git_error: str | None) -> dict[str, Any]:
        method = getattr(self._service(), "finish_guarded_code_restore", None)
        if method is None:
            raise HTTPException(status_code=503, detail="Runtime guarded restore capability is unavailable")
        return await method(workspace_id, operation_id, user_id, git_error=git_error)

    async def authorize_git_operation(self, workspace_id: str, operation_id: str) -> None:
        service = self._service()
        await service.verify_guarded_code_restore_lease(workspace_id, operation_id)
