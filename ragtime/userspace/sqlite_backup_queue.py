"""Durable runner for queued User Space SQLite backup captures."""

from __future__ import annotations

import asyncio
import fcntl
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.userspace.sqlite_capture_admission import _directory_flags, _open_directory_chain, inherit_capture_fds

logger = get_logger(__name__)
_POLL_SECONDS = 1.0
_IDLE_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 5.0)
_HEARTBEAT_SECONDS = 10.0
_TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


def _job_lock_directory() -> int:
    root_fd = _open_directory_chain(Path(settings.index_data_path))
    userspace_fd = directory_fd = -1
    try:
        try:
            userspace_fd = os.open("_userspace", _directory_flags(), dir_fd=root_fd)
        except FileNotFoundError:
            try:
                os.mkdir("_userspace", mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            userspace_fd = os.open("_userspace", _directory_flags(), dir_fd=root_fd)
        try:
            directory_fd = os.open("sqlite_backup_job_locks", _directory_flags(), dir_fd=userspace_fd)
        except FileNotFoundError:
            try:
                os.mkdir("sqlite_backup_job_locks", mode=0o700, dir_fd=userspace_fd)
            except FileExistsError:
                pass
            directory_fd = os.open("sqlite_backup_job_locks", _directory_flags(), dir_fd=userspace_fd)
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise OSError("backup job lock parent is not a directory")
        return directory_fd
    except BaseException:
        if directory_fd >= 0:
            os.close(directory_fd)
        raise
    finally:
        if userspace_fd >= 0:
            os.close(userspace_fd)
        os.close(root_fd)


def _lock_name(job_id: str) -> str:
    # UUID parsing prevents path traversal and keeps filenames canonical.
    return f"{UUID(str(job_id))}.lock"


def _try_job_lock(job_id: str) -> int | None:
    directory_fd = _job_lock_directory()
    try:
        fd = os.open(
            _lock_name(job_id),
            os.O_CREAT | os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("backup job lock is not a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        return fd
    except BaseException:
        os.close(fd)
        raise


def _release_job_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def _held_job_lock(job_id: str) -> Any:
    fd = _try_job_lock(job_id)
    if fd is None:
        yield None
        return
    try:
        yield fd
    finally:
        _release_job_lock(fd)


class SqliteBackupQueueService:
    """One local worker backed by a database-fenced global queue."""

    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            # Keep queue/history imports acyclic during application startup.
            from ragtime.userspace.sqlite_backup_queue_store import SqliteBackupQueueStore

            store = SqliteBackupQueueStore()
        self._store = store
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._owner_token = str(uuid4())
        self._owned_jobs: dict[str, str] = {}
        self._ownership_lost: set[str] = set()
        self._unknown_failures: set[str] = set()
        self._next_maintenance = 0.0

    async def start(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._stopping = False
        try:
            await self.recover_stale()
        except Exception:
            # Startup must not claim work based on an incomplete recovery scan;
            # the runner will retry recovery on its next maintenance pass.
            logger.exception("Could not recover stale SQLite backup jobs at startup")
        self._wake.set()
        self._worker_task = asyncio.create_task(self._run(), name="sqlite-backup-queue")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        # Mark only work this process owns. Running captures observe this at the
        # next database boundary and drain their confined child before finalizing.
        await asyncio.gather(
            *(self._store.cancel(workspace_id, job_id) for job_id, workspace_id in self._owned_jobs.items()),
            return_exceptions=True,
        )
        if self._worker_task is not None:
            await self._worker_task
        self._worker_task = None
        self._owned_jobs.clear()

    async def enqueue(self, workspace_id: str, **kwargs: Any) -> dict[str, Any]:
        job = await self._store.enqueue(workspace_id, **kwargs)
        self._wake.set()
        return job

    async def list_jobs(self, workspace_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._store.list_jobs(workspace_id, **kwargs)

    async def get_job(self, workspace_id: str, job_id: str) -> dict[str, Any] | None:
        return await self._store.get_job(workspace_id, job_id)

    async def cancel(self, workspace_id: str, job_id: str) -> dict[str, Any] | None:
        job = await self._store.cancel(workspace_id, job_id)
        self._wake.set()
        return job

    async def wait_for_job(self, workspace_id: str, job_id: str) -> dict[str, Any] | None:
        while True:
            job = await self.get_job(workspace_id, job_id)
            if job is None or job["status"] in _TERMINAL:
                return job
            await asyncio.sleep(_POLL_SECONDS)

    async def _run(self) -> None:
        idle_backoff_index = 0
        while not self._stopping:
            try:
                now = asyncio.get_running_loop().time()
                if now >= self._next_maintenance:
                    await self.recover_stale()
                    await self.prune_terminal()
                    self._next_maintenance = now + 60.0
                if self._stopping:
                    break
                # Clear before claiming so an enqueue, cancellation, or stop
                # that completes while the store await is in flight remains a
                # wake for the subsequent local wait.
                self._wake.clear()
                claimed = await self._store.claim_next(self._owner_token)
                if claimed is not None:
                    idle_backoff_index = 0
                    await self._run_claimed(claimed)
                    continue
                if self._stopping:
                    break
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=_IDLE_BACKOFF_SECONDS[idle_backoff_index])
                except TimeoutError:
                    idle_backoff_index = min(idle_backoff_index + 1, len(_IDLE_BACKOFF_SECONDS) - 1)
                else:
                    idle_backoff_index = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("SQLite backup queue worker failed")
                idle_backoff_index = 0
                await asyncio.sleep(_POLL_SECONDS)

    async def _run_claimed(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        owner_token = job.get("owner_token") or self._owner_token
        workspace_id = job["workspace_id"]
        self._owned_jobs[job_id] = workspace_id
        self._ownership_lost.discard(job_id)
        self._unknown_failures.discard(job_id)
        lock_fd: int | None = None
        heartbeat_task: asyncio.Task[None] | None = None
        try:
            lock_fd = _try_job_lock(job_id)
            if lock_fd is None:
                # A still-live controller owns the liveness fence; do not
                # disturb it. The durable owner fence prevents our finalizer.
                return
            latest = await self._store.get_job(workspace_id, job_id)
            if latest is None or latest.get("status") != "running" or latest.get("owner_token") != owner_token:
                return
            if self._stopping:
                await self._store.finish(job_id, owner_token, status="cancelled", backup_ids=[])
                return
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(job_id, owner_token))
            outcomes = await self._capture(job, owner_token, lock_fd)
            backup_ids = [outcome["id"] for outcome in outcomes if outcome.get("id") and outcome.get("status") in {"ready", "failed"}]
            if job_id in self._unknown_failures:
                finished = await self._store.finish(
                    job_id,
                    owner_token,
                    status="interrupted",
                    backup_ids=backup_ids,
                    error_message="SQLite backup worker lost database coordination while capture was running",
                )
                if not finished:
                    logger.warning("SQLite backup queue ownership lost before interruption job_id=%s", job_id)
                return
            cancelled = self._stopping or job_id in self._ownership_lost or await self._store.is_cancel_requested(job_id, owner_token)
            has_failure = any(outcome.get("status") == "failed" for outcome in outcomes)
            status = "cancelled" if cancelled else "failed" if has_failure else "completed"
            error_message = "One or more SQLite databases could not be captured" if has_failure else None
            if not await self._store.finish(job_id, owner_token, status=status, backup_ids=backup_ids, error_message=error_message):
                logger.warning("SQLite backup queue ownership lost before completion job_id=%s", job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("SQLite backup queue job failed job_id=%s", job_id)
            try:
                finished = await self._store.finish(
                    job_id,
                    owner_token,
                    status="interrupted",
                    backup_ids=await self._capture_job_backup_ids(workspace_id, job_id),
                    error_message="SQLite backup worker stopped before capture completion",
                )
                if not finished:
                    logger.warning("SQLite backup queue ownership lost before interruption job_id=%s", job_id)
            except Exception:
                logger.exception("Could not mark SQLite backup queue job interrupted job_id=%s", job_id)
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                # A cancellation reaching this finalizer is safe: the awaited
                # capture uses run_sqlite_blocking, which drains its thread and
                # confined child before propagating cancellation. Thus neither
                # this cancellation nor repeated shutdown cancellation releases
                # the job/slot fds while child I/O remains live.
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if lock_fd is not None:
                _release_job_lock(lock_fd)
            self._owned_jobs.pop(job_id, None)
            self._ownership_lost.discard(job_id)
            self._unknown_failures.discard(job_id)

    async def _capture_job_backup_ids(self, workspace_id: str, job_id: str) -> list[str]:
        """Keep catalog records already published before a runner failure visible."""
        from ragtime.userspace.sqlite_history import get_sqlite_history_service

        backups = await get_sqlite_history_service().list_backups(workspace_id)
        return [backup["id"] for backup in backups if backup.get("capture_job_id") == job_id and backup.get("status") in {"ready", "failed"}]

    async def _capture(self, job: dict[str, Any], owner_token: str, lock_fd: int) -> list[dict[str, Any]]:
        from ragtime.userspace.sqlite_history import get_sqlite_history_service

        async def cancel_check() -> bool:
            return (
                self._stopping
                or job["id"] in self._ownership_lost
                or job["id"] in self._unknown_failures
                or await self._store.is_cancel_requested(job["id"], owner_token)
            )

        async def progress_callback(completed: int, total: int) -> None:
            if not await self._store.progress(job["id"], owner_token, completed_databases=completed, total_databases=total):
                # The next database boundary observes the lost owner fence. Do
                # not raise CancelledError here: that would cancel this worker
                # rather than cooperatively draining the current capture.
                self._ownership_lost.add(job["id"])

        with inherit_capture_fds((lock_fd,)):
            return await get_sqlite_history_service().capture_workspace_databases(
                job["workspace_id"],
                trigger=job["trigger"],
                database_names=set(job.get("database_names") or ()) or None,
                snapshot_id=job.get("snapshot_id"),
                snapshot_git_commit_hash=job.get("snapshot_git_commit_hash"),
                capture_job_id=job["id"],
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )

    async def _heartbeat_loop(self, job_id: str, owner_token: str) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            try:
                if not await self._store.heartbeat(job_id, owner_token):
                    self._ownership_lost.add(job_id)
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                # This is not a user cancellation. Keep the liveness fd until
                # the capture reaches its next database boundary, then fence
                # the job as interrupted with a safe, non-sensitive message.
                logger.exception("SQLite backup queue heartbeat failed job_id=%s", job_id)
                self._unknown_failures.add(job_id)
                return

    async def recover_stale(self) -> list[str]:
        """Fence only stale jobs whose liveness lock is actually free."""
        interrupted: list[str] = []
        for job in await self._store.stale_running(older_than_seconds=120, limit=50):
            lock_fd = _try_job_lock(job["id"])
            if lock_fd is None:
                continue
            try:
                owner_token = job.get("owner_token")
                if not isinstance(owner_token, str):
                    logger.warning("SQLite backup queue stale job has no owner token job_id=%s", job["id"])
                    continue
                if await self._store.interrupt(job["id"], owner_token, error_message="Backup worker heartbeat expired"):
                    interrupted.append(job["id"])
            finally:
                _release_job_lock(lock_fd)
        return interrupted

    async def prune_terminal(self) -> list[str]:
        """Prune old metadata; remove a lock file only after proving it free."""
        ids = await self._store.prune_terminal(older_than_days=30, limit=100)
        for job_id in ids:
            lock_fd = _try_job_lock(job_id)
            if lock_fd is None:
                continue
            try:
                directory_fd = _job_lock_directory()
                try:
                    os.unlink(_lock_name(job_id), dir_fd=directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                logger.warning("Could not remove SQLite backup job lock job_id=%s", job_id)
            finally:
                _release_job_lock(lock_fd)
        return ids


_service: SqliteBackupQueueService | None = None


def get_sqlite_backup_queue_service() -> SqliteBackupQueueService:
    global _service
    if _service is None:
        _service = SqliteBackupQueueService()
    return _service
