"""Runtime-private storage helpers shared by capture, maintenance and export.

The repository gate intentionally sits above Restic's own locks.  It prevents a
local prune/check from racing an ingest/materialization while still allowing
ordinary reads and captures to share the gate.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import stat
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import ResticArtifact

_repository_gate_holds: ContextVar[dict[str, tuple[int, bool, int]]] = ContextVar("runtime_sqlite_history_repository_gate_holds", default={})
_repository_gate_waiters: dict[str, int] = {}
_repository_gate_waiter_changed: dict[str, asyncio.Event] = {}


def _repository_gate_key(root: Path) -> str:
    # Do not require the storage root to exist before opening the durable lock.
    # Absolute lexical paths are stable across inherited asyncio/thread contexts.
    return os.path.abspath(os.fspath(root / "_sqlite_history" / "repository-operation.lock"))


def _open_repository_gate(root: Path) -> int:
    directory = root / "_sqlite_history"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("runtime history storage is unsafe")
    path = directory / "repository-operation.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    details = os.fstat(fd)
    path_details = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(details.st_mode) or not os.path.samestat(details, path_details):
        os.close(fd)
        raise RuntimeError("runtime history repository gate is unsafe")
    return fd


def _acquire_repository_gate(root: Path, *, exclusive: bool) -> int:
    fd = _open_repository_gate(root)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _release_repository_gate(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _try_acquire_repository_gate(root: Path, *, exclusive: bool) -> int | None:
    """Try a flock once; callers yield rather than occupying an executor."""
    fd = _open_repository_gate(root)
    try:
        fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise


async def _drain_thread(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, bool]:
    """Do not lose a flock if task cancellation lands around ``to_thread``."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task), False
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result(), True


@contextmanager
def repository_gate(root: Path, *, exclusive: bool) -> Iterator[None]:
    """Cross-process repository/history gate with task-local reentrancy.

    Restic maintenance drains catalog tombstones while it already owns the
    exclusive gate.  Acquiring a second flock descriptor there can block on
    platforms that do not coalesce process-local flock ownership.  Context-local
    depth keeps nested work on the original descriptor; upgrading a shared hold
    remains forbidden so lock ordering stays explicit.
    """
    key = _repository_gate_key(root)
    held = _repository_gate_holds.get().get(key)
    if held is not None:
        fd, held_exclusive, depth = held
        if exclusive and not held_exclusive:
            raise RuntimeError("cannot upgrade shared runtime history repository gate")
        token = _repository_gate_holds.set({**_repository_gate_holds.get(), key: (fd, held_exclusive, depth + 1)})
        try:
            yield
        finally:
            _repository_gate_holds.reset(token)
        return
    fd = _open_repository_gate(root)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        token = _repository_gate_holds.set({**_repository_gate_holds.get(), key: (fd, exclusive, 1)})
        try:
            yield
        finally:
            _repository_gate_holds.reset(token)
    finally:
        _release_repository_gate(fd)


# Export/import consumers use this name to make their wider consistency barrier
# explicit while sharing the one durable gate with normal repository work.
history_export_gate = repository_gate


def held_repository_fds() -> tuple[int, ...]:
    """Keep the history barrier live if a mutating child outlives its parent."""
    return tuple({held[0] for held in _repository_gate_holds.get().values()})


@contextmanager
def _noop() -> Iterator[None]:
    yield


class AsyncRepositoryGate:
    """Cancellable, writer-aware repository gate without executor-held waits."""

    def __init__(self, root: Path, *, exclusive: bool) -> None:
        self._root = root
        self._exclusive = exclusive
        self._token: Any | None = None
        self._fd: int | None = None
        self._owns_fd = False
        self._waiting_writer = False

    @classmethod
    async def try_exclusive(cls, root: Path, *, budget_seconds: float = 0.05) -> "AsyncRepositoryGate | None":
        """Acquire an exclusive gate briefly, withdrawing writer intent on busy.

        Background maintenance uses this instead of joining the normal fair
        writer queue: a long-lived shared guard must not turn one maintenance
        pass into an indefinite admission barrier for unrelated readers.
        """
        if budget_seconds < 0:
            raise ValueError("repository gate budget must not be negative")
        gate = cls(root, exclusive=True)
        if await gate._acquire(budget_seconds=budget_seconds):
            return gate
        return None

    @staticmethod
    async def _backoff() -> None:
        # A short cancellable pause avoids spinning on an external flock.
        await asyncio.sleep(0.01)

    def _notify_waiters(self, key: str) -> None:
        event = _repository_gate_waiter_changed.setdefault(key, asyncio.Event())
        event.set()

    def _finish_waiting_writer(self, key: str) -> None:
        if self._waiting_writer:
            remaining = _repository_gate_waiters.get(key, 1) - 1
            if remaining:
                _repository_gate_waiters[key] = remaining
            else:
                _repository_gate_waiters.pop(key, None)
            self._waiting_writer = False
            self._notify_waiters(key)

    async def _acquire(self, *, budget_seconds: float | None = None) -> bool:
        key = _repository_gate_key(self._root)
        held = _repository_gate_holds.get().get(key)
        if held is not None:
            fd, held_exclusive, depth = held
            if self._exclusive and not held_exclusive:
                raise RuntimeError("cannot upgrade shared runtime history repository gate")
            self._token = _repository_gate_holds.set({**_repository_gate_holds.get(), key: (fd, held_exclusive, depth + 1)})
            return True
        if self._exclusive:
            _repository_gate_waiters[key] = _repository_gate_waiters.get(key, 0) + 1
            self._waiting_writer = True
            self._notify_waiters(key)
        try:
            deadline = None if budget_seconds is None else time.monotonic() + budget_seconds
            while True:
                # Once a local writer is waiting, do not admit fresh readers.
                if not self._exclusive and _repository_gate_waiters.get(key, 0):
                    event = _repository_gate_waiter_changed.setdefault(key, asyncio.Event())
                    event.clear()
                    await event.wait()
                    continue
                acquired_fd = _try_acquire_repository_gate(self._root, exclusive=self._exclusive)
                if acquired_fd is not None:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    self._finish_waiting_writer(key)
                    return False
                await self._backoff()
        except BaseException:
            self._finish_waiting_writer(key)
            raise
        self._finish_waiting_writer(key)
        self._fd = acquired_fd
        self._owns_fd = True
        self._token = _repository_gate_holds.set({**_repository_gate_holds.get(), key: (acquired_fd, self._exclusive, 1)})
        return True

    async def __aenter__(self) -> None:
        acquired = await self._acquire()
        assert acquired

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._token is not None:
            _repository_gate_holds.reset(self._token)
        if self._owns_fd and self._fd is not None:
            # Unlock/close are constant-time syscalls, so release never waits
            # behind a saturated default executor.
            _release_repository_gate(self._fd)
        self._owns_fd = False
        self._fd = None
        self._token = None

    async def aclose(self) -> None:
        """Release a gate returned by :meth:`try_exclusive`."""
        await self.__aexit__(None, None, None)


def restic_artifact(storage: dict[str, object], *, size_bytes: int, sha256: str):
    if storage.get("kind") != "restic":
        raise ValueError("history storage is not Restic")
    return ResticArtifact(
        repository_id=str(storage["repository_id"]),
        snapshot_id=str(storage["snapshot_id"]),
        path=str(storage["path"]),
        size_bytes=int(size_bytes),
        sha256=str(sha256),
    )


def referenced_snapshot_ids(runtime_root: Path, *, receipt_refs: Iterable[dict[str, Any]] = ()) -> set[str]:
    """Return the full logical graph that prevents explicit Restic forget.

    Only catalog rows, previews/maintenance references supplied by the caller,
    and nonterminal receipt references count.  This intentionally returns IDs,
    never repository paths or pack names.
    """
    referenced: set[str] = set()
    workspaces = runtime_root / "workspaces"
    if workspaces.is_dir() and not workspaces.is_symlink():
        for workspace in workspaces.iterdir():
            manifest = workspace / "sqlite_backups" / "manifest-v1.json"
            if not manifest.is_file() or manifest.is_symlink():
                continue
            import json

            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # A corrupt catalog is fail-closed: maintenance must not forget.
                raise RuntimeError("runtime history catalog is unreadable")
            for row in payload.get("backups", []):
                storage = row.get("storage") if isinstance(row, dict) else None
                snapshot_id = storage.get("snapshot_id") if isinstance(storage, dict) else None
                if isinstance(snapshot_id, str):
                    referenced.add(snapshot_id)
    for receipt in receipt_refs:
        if receipt.get("phase") in {"completed", "failed", "cancelled", "interrupted"}:
            continue
        for ref in receipt.get("repository_refs", []):
            snapshot_id = ref.get("snapshot_id") if isinstance(ref, dict) else None
            if isinstance(snapshot_id, str):
                referenced.add(snapshot_id)
    return referenced


def eligible_snapshot_ids(runtime_root: Path, candidates: Iterable[str], *, receipt_refs: Iterable[dict[str, Any]] = ()) -> list[str]:
    """Filter explicit snapshot IDs against the complete durable reference graph."""
    referenced = referenced_snapshot_ids(runtime_root, receipt_refs=receipt_refs)
    return sorted({value for value in candidates if isinstance(value, str) and value not in referenced})
