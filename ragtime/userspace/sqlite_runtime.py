"""Runtime coordination for protected User Space SQLite operations.

The durable marker is deliberately independent from the runtime process.  A
crashed control plane must not accidentally restart a workspace whose database
publication may have been interrupted.
"""

from __future__ import annotations

import asyncio
import contextvars
import fcntl
import json
import logging
import os
import re
import stat
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TypeVar
from uuid import uuid4

from fastapi import HTTPException

from ragtime.config import settings
from ragtime.core.runtime_manager_client import runtime_manager_enabled, runtime_manager_request

_WORKSPACE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_MARKER_NAME = "sqlite-maintenance-intent.json"
_MARKER_LOCK_SUFFIX = ".lock"
_OPERATION_LOCK_NAME = "sqlite-operation.lock"
_T = TypeVar("_T")
_logger = logging.getLogger(__name__)

_maintenance_held: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar("_sqlite_maintenance_held", default=frozenset())
_operation_held: contextvars.ContextVar[dict[str, tuple[bool, int, int, asyncio.Task[Any]]]] = contextvars.ContextVar("_sqlite_operation_held", default={})
_recovery_held: contextvars.ContextVar[dict[str, asyncio.Task[Any]]] = contextvars.ContextVar("_sqlite_recovery_held", default={})


def _workspace_dir(workspace_id: str) -> Path:
    if not _WORKSPACE_ID_RE.fullmatch(str(workspace_id or "")):
        raise HTTPException(status_code=400, detail="Invalid workspace ID")
    return Path(settings.index_data_path).resolve() / "_userspace" / "workspaces" / workspace_id


def _canonical_root(workspace_id: str) -> Path:
    return _workspace_dir(workspace_id) / "files"


def _marker_path(workspace_id: str) -> Path:
    return _workspace_dir(workspace_id) / "sqlite_backups" / _MARKER_NAME


def _operation_lock_path(workspace_id: str) -> Path:
    return _workspace_dir(workspace_id) / "sqlite_backups" / _OPERATION_LOCK_NAME


async def run_sqlite_blocking(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run blocking SQLite coordination work without orphaning it on cancellation."""
    context = contextvars.copy_context()
    task = asyncio.create_task(asyncio.to_thread(context.run, func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Do not release a fence or close its descriptor until the kernel call
        # finished.  Repeated cancellation is expected during request teardown.
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                continue
        try:
            task.result()
        except BaseException as exc:
            _logger.error("Cancelled SQLite blocking operation completed with an error", exc_info=exc)
        raise


def _open_regular_lock(path: Path) -> int:
    """Open a lock leaf below a no-follow protected directory chain."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    root_fd = parent_fd = -1
    try:
        root_fd = os.open(path.parent.parent, directory_flags)
        try:
            parent_fd = os.open(path.parent.name, directory_flags, dir_fd=root_fd)
        except FileNotFoundError:
            os.mkdir(path.parent.name, mode=0o700, dir_fd=root_fd)
            parent_fd = os.open(path.parent.name, directory_flags, dir_fd=root_fd)
        fd = os.open(path.name, file_flags, 0o600, dir_fd=parent_fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise OSError("SQLite lock is not a regular file")
        return fd
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=423, detail="SQLite operation lock path is unsafe") from exc
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _acquire_operation_lock(path: Path, *, exclusive: bool, nonblocking: bool = False) -> int:
    fd = _open_regular_lock(path)
    flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if nonblocking:
        flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(fd, flags)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _release_operation_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@asynccontextmanager
async def _sqlite_workspace_operation(workspace_id: str, *, exclusive: bool, nonblocking: bool = False) -> AsyncIterator[None]:
    """Hold a process-wide flock for the full control-plane operation body."""
    held = _operation_held.get()
    current = held.get(workspace_id)
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("SQLite workspace operation requires an asyncio task")
    if current is not None and current[3] is task:
        current_exclusive, depth, fd, _owner = current
        if exclusive and not current_exclusive:
            raise HTTPException(status_code=423, detail="SQLite operation cannot upgrade a shared workspace lock")
        updated = dict(held)
        updated[workspace_id] = (current_exclusive, depth + 1, fd, task)
        token = _operation_held.set(updated)
        try:
            yield
        finally:
            _operation_held.reset(token)
        return

    acquired: list[int] = []

    def acquire() -> int:
        fd = _acquire_operation_lock(_operation_lock_path(workspace_id), exclusive=exclusive, nonblocking=nonblocking)
        acquired.append(fd)
        return fd

    try:
        fd = await run_sqlite_blocking(acquire)
    except BlockingIOError as exc:
        raise HTTPException(status_code=423, detail="SQLite workspace operation is active") from exc
    except asyncio.CancelledError:
        if acquired:
            await run_sqlite_blocking(_release_operation_lock, acquired[0])
        raise
    if task is None:  # pragma: no cover - async context managers always run in a task.
        raise RuntimeError("SQLite workspace operation requires an asyncio task")
    token = _operation_held.set({**held, workspace_id: (exclusive, 1, fd, task)})
    try:
        yield
    finally:
        _operation_held.reset(token)
        await run_sqlite_blocking(_release_operation_lock, fd)


@contextmanager
def _marker_lock(path: Path):
    """Serialize marker ownership changes across control-plane processes."""
    lock_path = path.with_name(f"{path.name}{_MARKER_LOCK_SUFFIX}")
    lock_fd = _open_regular_lock(lock_path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _write_temporary_marker(path: Path, payload: dict[str, object]) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _fsync_marker_directory(path: Path) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_marker(path: Path) -> dict[str, object] | None:
    """Return a validated marker payload, rejecting unsafe marker shapes.

    Symlinks, directories, and malformed markers raise HTTP 423.
    Missing markers return None.
    """
    try:
        marker_stat = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(marker_stat.st_mode):
        raise HTTPException(status_code=423, detail="SQLite maintenance marker is unsafe")

    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        marker_fd = os.open(path, flags)
        with os.fdopen(marker_fd, "r", encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise HTTPException(status_code=423, detail="SQLite maintenance marker is unsafe")
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=423, detail="SQLite maintenance marker is unsafe") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("lease_id"), str):
        raise HTTPException(status_code=423, detail="SQLite maintenance marker is unsafe")
    return payload


# Alias for internal use
_read_regular_marker = read_marker


def _claim_marker(path: Path, payload: dict[str, object]) -> None:
    """Create the initial intent atomically without replacing another lease."""
    with _marker_lock(path):
        if os.path.lexists(path):
            raise HTTPException(status_code=423, detail="SQLite workspace maintenance is already in progress")
        temporary = _write_temporary_marker(path, payload)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise HTTPException(status_code=423, detail="SQLite workspace maintenance is already in progress") from exc
            _fsync_marker_directory(path)
        finally:
            temporary.unlink(missing_ok=True)
            _fsync_marker_directory(path)


def _update_owned_marker(path: Path, lease_id: str, payload: dict[str, object]) -> None:
    """Advance marker state only while the expected lease still owns it."""
    with _marker_lock(path):
        current = _read_regular_marker(path)
        if current is None or current.get("lease_id") != lease_id or payload.get("lease_id") != lease_id:
            raise HTTPException(status_code=423, detail="SQLite maintenance marker ownership was lost")
        temporary = _write_temporary_marker(path, payload)
        try:
            os.replace(temporary, path)
            _fsync_marker_directory(path)
        finally:
            temporary.unlink(missing_ok=True)


def _assert_owned_marker(path: Path, lease_id: str) -> None:
    with _marker_lock(path):
        current = _read_regular_marker(path)
        if current is None or current.get("lease_id") != lease_id:
            raise HTTPException(status_code=423, detail="SQLite maintenance marker ownership was lost")


def _remove_owned_marker(path: Path, lease_id: str) -> bool:
    """Remove a marker only if it is a regular marker for the expected lease."""
    with _marker_lock(path):
        current = _read_regular_marker(path)
        if current is None or current.get("lease_id") != lease_id:
            return False
        path.unlink()
        _fsync_marker_directory(path)
        return True


def _disabled_runtime_evidence(workspace_dir: Path) -> bool:
    """Conservatively reject offline maintenance with a retained runtime tree."""
    rootfs_workspace = workspace_dir / "rootfs" / "workspace"
    try:
        return rootfs_workspace.exists() and any(rootfs_workspace.iterdir())
    except OSError:
        return True


async def assert_sqlite_workspace_maintenance_held(workspace_id: str) -> None:
    """Assert that the current task holds maintenance for the workspace.

    Raises HTTP 423 if maintenance is not held.
    """
    held = _maintenance_held.get()
    if workspace_id not in held:
        raise HTTPException(
            status_code=423,
            detail="SQLite workspace maintenance is not held in current context",
        )


async def assert_sqlite_workspace_available(workspace_id: str) -> None:
    """Reject platform SQLite mutations while a durable maintenance intent exists."""
    marker = _marker_path(workspace_id)
    if os.path.lexists(marker):
        raise HTTPException(
            status_code=423,
            detail="Workspace SQLite maintenance recovery is required before database access",
        )


@asynccontextmanager
async def sqlite_workspace_access(workspace_id: str, *, maintenance: bool = False) -> AsyncIterator[Path]:
    """Yield a pinned authoritative root; never guess after runtime ambiguity."""
    if not _workspace_dir(workspace_id).is_dir():
        raise HTTPException(status_code=404, detail="Workspace not found")
    async with _sqlite_workspace_operation(workspace_id, exclusive=maintenance):
        async with _sqlite_workspace_access_locked(workspace_id, maintenance=maintenance) as root:
            yield root


@asynccontextmanager
async def _sqlite_workspace_access_locked(workspace_id: str, *, maintenance: bool = False) -> AsyncIterator[Path]:
    """Implement access after the local shared/exclusive operation lock is held."""
    workspace_dir = _workspace_dir(workspace_id)
    canonical = _canonical_root(workspace_id)
    if not workspace_dir.is_dir():
        raise HTTPException(status_code=404, detail="Workspace not found")
    await assert_sqlite_workspace_available(workspace_id)
    lease_id = uuid4().hex
    marker = _marker_path(workspace_id)
    if not runtime_manager_enabled():
        if not canonical.is_dir():
            raise HTTPException(status_code=503, detail="Canonical workspace files are unavailable")
        if not maintenance:
            yield canonical
            return
        await run_sqlite_blocking(
            _claim_marker,
            marker,
            {"workspace_id": workspace_id, "lease_id": lease_id, "state": "offline-acquiring", "origin": "offline"},
        )
        if _disabled_runtime_evidence(workspace_dir):
            raise HTTPException(status_code=503, detail="Runtime state is ambiguous while runtime manager is disabled")
        try:
            await run_sqlite_blocking(
                _update_owned_marker,
                marker,
                lease_id,
                {"workspace_id": workspace_id, "lease_id": lease_id, "state": "offline-active", "origin": "offline"},
            )
            offline_token = _maintenance_held.set(_maintenance_held.get() | {workspace_id})
            try:
                yield canonical
            finally:
                _maintenance_held.reset(offline_token)
        except BaseException:
            # An interrupted offline publication is no safer than an online one.
            raise
        else:
            if not await run_sqlite_blocking(_remove_owned_marker, marker, lease_id):
                raise HTTPException(status_code=423, detail="SQLite maintenance marker ownership was lost")
        return

    if maintenance:
        await run_sqlite_blocking(
            _claim_marker,
            marker,
            {"workspace_id": workspace_id, "lease_id": lease_id, "state": "acquiring", "origin": "online"},
        )
    acquired = False
    online_token: contextvars.Token[frozenset[str]] | None = None
    try:
        result = await runtime_manager_request(
            "POST",
            f"/workspaces/{workspace_id}/sqlite-maintenance",
            json_payload={"lease_id": lease_id, "maintenance": maintenance},
            retry_safe=False,
            surface_error_status=True,
        )
        acquired = True
        root = Path(str(result.get("authoritative_root") or "")).resolve()
        allowed = (canonical.resolve(), (workspace_dir / "rootfs" / "workspace").resolve())
        if root not in allowed or not root.is_dir():
            raise HTTPException(status_code=503, detail="Runtime returned an invalid authoritative workspace root")
        if maintenance:
            await run_sqlite_blocking(
                _update_owned_marker,
                marker,
                lease_id,
                {"workspace_id": workspace_id, "lease_id": lease_id, "state": "active", "origin": "online"},
            )
            online_token = _maintenance_held.set(_maintenance_held.get() | {workspace_id})
        try:
            yield root
        finally:
            if online_token is not None:
                _maintenance_held.reset(online_token)
    except BaseException as exc:
        # A clean conflict proves the worker never admitted this maintenance
        # lease; unlike transport/5xx ambiguity it is safe to clear intent.
        if maintenance and not acquired and isinstance(exc, HTTPException) and exc.status_code == 409:
            await run_sqlite_blocking(_remove_owned_marker, marker, lease_id)
        # Leave the durable marker on every uncertain acquire/release path.
        if acquired and not maintenance:
            await runtime_manager_request("DELETE", f"/workspaces/{workspace_id}/sqlite-maintenance/{lease_id}", retry_safe=False)
        raise
    else:
        if acquired:
            await runtime_manager_request("DELETE", f"/workspaces/{workspace_id}/sqlite-maintenance/{lease_id}", retry_safe=False)
        if maintenance:
            if not await run_sqlite_blocking(_remove_owned_marker, marker, lease_id):
                raise HTTPException(status_code=423, detail="SQLite maintenance marker ownership was lost")


@asynccontextmanager
async def sqlite_workspace_recovery(workspace_id: str, lease_id: str) -> AsyncIterator[None]:
    """Own recovery exclusively from receipt handling through marker release.

    A durable marker is crash evidence, not proof that its former holder is
    dead.  The non-blocking exclusive flock is that proof for control-plane
    SQLite operations; manager-origin markers additionally require the manager
    to be available to release the runtime fence.
    """
    marker = _marker_path(workspace_id)
    async with _sqlite_workspace_operation(workspace_id, exclusive=True, nonblocking=True):
        if not os.path.lexists(marker):
            raise HTTPException(status_code=404, detail="SQLite maintenance recovery is not pending")
        payload = await run_sqlite_blocking(read_marker, marker)
        if payload is None or payload.get("lease_id") != lease_id:
            raise HTTPException(status_code=423, detail="SQLite maintenance marker ownership was lost")
        origin = payload.get("origin")
        if origin == "offline":
            if _disabled_runtime_evidence(_workspace_dir(workspace_id)):
                raise HTTPException(
                    status_code=503,
                    detail="Offline SQLite recovery is unsafe while a runtime tree may be active; restore the runtime manager",
                )
        elif origin != "online" or not runtime_manager_enabled():
            raise HTTPException(
                status_code=503,
                detail="SQLite recovery requires the runtime manager for online or ambiguous maintenance state",
            )
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - async context managers always run in a task.
            raise RuntimeError("SQLite recovery requires an asyncio task")
        token = _recovery_held.set({**_recovery_held.get(), workspace_id: task})
        try:
            yield
        finally:
            _recovery_held.reset(token)


async def sqlite_workspace_operation_active(workspace_id: str) -> bool:
    """Advisory liveness probe: true means an operation lock cannot be owned now."""
    if not _workspace_dir(workspace_id).is_dir():
        return False
    if workspace_id in _operation_held.get():
        return True
    try:
        fd = await run_sqlite_blocking(_acquire_operation_lock, _operation_lock_path(workspace_id), exclusive=True, nonblocking=True)
    except (BlockingIOError, OSError):
        return True
    await run_sqlite_blocking(_release_operation_lock, fd)
    return False


async def recover_sqlite_workspace_maintenance(workspace_id: str, lease_id: str, *, action: str) -> None:
    """Release a verified recovery lease without reacquiring its operation lock."""
    if action not in {"complete", "abort"}:
        raise HTTPException(status_code=400, detail="Invalid SQLite maintenance recovery action")
    if _recovery_held.get().get(workspace_id) is not asyncio.current_task():
        async with sqlite_workspace_recovery(workspace_id, lease_id):
            await recover_sqlite_workspace_maintenance(workspace_id, lease_id, action=action)
        return
    marker = _marker_path(workspace_id)
    await run_sqlite_blocking(_assert_owned_marker, marker, lease_id)
    if runtime_manager_enabled():
        await runtime_manager_request("DELETE", f"/workspaces/{workspace_id}/sqlite-maintenance/{lease_id}", retry_safe=False)
    if not await run_sqlite_blocking(_remove_owned_marker, marker, lease_id):
        raise HTTPException(status_code=423, detail="SQLite maintenance marker ownership was lost")
