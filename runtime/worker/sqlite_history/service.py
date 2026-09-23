"""Protected, file-backed SQLite history for User Space workspaces.

The catalog deliberately contains no PostgreSQL state: immutable database blobs
are written before an atomically replaced manifest under the workspace sibling
``sqlite_backups`` directory.  Runtime fencing and recovery semantics live in
the shared runtime/engine contracts; this module owns catalog integrity and the
control-plane orchestration around those contracts.
"""

from __future__ import annotations

import asyncio
import contextvars
import fcntl
import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePath
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Literal
from uuid import uuid4

from fastapi import HTTPException

from runtime.core.secure_files import SecureFileError, delete_file, open_directory, publish_regular_file, sha256_regular_file
from runtime.core.sqlite_capture_admission import (
    inherit_capture_fds,
)
from runtime.core.sqlite_capture_admission import (
    run_admitted_subprocess as _run_admitted_subprocess,
)
from runtime.core.sqlite_history_scratch import managed_scratch_bytes, regular_tree_bytes, remove_orphaned_scratch
from runtime.core.sqlite_recovery import (
    SqliteRecoveryError,
    capture_database,
    database_fingerprint,
    migration_fingerprint,
    prepare_restore,
)
from runtime.core.sqlite_workspace_state import (
    MARKER_NAME,
    SqliteWorkspaceStateError,
    claim_marker,
    remove_owned_marker,
    validate_workspace_id,
    workspace_operation,
)
from runtime.core.sqlite_workspace_state import (
    read_marker as _read_runtime_marker,
)

from .models import RESTIC_IMAGE_PATH, ResticArtifact
from .storage import held_repository_fds

_TRIGGERS = {"manual", "snapshot", "scheduled", "pre_restore"}
_DATABASE_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".db3"}
_RETENTION_DAYS = 30
_MAX_READY_PER_DATABASE = 100
_MAX_WORKSPACE_BYTES = 1 << 30
_PREVIEW_TTL = timedelta(minutes=15)
_PRE_RESTORE_MINIMUM = timedelta(days=7)
_DOWNLOAD_TTL = timedelta(hours=1)

logger = logging.getLogger(__name__)


class _RuntimeHistoryContext:
    """Injected worker lease boundary; deliberately contains no app services."""

    def __init__(self, root: Path, worker: Any) -> None:
        self.root = root
        self.worker = worker

    def history_root(self, workspace_id: str) -> Path:
        return self.root / "workspaces" / validate_workspace_id(workspace_id) / "sqlite_backups"


_runtime_context: contextvars.ContextVar[_RuntimeHistoryContext | None] = contextvars.ContextVar("runtime_sqlite_history_context", default=None)
_maintenance_held: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar("runtime_sqlite_history_maintenance", default=frozenset())
_capture_pass_fds: contextvars.ContextVar[tuple[int, ...]] = contextvars.ContextVar("runtime_sqlite_history_capture_pass_fds", default=())
_workspace_operation_depth: contextvars.ContextVar[dict[str, tuple[bool, int]]] = contextvars.ContextVar(
    "runtime_sqlite_history_workspace_operations", default={}
)


async def run_sqlite_blocking(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Drain blocking work before cancellation releases a kernel fence."""
    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        raise


@asynccontextmanager
async def capture_request_admission() -> AsyncIterator[None]:
    """The child-slot admission is held by ``run_admitted_subprocess`` itself."""
    yield


@asynccontextmanager
async def _workspace_operation(workspace_id: str, *, exclusive: bool) -> AsyncIterator[None]:
    """Use the inspector's shared operation flock for every history access.

    The fd-backed flock is deliberately held outside the thread used for SQLite
    work.  This keeps it live across awaits and avoids the historical gap where
    ordinary captures only held the worker lease.
    """
    context = _runtime_context.get()
    if context is None:
        raise RuntimeError("Runtime SQLite history context is not installed")
    workspace_id = validate_workspace_id(workspace_id)
    held = _workspace_operation_depth.get()
    current = held.get(workspace_id)
    if current is not None:
        held_exclusive, depth = current
        if exclusive and not held_exclusive:
            raise HTTPException(status_code=423, detail="SQLite operation cannot upgrade a shared workspace lock")
        token = _workspace_operation_depth.set({**held, workspace_id: (held_exclusive, depth + 1)})
        try:
            yield
        finally:
            _workspace_operation_depth.reset(token)
        return
    manager = workspace_operation(context.root, workspace_id, exclusive=exclusive)
    await run_sqlite_blocking(manager.__enter__)
    token = _workspace_operation_depth.set({**held, workspace_id: (exclusive, 1)})
    try:
        yield
    finally:
        _workspace_operation_depth.reset(token)
        await run_sqlite_blocking(manager.__exit__, None, None, None)


def run_admitted_subprocess(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    context = _runtime_context.get()
    if context is None:
        raise RuntimeError("Runtime SQLite history context is not installed")
    supplied = tuple(kwargs.pop("pass_fds", ()))
    with inherit_capture_fds(tuple(dict.fromkeys((*_capture_pass_fds.get(), *held_repository_fds(), *supplied)))):
        return _run_admitted_subprocess(context.root, command, **kwargs)


def read_marker(path: Path) -> dict[str, object] | None:
    try:
        return _read_runtime_marker(path)
    except SqliteWorkspaceStateError as exc:
        raise HTTPException(status_code=423, detail="SQLite maintenance marker is unsafe") from exc


@asynccontextmanager
async def sqlite_workspace_access(workspace_id: str, *, maintenance: bool = False) -> AsyncIterator[Path]:
    context = _runtime_context.get()
    if context is None:
        raise RuntimeError("Runtime SQLite history context is not installed")
    workspace_id = validate_workspace_id(workspace_id)
    lease_id = uuid4().hex
    marker = context.history_root(workspace_id) / MARKER_NAME
    if maintenance:
        await run_sqlite_blocking(
            claim_marker,
            marker,
            {"workspace_id": workspace_id, "lease_id": lease_id, "state": "acquiring", "origin": "runtime"},
        )
    acquired = False
    released = False
    body_completed = False
    acquire_rejected = False
    held_token: contextvars.Token[frozenset[str]] | None = None
    try:
        async with _workspace_operation(workspace_id, exclusive=maintenance):
            try:
                result = await context.worker.acquire_sqlite_workspace_access(workspace_id, lease_id, maintenance=maintenance)
            except HTTPException as exc:
                acquire_rejected = exc.status_code == 409
                raise
            acquired = True
            files = Path(str(result["authoritative_root"]))
            if not files.is_dir():
                raise HTTPException(status_code=503, detail="Authoritative runtime workspace is unavailable")
            if maintenance:
                held_token = _maintenance_held.set(_maintenance_held.get() | {workspace_id})
            try:
                yield files
                body_completed = True
            finally:
                if held_token is not None:
                    _maintenance_held.reset(held_token)
                # Release the worker lease before its operation flock.  If release
                # is ambiguous, preserve the marker and retain fail-closed state.
                if acquired:
                    await context.worker.release_sqlite_workspace_access(workspace_id, lease_id)
                    acquired = False
                    released = True
    finally:
        if acquired:
            await context.worker.release_sqlite_workspace_access(workspace_id, lease_id)
            released = True
        # A marker claimed before the worker rejects maintenance is pre-intent
        # state and may be removed.  Once a lease was accepted only a durable
        # terminal receipt is allowed to release it.
        if maintenance and (released or not acquired):
            # Only a terminal receipt may clear this crash marker.  The legacy
            # apply/recovery implementation intentionally leaves it on failure.
            root = context.history_root(workspace_id)
            try:
                manifest = SqliteHistoryService(lambda _wid: files)._load(root, workspace_id)
                terminal = any(
                    row.get("lease_id") == lease_id and row.get("status") in {"completed", "aborted"} for row in manifest.get("operations", {}).values()
                )
            except Exception:
                terminal = False
            if released and (terminal or body_completed):
                await run_sqlite_blocking(remove_owned_marker, marker, lease_id)
            elif acquire_rejected:
                # A definite conflict proves the worker did not accept the lease;
                # transport/cancellation ambiguity must retain the crash fence.
                await run_sqlite_blocking(remove_owned_marker, marker, lease_id)


async def assert_sqlite_workspace_maintenance_held(workspace_id: str) -> None:
    if workspace_id not in _maintenance_held.get():
        raise HTTPException(status_code=423, detail="SQLite maintenance is not held")


async def sqlite_workspace_operation_active(workspace_id: str) -> bool:
    context = _runtime_context.get()
    if context is None:
        return True
    try:
        with workspace_operation(context.root, workspace_id, exclusive=True, nonblocking=True):
            return False
    except (BlockingIOError, SqliteWorkspaceStateError):
        return True


@asynccontextmanager
async def sqlite_workspace_recovery(workspace_id: str, lease_id: str) -> AsyncIterator[None]:
    context = _runtime_context.get()
    if context is None:
        raise RuntimeError("Runtime SQLite history context is not installed")
    with workspace_operation(context.root, workspace_id, exclusive=True, nonblocking=True):
        marker = read_marker(context.history_root(workspace_id) / MARKER_NAME)
        if marker is None or marker.get("lease_id") != lease_id:
            raise HTTPException(status_code=423, detail="SQLite maintenance marker ownership was lost")
        yield


async def recover_sqlite_workspace_maintenance(workspace_id: str, lease_id: str, *, action: str) -> None:
    context = _runtime_context.get()
    if context is None:
        raise RuntimeError("Runtime SQLite history context is not installed")
    removed = await run_sqlite_blocking(remove_owned_marker, context.history_root(workspace_id) / MARKER_NAME, lease_id)
    if not removed:
        raise HTTPException(status_code=423, detail="SQLite maintenance marker ownership was lost")


class _PreIntentFailure(Exception):
    """A validation/safety failure that may safely release maintenance."""

    def __init__(self, error: Exception) -> None:
        self.error = error


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_error(exc: Exception) -> str:
    # Engine errors are deliberately user-safe.  Do not leak filesystem paths.
    text = str(exc).replace("\\", "/")
    return text[:400] or "SQLite history operation failed"


def _validate_database_name(name: str) -> str:
    value = str(name or "").strip()
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid database name")
    if Path(value).suffix.lower() not in _DATABASE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Invalid database name")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_directory(path: Path, *, create: bool = False) -> None:
    """Ensure a controlled history directory is never a link or special file."""
    try:
        details = path.lstat()
    except FileNotFoundError:
        if not create:
            raise HTTPException(status_code=409, detail="SQLite history storage is unavailable")
        path.mkdir(parents=False)
        details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise HTTPException(status_code=409, detail="SQLite history storage is unavailable")


def _history_subdirectory(root: Path, name: str, *, create: bool = False) -> Path:
    _require_regular_directory(root)
    path = root / name
    _require_regular_directory(path, create=create)
    return path


def _require_regular_file(path: Path, *, allow_missing: bool = False) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        if allow_missing and not os.path.lexists(path):
            return False
        raise HTTPException(status_code=409, detail="SQLite history storage is invalid")
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise HTTPException(status_code=409, detail="SQLite history storage is invalid")
    return True


@contextmanager
def _catalog_lock(root: Path, *, blocking: bool = True) -> Iterator[None]:
    _require_regular_directory(root.parent)
    _require_regular_directory(root, create=True)
    lock_path = root / ".lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise HTTPException(status_code=409, detail="SQLite history storage is invalid") from exc
    with os.fdopen(lock_fd, "a+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise HTTPException(status_code=409, detail="SQLite history storage is invalid")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise HTTPException(status_code=503, detail="SQLite history catalog is busy") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class SqliteHistoryService:
    def __init__(self, files_dir_for_workspace: Callable[[str], Path]) -> None:
        self._files_dir_for_workspace = files_dir_for_workspace
        self._scheduler_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._schedule_cursor = 0
        self._next_due_cache: dict[str, datetime] = {}
        self._next_cleanup_cache: dict[str, datetime] = {}

    def _root(self, workspace_id: str) -> Path:
        workspace_id = validate_workspace_id(workspace_id)
        files = self._files_dir_for_workspace(workspace_id)
        try:
            files_details = files.lstat()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail="SQLite history storage is unavailable") from exc
        if stat.S_ISLNK(files_details.st_mode) or not stat.S_ISDIR(files_details.st_mode):
            raise HTTPException(status_code=409, detail="SQLite history storage is unavailable")
        root = files.parent / "sqlite_backups"
        # ``files`` itself must be a real controlled workspace directory.  Do
        # not follow a user-created history symlink outside that workspace.
        if root.exists() and root.is_symlink():
            raise HTTPException(status_code=409, detail="SQLite history storage is unavailable")
        return root

    @staticmethod
    def _manifest_path(root: Path) -> Path:
        return root / "manifest-v1.json"

    @staticmethod
    def _protected_path(root: Path, relative: str, directory: str) -> Path:
        """Return a direct child only after no-follow parent/leaf validation."""
        _require_regular_directory(root)
        raw_parent = root / directory
        try:
            parent_details = raw_parent.lstat()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail="SQLite history storage is invalid") from exc
        if stat.S_ISLNK(parent_details.st_mode) or not stat.S_ISDIR(parent_details.st_mode):
            raise HTTPException(status_code=409, detail="SQLite history storage is invalid")
        parts = PurePath(str(relative or "")).parts
        if len(parts) != 2 or parts[0] != directory or parts[1] in {"", ".", ".."}:
            raise HTTPException(status_code=409, detail="SQLite history storage is invalid")
        candidate = raw_parent / parts[1]
        try:
            details = candidate.lstat()
        except FileNotFoundError:
            return candidate
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise HTTPException(status_code=409, detail="SQLite history storage is invalid")
        return candidate

    def _load(self, root: Path, workspace_id: str) -> dict[str, Any]:
        path = self._manifest_path(root)
        if not _require_regular_file(path, allow_missing=True):
            return {
                "version": 1,
                "workspace_id": workspace_id,
                "backups": [],
                "previews": {},
                "operations": {},
                "last_scheduled_at": None,
                "next_scheduled_at": None,
            }
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("workspace_id") != workspace_id:
                raise ValueError
            value.setdefault("backups", [])
            value.setdefault("previews", {})
            value.setdefault("operations", {})
            value.setdefault("next_scheduled_at", None)
            return value
        except Exception as exc:
            raise HTTPException(status_code=409, detail="SQLite history catalog is unreadable") from exc

    def _save(self, root: Path, manifest: dict[str, Any]) -> None:
        _require_regular_directory(root.parent)
        _require_regular_directory(root, create=True)
        _require_regular_file(self._manifest_path(root), allow_missing=True)
        fd, temp_name = tempfile.mkstemp(prefix="manifest-", suffix=".json", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(manifest, out, sort_keys=True, separators=(",", ":"))
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp_name, self._manifest_path(root))
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def _backup_response(self, row: dict[str, Any], *, can_delete: bool = True) -> dict[str, Any]:
        return {**row, "can_restore": row.get("status") == "ready", "can_delete": can_delete}

    @staticmethod
    def _protected_backup_ids(manifest: dict[str, Any]) -> set[str]:
        """One policy for list/delete/retention/quota eviction protection."""
        protected = {str(item.get("backup_id")) for item in manifest.get("previews", {}).values()}
        for operation in manifest.get("operations", {}).values():
            if operation.get("status") == "intent":
                protected.update(str(operation.get(key)) for key in ("safety_backup_id", "backup_id") if operation.get(key))
        ready: list[dict[str, Any]] = []
        newest_by_database: dict[str, dict[str, Any]] = {}
        for row in manifest.get("backups", []):
            if row.get("status") != "ready":
                continue
            ready.append(row)
            name = str(row.get("database_name"))
            newest = newest_by_database.get(name)
            if newest is None or str(row.get("created_at")) > str(newest.get("created_at")):
                newest_by_database[name] = row
        protected.update(str(row["id"]) for row in newest_by_database.values())
        cutoff = _now() - _PRE_RESTORE_MINIMUM
        protected.update(str(row["id"]) for row in ready if row.get("trigger") == "pre_restore" and datetime.fromisoformat(row["created_at"]) >= cutoff)
        return protected

    async def get_snapshot_ids_with_backups(self, workspace_id: str) -> set[str]:
        """Return the set of snapshot IDs that have at least one backup in the manifest.

        Returns an empty set when the SQLite history storage is unavailable or the
        manifest does not exist yet, so this is safe to call unconditionally.
        """
        try:
            root = self._root(workspace_id)
        except Exception:
            return set()

        def read() -> set[str]:
            try:
                with _catalog_lock(root):
                    manifest = self._load(root, workspace_id)
                    return {str(row["snapshot_id"]) for row in manifest.get("backups", []) if row.get("snapshot_id")}
            except Exception:
                return set()

        return await run_sqlite_blocking(read)

    async def list_backups(self, workspace_id: str, *, database_name: str | None = None, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        root = self._root(workspace_id)

        def read() -> list[dict[str, Any]]:
            if not os.path.lexists(root):
                return []
            _require_regular_directory(root)
            # Writers publish the whole manifest by atomic replace. A listing is
            # advisory and may safely observe the previous complete generation;
            # mutations revalidate under the catalog lock. Do not queue every
            # SSE poll behind a multi-minute Restic capture/materialization.
            manifest = self._load(root, workspace_id)
            rows = manifest["backups"]
            if database_name:
                rows = [row for row in rows if row["database_name"] == _validate_database_name(database_name)]
            if snapshot_id:
                rows = [row for row in rows if row.get("snapshot_id") == snapshot_id]
            protected = self._protected_backup_ids(manifest)
            return [
                self._backup_response(dict(row), can_delete=row.get("id") not in protected)
                for row in sorted(rows, key=lambda row: row["created_at"], reverse=True)
            ]

        return await run_sqlite_blocking(read)

    async def interrupted_maintenance(self, workspace_id: str) -> dict[str, Any] | None:
        if await sqlite_workspace_operation_active(workspace_id):
            return {"state": "active", "operation_id": None, "detail": "SQLite workspace operation is active", "can_complete": False, "can_abort": False}
        root = self._root(workspace_id)

        def read() -> dict[str, Any] | None:
            with _catalog_lock(root):
                manifest = self._load(root, workspace_id)
                operations = manifest["operations"]
                pending = next(((operation_id, row) for operation_id, row in operations.items() if row.get("status") == "intent"), None)
                if not pending:
                    marker = root / "sqlite-maintenance-intent.json"
                    # Use the hardened reader so an unsafe marker (symlink,
                    # directory, malformed JSON, missing lease) surfaces as an
                    # operator-recoverable state instead of silently vanishing
                    # while assert_sqlite_workspace_available still fail-closes.
                    try:
                        payload = read_marker(marker)
                    except HTTPException:
                        return {
                            "state": "invalid",
                            "operation_id": "unknown",
                            "detail": "SQLite maintenance fence is invalid and requires operator recovery",
                            "can_complete": False,
                            "can_abort": False,
                        }
                    if payload is None:
                        return None
                    lease_id = str(payload.get("lease_id") or "")
                    if not lease_id:
                        return {
                            "state": "invalid",
                            "operation_id": "unknown",
                            "detail": "SQLite maintenance fence is invalid and requires operator recovery",
                            "can_complete": False,
                            "can_abort": False,
                        }
                    # A marker left behind after a terminal receipt means the
                    # operation already finished but the idempotent release did
                    # not complete.  Surface it as a release-retry of the SAME
                    # terminal action so the operator can clear the fence without
                    # republishing; recover_operation handles this idempotently.
                    terminal = next(
                        (
                            (operation_id, row)
                            for operation_id, row in operations.items()
                            if row.get("lease_id") == lease_id and row.get("status") in {"completed", "aborted"}
                        ),
                        None,
                    )
                    if terminal is not None:
                        terminal_id, terminal_row = terminal
                        aborted = terminal_row.get("status") == "aborted"
                        return {
                            "state": "release_pending",
                            "operation_id": terminal_id,
                            "detail": "SQLite maintenance finished but the runtime fence still needs to be released",
                            "can_complete": not aborted,
                            "can_abort": aborted,
                        }
                    return {
                        "state": "interrupted",
                        "operation_id": lease_id,
                        "detail": "SQLite maintenance was interrupted before a restore operation was recorded",
                        "can_complete": False,
                        "can_abort": True,
                    }
                operation_id, row = pending
                # Abort is safe only before publication.  A persisted
                # publication-state marker makes the UI require completion.
                state = str(row.get("publication_state") or "prepublication")
                return {
                    "state": "interrupted",
                    "operation_id": operation_id,
                    "detail": "SQLite restore maintenance was interrupted",
                    "can_complete": True,
                    "can_abort": state == "prepublication",
                }

        return await run_sqlite_blocking(read)

    async def run_maintenance_once(self) -> None:
        """Run a bounded, fair maintenance pass; cleanup is not capture-gated."""
        from ragtime.userspace.service import userspace_service

        workspaces = (
            sorted(path.name for path in (userspace_service.root_path / "workspaces").iterdir() if path.is_dir() and not path.is_symlink())
            if (userspace_service.root_path / "workspaces").is_dir()
            else []
        )
        active = set(workspaces)
        self._next_due_cache = {workspace_id: due for workspace_id, due in self._next_due_cache.items() if workspace_id in active}
        self._next_cleanup_cache = {workspace_id: due for workspace_id, due in self._next_cleanup_cache.items() if workspace_id in active}
        if workspaces:
            offset = self._schedule_cursor % len(workspaces)
            workspaces = workspaces[offset:] + workspaces[:offset]
            self._schedule_cursor = (offset + 1) % len(workspaces)
        deadline = asyncio.get_running_loop().time() + 30
        admitted = 0
        for workspace_id in workspaces:
            if admitted >= 8 or asyncio.get_running_loop().time() >= deadline:
                break
            try:
                now = _now()
                next_due = self._next_due_cache.get(workspace_id)
                next_cleanup = self._next_cleanup_cache.get(workspace_id)
                # A fully initialized workspace needs no filesystem/catalog work
                # until one cached deadline is reached.
                if next_due is not None and next_cleanup is not None and next_due > now and next_cleanup > now:
                    continue
                root = self._root(workspace_id)
                # The liveness flock is deliberately held across the durable
                # claim, child capture, and completion.  A stale claim is only
                # recoverable after this lock has proved its former owner gone.
                liveness = await self._acquire_scheduled_liveness(root)
                if liveness is None:
                    continue
                try:
                    cleanup_due = self._next_cleanup_cache.get(workspace_id, now) <= now
                    if cleanup_due:
                        due = await run_sqlite_blocking(self._cleanup_and_due_sync, root, workspace_id, True)
                        self._next_cleanup_cache[workspace_id] = now + timedelta(hours=1)
                        self._next_due_cache[workspace_id] = await run_sqlite_blocking(self._next_scheduled_due_sync, root, workspace_id, True)
                    else:
                        due = self._next_due_cache.get(workspace_id, now) <= now
                    claim = await run_sqlite_blocking(self._claim_scheduled_due_sync, root, workspace_id, True, True) if due else None
                    if not claim:
                        self._next_due_cache[workspace_id] = await run_sqlite_blocking(self._next_scheduled_due_sync, root, workspace_id, True)
                        continue
                    try:
                        # Capture execution belongs to the durable queue.  Keep
                        # the manifest claim only until the enqueue is accepted;
                        # the queue owns retries and later failure visibility.
                        from ragtime.userspace.sqlite_backup_queue import get_sqlite_backup_queue_service

                        await get_sqlite_backup_queue_service().enqueue(
                            workspace_id,
                            trigger="scheduled",
                            request_key=f"scheduled:{workspace_id}:{claim}",
                        )
                    except Exception:
                        # Do not advance the manifest until a durable queue row
                        # exists.  Releasing this claim makes the next scheduler
                        # pass retry the enqueue without performing SQLite I/O.
                        raise
                    else:
                        admitted += 1
                        await run_sqlite_blocking(self._complete_scheduled_attempt_sync, root, workspace_id, claim, True)
                        self._next_due_cache[workspace_id] = _now() + timedelta(hours=1)
                finally:
                    await run_sqlite_blocking(self._release_scheduled_liveness_lock, liveness)
            except Exception as exc:
                # Runtime may be unavailable; the next hourly pass retries.
                logger.warning(
                    "SQLite history scheduled maintenance failed workspace_id=%s error_type=%s",
                    workspace_id,
                    type(exc).__name__,
                    exc_info=True,
                )
                continue

    @staticmethod
    def _try_scheduled_liveness_lock(root: Path) -> Any | None:
        _require_regular_directory(root.parent)
        _require_regular_directory(root, create=True)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        handle: Any | None = None
        try:
            fd = os.open(root / ".scheduled-attempt.lock", flags, 0o600)
            handle = os.fdopen(fd, "a+b")
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                handle.close()
                return None
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except (OSError, BlockingIOError):
            if handle is not None:
                handle.close()
            return None

    async def _acquire_scheduled_liveness(self, root: Path) -> Any | None:
        """Release a thread-acquired flock even if this task is cancelled."""
        task = asyncio.create_task(run_sqlite_blocking(self._try_scheduled_liveness_lock, root))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # `run_sqlite_blocking` drains its worker; do the same here so a
            # cancellation cannot strand a flock acquired just after cancel.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            if not task.cancelled() and task.exception() is None and task.result() is not None:
                release = asyncio.create_task(run_sqlite_blocking(self._release_scheduled_liveness_lock, task.result()))
                while not release.done():
                    try:
                        await asyncio.shield(release)
                    except asyncio.CancelledError:
                        continue
                release.result()
            raise

    @staticmethod
    def _release_scheduled_liveness_lock(handle: Any) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    @staticmethod
    def _scheduled_initial_due(workspace_id: str, now: datetime) -> datetime:
        return now + timedelta(seconds=int(hashlib.sha256(workspace_id.encode()).hexdigest()[:8], 16) % 301)

    def _claim_scheduled_due_sync(self, root: Path, workspace_id: str, try_lock: bool = False, recover_orphan_claim: bool = False) -> str | None:
        """Atomically decide/claim a due run so replicas cannot overlap."""
        with _catalog_lock(root, blocking=not try_lock):
            manifest = self._load(root, workspace_id)
            now = _now()
            claim = manifest.get("scheduled_claim")
            if claim and not recover_orphan_claim:
                return None
            next_due = manifest.get("next_scheduled_at")
            if next_due is None:
                # Legacy manifests that already have a successful timestamp keep
                # their hourly cadence; first-seen workspaces get only jitter.
                previous = manifest.get("last_scheduled_at")
                next_time = (datetime.fromisoformat(previous) + timedelta(hours=1)) if previous else self._scheduled_initial_due(workspace_id, now)
                manifest["next_scheduled_at"] = next_time.isoformat()
                self._save(root, manifest)
                return None
            if datetime.fromisoformat(next_due) > now:
                return None
            claim_id = uuid4().hex
            manifest["scheduled_claim"] = {"claimed_at": now.isoformat(), "claim_id": claim_id}
            self._save(root, manifest)
            return claim_id

    def _next_scheduled_due_sync(self, root: Path, workspace_id: str, try_lock: bool = False) -> datetime:
        with _catalog_lock(root, blocking=not try_lock):
            manifest = self._load(root, workspace_id)
            value = manifest.get("next_scheduled_at")
            return datetime.fromisoformat(value) if value else self._scheduled_initial_due(workspace_id, _now())

    def _complete_scheduled_attempt_sync(self, root: Path, workspace_id: str, claim_id: str, success: bool) -> bool:
        with _catalog_lock(root):
            manifest = self._load(root, workspace_id)
            claim = manifest.get("scheduled_claim") or {}
            if claim.get("claim_id") != claim_id:
                return False
            now = _now()
            manifest.pop("scheduled_claim", None)
            if success:
                manifest["last_scheduled_at"] = now.isoformat()
            manifest["next_scheduled_at"] = (now + (timedelta(hours=1) if success else timedelta(minutes=5))).isoformat()
            self._save(root, manifest)
            return True

    def _cleanup_and_due_sync(self, root: Path, workspace_id: str, try_lock: bool = False) -> bool:
        with _catalog_lock(root, blocking=not try_lock):
            # Capture children can be killed without running their context
            # cleanup.  The owner flock is the liveness proof; never age-delete.
            remove_orphaned_scratch(root)
            blob_root = root / "blobs"
            if blob_root.is_dir() and not blob_root.is_symlink():
                remove_orphaned_scratch(blob_root)
            manifest = self._load(root, workspace_id)
            before = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            now = _now()
            referenced = {str(row.get("blob")) for row in manifest["backups"] if row.get("blob")}
            previews = manifest["previews"]
            operations = manifest["operations"]
            active_preview_ids = {str(row.get("preview_id")) for row in operations.values() if row.get("status") == "intent" and row.get("preview_id")}
            expired_candidates: set[str] = set()
            for preview_id, preview in list(previews.items()):
                if datetime.fromisoformat(preview["expires_at"]) <= now:
                    if preview_id in active_preview_ids:
                        continue
                    expired_candidates.add(str(preview["candidate"]))
                    previews.pop(preview_id)
            candidate_refs = (
                referenced
                | expired_candidates
                | {str(row.get("candidate")) for row in previews.values()}
                | {str(row.get("candidate")) for row in operations.values() if row.get("status") == "intent"}
            )
            for directory in (root / "blobs", root / "candidates"):
                if directory.is_dir() and not directory.is_symlink():
                    for entry in directory.iterdir():
                        rel = entry.relative_to(root).as_posix()
                        if entry.is_file() and not entry.is_symlink() and rel not in candidate_refs:
                            entry.unlink()
            download_dir = root / "downloads"
            if download_dir.is_dir() and not download_dir.is_symlink():
                for entry in download_dir.iterdir():
                    if entry.is_file() and not entry.is_symlink() and datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc) < now - _DOWNLOAD_TTL:
                        entry.unlink()
            removed = self._prune_locked(root, manifest)
            next_due = manifest.get("next_scheduled_at")
            if next_due is None:
                previous = manifest.get("last_scheduled_at")
                next_time = datetime.fromisoformat(previous) + timedelta(hours=1) if previous else self._scheduled_initial_due(workspace_id, now)
                manifest["next_scheduled_at"] = next_time.isoformat()
                due = False
            else:
                due = datetime.fromisoformat(next_due) <= now
            if json.dumps(manifest, sort_keys=True, separators=(",", ":")) != before:
                self._save(root, manifest)
            for candidate in expired_candidates:
                self._protected_path(root, candidate, "candidates").unlink(missing_ok=True)
            self._unlink_unreferenced(root, manifest, removed)
            return due

    def start(self) -> None:
        if self._scheduler_task is None or self._scheduler_task.done():
            self._stopping = False
            self._scheduler_task = asyncio.create_task(self._scheduler(), name="sqlite-history-maintenance")

    async def stop(self) -> None:
        self._stopping = True
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

    async def _scheduler(self) -> None:
        while not self._stopping:
            await self.run_maintenance_once()
            await asyncio.sleep(60)

    async def capture_workspace_databases(
        self,
        workspace_id: str,
        *,
        trigger: Literal["manual", "snapshot", "scheduled", "pre_restore"],
        snapshot_id: str | None = None,
        snapshot_git_commit_hash: str | None = None,
        mandatory: bool = False,
        database_names: set[str] | None = None,
        files_dir: Path | None = None,
        capture_job_id: str | None = None,
        capture_operation_id: str | None = None,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> list[dict[str, Any]]:
        workspace_id = validate_workspace_id(workspace_id)
        normalized_fds = tuple(dict.fromkeys(fd for fd in pass_fds if isinstance(fd, int) and fd >= 0))
        inherited_token = _capture_pass_fds.set(tuple(dict.fromkeys((*_capture_pass_fds.get(), *normalized_fds))))
        try:
            return await self._capture_workspace_databases(
                workspace_id,
                trigger=trigger,
                snapshot_id=snapshot_id,
                snapshot_git_commit_hash=snapshot_git_commit_hash,
                mandatory=mandatory,
                database_names=database_names,
                files_dir=files_dir,
                capture_job_id=capture_job_id,
                capture_operation_id=capture_operation_id,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
        finally:
            _capture_pass_fds.reset(inherited_token)

    async def _capture_workspace_databases(
        self,
        workspace_id: str,
        *,
        trigger: Literal["manual", "snapshot", "scheduled", "pre_restore"],
        snapshot_id: str | None = None,
        snapshot_git_commit_hash: str | None = None,
        mandatory: bool = False,
        database_names: set[str] | None = None,
        files_dir: Path | None = None,
        capture_job_id: str | None = None,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
        capture_operation_id: str | None = None,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        if trigger not in _TRIGGERS:
            raise ValueError("invalid history trigger")
        if files_dir is None:
            # sqlite_workspace_access already calls assert_sqlite_workspace_available internally.
            async with capture_request_admission():
                async with sqlite_workspace_access(workspace_id) as pinned_files_dir:
                    return await self._capture_workspace_databases(
                        workspace_id,
                        trigger=trigger,
                        snapshot_id=snapshot_id,
                        snapshot_git_commit_hash=snapshot_git_commit_hash,
                        mandatory=mandatory,
                        database_names=database_names,
                        files_dir=pinned_files_dir,
                        capture_job_id=capture_job_id,
                        capture_operation_id=capture_operation_id,
                        cancel_check=cancel_check,
                        progress_callback=progress_callback,
                    )
        else:
            if trigger == "pre_restore":
                await assert_sqlite_workspace_maintenance_held(workspace_id)
            root = self._root(workspace_id)
            names = self._database_names(files_dir)
            if database_names is not None:
                wanted = {_validate_database_name(name) for name in database_names}
                names = [name for name in names if name in wanted]
            results: list[dict[str, Any]] = []
            total = len(names)
            for name in names:
                # Queue cancellation is cooperative at database boundaries.
                # Database/store failures deliberately propagate: treating an
                # unknown cancellation state as a user cancellation can hide a
                # capture that may still be in progress elsewhere.
                if trigger != "pre_restore" and cancel_check is not None and await cancel_check():
                    break
                try:
                    results.append(
                        await run_sqlite_blocking(
                            self._capture_one,
                            workspace_id,
                            root,
                            files_dir,
                            name,
                            trigger,
                            snapshot_id,
                            snapshot_git_commit_hash,
                            capture_job_id,
                            capture_operation_id,
                        )
                    )
                except Exception as exc:
                    failed = await run_sqlite_blocking(
                        self._record_failure,
                        workspace_id,
                        root,
                        name,
                        trigger,
                        snapshot_id,
                        snapshot_git_commit_hash,
                        exc,
                        capture_job_id,
                    )
                    results.append(failed)
                    if mandatory:
                        raise HTTPException(status_code=409, detail="Mandatory SQLite safety backup failed") from exc
                if progress_callback is not None:
                    await progress_callback(len(results), total)
            return results

    @staticmethod
    def _database_names(files_dir: Path) -> list[str]:
        try:
            with open_directory(files_dir, ".ragtime/db") as database_fd:
                names = []
                for name in os.listdir(database_fd):
                    if Path(name).suffix.lower() not in _DATABASE_SUFFIXES:
                        continue
                    try:
                        entry = os.stat(name, dir_fd=database_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISREG(entry.st_mode):
                        names.append(name)
                return sorted(names)
        except (FileNotFoundError, SecureFileError):
            return []

    def _capture_one(
        self,
        workspace_id: str,
        root: Path,
        files_dir: Path,
        name: str,
        trigger: str,
        snapshot_id: str | None,
        commit: str | None,
        capture_job_id: str | None = None,
        capture_operation_id: str | None = None,
    ) -> dict[str, Any]:
        with _catalog_lock(root):
            return self._capture_one_locked(workspace_id, root, files_dir, name, trigger, snapshot_id, commit, capture_job_id, capture_operation_id)

    def _capture_one_locked(
        self,
        workspace_id: str,
        root: Path,
        files_dir: Path,
        name: str,
        trigger: str,
        snapshot_id: str | None,
        commit: str | None,
        capture_job_id: str | None = None,
        capture_operation_id: str | None = None,
    ) -> dict[str, Any]:
        _validate_database_name(name)
        backup_id = str(uuid4())
        blob_dir = _history_subdirectory(root, "blobs", create=True)
        destination = blob_dir / f"{backup_id}.sqlite3"
        manifest = self._load(root, workspace_id)
        if capture_job_id is not None:
            existing = next(
                (row for row in manifest["backups"] if row.get("capture_job_id") == capture_job_id and row.get("database_name") == name),
                None,
            )
            if existing is not None:
                return self._backup_response(dict(existing))
        # A matching source token is only a hint until the immutable blob's
        # recorded checksum and size are revalidated under this catalog lock.
        # Safety captures intentionally do not shortcut this full capture.
        has_token_candidate = any(
            row.get("status") == "ready" and row.get("database_name") == name and isinstance(row.get("source_token"), str) for row in manifest["backups"]
        )
        source_token = self._probe_confined(files_dir, name) if trigger != "pre_restore" and has_token_candidate else None
        reusable = self._latest_reusable_blob(root, manifest, name, source_token)
        if reusable is not None:
            if trigger == "scheduled":
                logger.info("SQLite history capture outcome=skipped_unchanged workspace_id=%s database_name=%s", workspace_id, name)
                return {"outcome": "skipped_unchanged", "database_name": name, "source_token": source_token}
            row = self._new_alias_row(workspace_id, name, trigger, snapshot_id, commit, reusable, source_token, capture_job_id)
            manifest["backups"].append(row)
            removed = self._prune_locked(root, manifest)
            self._save(root, manifest)
            self._unlink_unreferenced(root, manifest, removed)
            logger.info("SQLite history capture outcome=reused workspace_id=%s database_name=%s logical_bytes=%s", workspace_id, name, row["size_bytes"])
            return self._backup_response(row)
        # Free ordinary expired history before rejecting a safe new capture.
        removed = self._prune_locked(root, manifest)
        self._enforce_quota(root, manifest, self._database_size_estimate(files_dir, name), planned_removed=removed)
        # Eviction is durable before expensive source work: a capture failure
        # cannot leave catalog rows pointing at blobs already evicted.
        self._save(root, manifest)
        self._unlink_unreferenced(root, manifest, removed)
        capture = self._capture_confined(files_dir, name, blob_dir, destination.name)
        # A WAL-backed source can be materially larger than its main file.  The
        # reservation above is conservative; produced bytes are authoritative.
        try:
            # The produced blob already contributes to disk usage; use zero
            # additional reservation for the authoritative post-publication
            # check while retaining candidates/downloads/scratch in the total.
            self._enforce_quota(root, manifest, int(capture["size_bytes"]))
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        row = {
            "id": backup_id,
            "workspace_id": workspace_id,
            "database_name": name,
            "created_at": _now().isoformat(),
            "trigger": trigger,
            "snapshot_id": snapshot_id,
            "snapshot_git_commit_hash": commit,
            "status": "ready",
            "size_bytes": int(capture["size_bytes"]),
            "sha256": str(capture["sha256"]),
            "error": None,
            "blob": f"blobs/{destination.name}",
            "fingerprint": capture.get("fingerprint"),
            "schema_hash": capture.get("schema_hash"),
            "source_token": capture.get("source_token"),
            "capture_job_id": capture_job_id,
        }
        # A pre-restore capture always reaches this point: it has made a fresh,
        # full SQLite image.  Revalidate a matching immutable image before
        # ingesting so an unchanged capture aliases its verified storage instead
        # of creating an unreferenced Restic snapshot.
        duplicate = self._latest_reusable_blob(root, manifest, name, None, sha256=row["sha256"], size=row["size_bytes"])
        if duplicate is not None:
            if isinstance(duplicate.get("storage"), dict):
                row["storage"] = dict(duplicate["storage"])
                row["blob"] = None
            else:
                row["blob"] = duplicate["blob"]
        else:
            storage = self._store_captured_image(workspace_id, capture_operation_id or capture_job_id or backup_id, destination, row)
            if storage is not None:
                # Do not publish a ready row until the repository's verified
                # snapshot exists. ``blob`` remains absent for Restic rows so
                # legacy readers cannot treat a temporary image as authoritative.
                row["storage"] = storage
                row["blob"] = None
        manifest["backups"].append(row)
        removed = self._prune_locked(root, manifest)
        self._save(root, manifest)
        if duplicate is not None or (isinstance(row.get("storage"), dict) and row["storage"].get("kind") == "restic"):
            # The capture file was a private immutable ingest staging image.  It
            # may be removed only after the catalog reference is durable.
            destination.unlink(missing_ok=True)
        self._unlink_unreferenced(root, manifest, removed)
        logger.info(
            "SQLite history capture outcome=%s workspace_id=%s database_name=%s new_blob_bytes=%s logical_bytes=%s",
            "reused" if duplicate else "captured",
            workspace_id,
            name,
            0 if duplicate else row["size_bytes"],
            row["size_bytes"],
        )
        return self._backup_response(row)

    def _store_captured_image(self, workspace_id: str, operation_id: str, destination: Path, row: dict[str, Any]) -> dict[str, Any] | None:
        """Storage hook. Legacy service keeps its verified blob; runtime overrides."""
        return None

    def _latest_reusable_blob(
        self, root: Path, manifest: dict[str, Any], name: str, token: str | None, *, sha256: str | None = None, size: int | None = None
    ) -> dict[str, Any] | None:
        if token is None and sha256 is None:
            return None
        rows = sorted(
            (row for row in manifest["backups"] if row.get("status") == "ready" and row.get("database_name") == name),
            key=lambda row: str(row.get("created_at")),
            reverse=True,
        )
        validation_cache: dict[tuple[str, Any, int], bool] = {}
        for row in rows:
            if token is not None and row.get("source_token") != token:
                continue
            if sha256 is not None and (row.get("sha256") != sha256 or int(row.get("size_bytes") or -1) != size):
                continue
            try:
                storage = row.get("storage")
                if isinstance(storage, dict) and storage.get("kind") == "restic":
                    if not self._restic_row_is_materializable(root, row):
                        continue
                    return row
                blob = self._protected_path(root, str(row.get("blob") or ""), "blobs")
                expected_size = int(row.get("size_bytes") or -1)
                key = (str(row.get("blob") or ""), row.get("sha256"), expected_size)
                valid = validation_cache.get(key)
                if valid is None:
                    valid = blob.is_file() and blob.stat().st_size == expected_size and _sha256(blob) == row.get("sha256")
                    validation_cache[key] = valid
                if not valid:
                    continue
            except HTTPException:
                continue
            return row
        return None

    def _restic_row_is_materializable(self, root: Path, row: dict[str, Any]) -> bool:
        # Base service never owns a repository. Runtime overrides this with a
        # verified materialization before permitting alias reuse.
        return False

    @staticmethod
    def _new_alias_row(
        workspace_id: str,
        name: str,
        trigger: str,
        snapshot_id: str | None,
        commit: str | None,
        source: dict[str, Any],
        token: str | None,
        capture_job_id: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "id": str(uuid4()),
            "workspace_id": workspace_id,
            "database_name": name,
            "created_at": _now().isoformat(),
            "trigger": trigger,
            "snapshot_id": snapshot_id,
            "snapshot_git_commit_hash": commit,
            "status": "ready",
            "size_bytes": int(source["size_bytes"]),
            "sha256": source["sha256"],
            "error": None,
            "blob": source["blob"],
            "fingerprint": source.get("fingerprint"),
            "schema_hash": source.get("schema_hash"),
            "source_token": token,
            "capture_job_id": capture_job_id,
        }
        if isinstance(source.get("storage"), dict):
            row["storage"] = dict(source["storage"])
            row["blob"] = None
        return row

    @staticmethod
    def _database_size_estimate(files_dir: Path, name: str) -> int:
        """Estimate main and SQLite sidecars through no-follow directory FDs."""
        total = 0
        try:
            with open_directory(files_dir, ".ragtime/db") as database_fd:
                for suffix in ("", "-wal", "-shm", "-journal"):
                    try:
                        stat_result = os.stat(name + suffix, dir_fd=database_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if not stat.S_ISREG(stat_result.st_mode):
                        raise HTTPException(status_code=409, detail="SQLite database is unsafe")
                    total += stat_result.st_size
        except SecureFileError as exc:
            raise HTTPException(status_code=409, detail="SQLite database is unsafe") from exc
        return total

    @staticmethod
    def _capture_confined(files_dir: Path, database_name: str, blob_dir: Path, destination_name: str) -> dict[str, Any]:
        """Capture through a Landlock child; never path-fallback on failure.

        SQLite can create/read WAL and SHM files itself, so a descriptor checked
        in this process is insufficient.  The child inherits pinned directory
        descriptors and is filesystem-confined before SQLite opens any name.
        """
        try:
            with open_directory(files_dir, ".ragtime/db") as source_fd, open_directory(blob_dir.parent, "blobs", create=True) as destination_fd:
                for fd in (source_fd, destination_fd):
                    os.set_inheritable(fd, True)
                completed = run_admitted_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "runtime.worker.sqlite_history.capture_child",
                        "--source-fd",
                        str(source_fd),
                        "--destination-fd",
                        str(destination_fd),
                        "--source-name",
                        database_name,
                        "--destination-name",
                        destination_name,
                    ],
                    pass_fds=(source_fd, destination_fd),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
        except (OSError, SecureFileError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=503, detail="Secure SQLite capture confinement is unavailable") from exc
        if completed.returncode:
            logger.warning("Confined SQLite capture failed returncode=%s stderr=%s", completed.returncode, completed.stderr[:400])
            raise HTTPException(status_code=503, detail="Secure SQLite capture confinement is unavailable")
        try:
            result = json.loads(completed.stdout)
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Secure SQLite capture confinement is unavailable") from exc

    @staticmethod
    def _probe_confined(files_dir: Path, database_name: str) -> str | None:
        """Ask the confined child for a conservative source token only."""
        try:
            with open_directory(files_dir, ".ragtime/db") as source_fd, open_directory(files_dir.parent, "sqlite_backups", create=True) as destination_fd:
                for fd in (source_fd, destination_fd):
                    os.set_inheritable(fd, True)
                completed = run_admitted_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "runtime.worker.sqlite_history.capture_child",
                        "--source-fd",
                        str(source_fd),
                        "--destination-fd",
                        str(destination_fd),
                        "--source-name",
                        database_name,
                        "--destination-name",
                        "blobs/.probe",
                        "--probe-source",
                    ],
                    pass_fds=(source_fd, destination_fd),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
            if completed.returncode:
                return None
            result = json.loads(completed.stdout)
            token = result.get("source_token") if isinstance(result, dict) else None
            return token if isinstance(token, str) else None
        except (OSError, SecureFileError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return None

    def _record_failure(
        self,
        workspace_id: str,
        root: Path,
        name: str,
        trigger: str,
        snapshot_id: str | None,
        commit: str | None,
        exc: Exception,
        capture_job_id: str | None = None,
    ) -> dict[str, Any]:
        with _catalog_lock(root):
            manifest = self._load(root, workspace_id)
            row = {
                "id": str(uuid4()),
                "workspace_id": workspace_id,
                "database_name": name,
                "created_at": _now().isoformat(),
                "trigger": trigger,
                "snapshot_id": snapshot_id,
                "snapshot_git_commit_hash": commit,
                "status": "failed",
                "size_bytes": 0,
                "sha256": None,
                "error": _safe_error(exc),
                "blob": None,
                "capture_job_id": capture_job_id,
            }
            manifest["backups"].append(row)
            self._save(root, manifest)
            return self._backup_response(row)

    def _enforce_quota(
        self, root: Path, manifest: dict[str, Any], incoming: int, *, protected_ids: set[str] | None = None, planned_removed: set[str] | None = None
    ) -> None:
        """Reserve space without ever leaving a saved row without its blob.

        Decide the complete eviction set before changing disk.  When eviction is
        possible, persist the reduced catalog first; an unlink failure merely
        leaves an orphan which the ordinary sweep can safely collect.
        """
        # Quota is a logical per-workspace charge, never a repository-pack
        # measurement. Restic compression/cross-workspace dedup must not reduce
        # another workspace's charge; aliases of the same image count once.
        from .catalog import HistoryCatalog

        used = HistoryCatalog.logical_charge(manifest) + self._history_disk_usage(root)
        # ``planned_removed`` has already been removed from this in-memory
        # manifest by retention. Its charge is therefore already excluded by
        # ``logical_charge``; do not substitute legacy file sizes here.
        if used + incoming > _MAX_WORKSPACE_BYTES:
            protected = self._protected_backup_ids(manifest) | (protected_ids or set())
            candidates = sorted(
                (row for row in manifest["backups"] if row.get("status") == "ready" and row["id"] not in protected),
                key=lambda row: row["created_at"],
            )
            image_references: dict[tuple[str, int], int] = {}
            for row in manifest["backups"]:
                if row.get("status") == "ready" and isinstance(row.get("sha256"), str):
                    image = (str(row["sha256"]), int(row.get("size_bytes") or 0))
                    image_references[image] = image_references.get(image, 0) + 1
            eviction_ids: set[str] = set()
            for row in candidates:
                if used + incoming <= _MAX_WORKSPACE_BYTES:
                    break
                eviction_ids.add(str(row["id"]))
                # Aliases only free logical quota after the final same-image
                # reference is evicted, irrespective of legacy/Restic storage.
                image = (str(row.get("sha256") or ""), int(row.get("size_bytes") or 0))
                image_references[image] = image_references.get(image, 1) - 1
                if image_references[image] == 0:
                    used -= image[1]
            if used + incoming > _MAX_WORKSPACE_BYTES:
                raise HTTPException(status_code=409, detail="SQLite history quota would be exceeded")
            manifest["backups"] = [row for row in manifest["backups"] if str(row["id"]) not in eviction_ids]
            self._save(root, manifest)
            self._unlink_unreferenced(root, manifest, {str(row.get("blob") or "") for row in candidates if str(row["id"]) in eviction_ids})

    @staticmethod
    def _history_disk_usage(root: Path) -> int:
        # Catalog logical charge already accounts for managed backup blobs.
        # Temporary candidates/downloads and scratch have no ready-image charge;
        # count their real bytes without charging legacy blobs a second time.
        total = managed_scratch_bytes(root)
        blobs = root / "blobs"
        if blobs.is_dir() and not blobs.is_symlink():
            total += managed_scratch_bytes(blobs)
        for name in ("candidates", "downloads"):
            total += regular_tree_bytes(root / name)
        return total

    def _prune_locked(self, root: Path, manifest: dict[str, Any]) -> set[str]:
        cutoff = _now() - timedelta(days=_RETENTION_DAYS)
        ready_by_db: dict[str, list[dict[str, Any]]] = {}
        for row in manifest["backups"]:
            if row.get("status") == "ready":
                ready_by_db.setdefault(row["database_name"], []).append(row)
        removable: set[str] = set()
        protected_backup_ids = self._protected_backup_ids(manifest)
        for rows in ready_by_db.values():
            rows.sort(key=lambda row: row["created_at"], reverse=True)
            for row in rows[_MAX_READY_PER_DATABASE:]:
                if row.get("trigger") != "pre_restore" and row["id"] not in protected_backup_ids:
                    removable.add(row["id"])
            for row in rows[1:]:
                old_pre_restore = row.get("trigger") == "pre_restore" and datetime.fromisoformat(row["created_at"]) < _now() - _PRE_RESTORE_MINIMUM
                if (
                    (row.get("trigger") != "pre_restore" or old_pre_restore)
                    and datetime.fromisoformat(row["created_at"]) < cutoff
                    and row["id"] not in protected_backup_ids
                ):
                    removable.add(row["id"])
        # Failed captures are diagnostic only; retain one month rather than
        # allowing repeated scheduled failures to grow the manifest forever.
        for row in manifest["backups"]:
            if row.get("status") == "failed" and datetime.fromisoformat(row["created_at"]) < cutoff:
                removable.add(row["id"])
        removed_rows = [row for row in manifest["backups"] if row["id"] in removable]
        kept = []
        removed_blobs: set[str] = set()
        for row in manifest["backups"]:
            if row["id"] not in removable:
                kept.append(row)
                continue
            if row.get("blob"):
                removed_blobs.add(str(row["blob"]))
        manifest["backups"] = kept
        self._queue_restic_tombstones(manifest, removed_rows)
        return removed_blobs

    @staticmethod
    def _queue_restic_tombstones(manifest: dict[str, Any], removed_rows: list[dict[str, Any]]) -> None:
        """Persist candidate IDs; maintenance later proves the global graph."""
        surviving = {
            str(storage.get("snapshot_id"))
            for row in manifest.get("backups", [])
            for storage in [row.get("storage")]
            if isinstance(storage, dict) and storage.get("kind") == "restic"
        }
        pending = manifest.setdefault("restic_forget_tombstones", [])
        for row in removed_rows:
            storage = row.get("storage")
            snapshot_id = storage.get("snapshot_id") if isinstance(storage, dict) and storage.get("kind") == "restic" else None
            if isinstance(snapshot_id, str) and snapshot_id not in surviving and snapshot_id not in pending:
                pending.append(snapshot_id)

    def _unlink_unreferenced(self, root: Path, manifest: dict[str, Any], candidates: set[str]) -> None:
        """Only unlink after the manifest durably stopped referencing a blob."""
        referenced = {str(row.get("blob")) for row in manifest.get("backups", []) if row.get("blob")}
        for relative in candidates - referenced:
            if relative:
                self._protected_path(root, relative, "blobs").unlink(missing_ok=True)

    def _materialize_row_sync(self, root: Path, row: dict[str, Any], destination: Path) -> None:
        """Materialize legacy local storage; runtime storage overrides this hook."""
        storage = row.get("storage")
        if isinstance(storage, dict) and storage.get("kind") == "restic":
            raise HTTPException(status_code=503, detail="SQLite history repository is unavailable")
        source = self._protected_path(root, str(row.get("blob") or ""), "blobs")
        shutil.copyfile(source, destination)
        if _sha256(destination) != row.get("sha256"):
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail="SQLite backup integrity verification failed")

    async def preview(
        self, workspace_id: str, backup_id: str, *, mode: str, conflict_policy: str, table_policies: dict[str, str] | None, user_id: str
    ) -> dict[str, Any]:
        # sqlite_workspace_access already calls assert_sqlite_workspace_available internally.
        async with sqlite_workspace_access(workspace_id) as files_dir:
            root = self._root(workspace_id)
            return await run_sqlite_blocking(self._preview_sync, workspace_id, root, files_dir, backup_id, mode, conflict_policy, table_policies, user_id)

    def _preview_sync(
        self, workspace_id: str, root: Path, files_dir: Path, backup_id: str, mode: str, policy: str, table_policies: dict[str, str] | None, user_id: str
    ) -> dict[str, Any]:
        with _catalog_lock(root):
            manifest = self._load(root, workspace_id)
            row = next((item for item in manifest["backups"] if item["id"] == backup_id and item.get("status") == "ready"), None)
            if not row:
                raise HTTPException(status_code=404, detail="SQLite backup not found")
            temporary_backup: Path | None = None
            if isinstance(row.get("storage"), dict) and row["storage"].get("kind") == "restic":
                blob_dir = _history_subdirectory(root, "blobs", create=True)
                temporary_backup = blob_dir / f".preview-{uuid4()}.sqlite3"
                self._materialize_row_sync(root, row, temporary_backup)
                backup = temporary_backup
                backup_relative = f"blobs/{backup.name}"
            else:
                backup = self._protected_path(root, row["blob"], "blobs")
                backup_relative = str(row["blob"])
            if not backup.is_file() or _sha256(backup) != row["sha256"]:
                raise HTTPException(status_code=409, detail="SQLite backup integrity verification failed")
            migration_dir = files_dir / ".ragtime" / "db" / "migrations"
            # Preview preparation can hold a source online-backup copy and emit
            # a full candidate. Reserve both conservative inputs before the
            # child starts, then validate actual disk usage before publication.
            reservation = backup.stat().st_size + self._database_size_estimate(files_dir, row["database_name"])
            self._enforce_quota(root, manifest, reservation, protected_ids={backup_id})
            candidate_dir = _history_subdirectory(root, "candidates", create=True)
            candidate = candidate_dir / f"{uuid4()}.sqlite3"
            # Pin one online-backup copy while access is held.  Both candidate
            # preparation and the preview fingerprint use precisely this copy,
            # never a later mutable live file.
            try:
                result = self._preview_confined(
                    files_dir,
                    root,
                    backup_relative,
                    row["database_name"],
                    f"candidates/{candidate.name}",
                    mode,
                    policy,
                    table_policies or {},
                )
                self._enforce_quota(root, manifest, 0, protected_ids={backup_id})
            except Exception:
                self._protected_path(root, f"candidates/{candidate.name}", "candidates").unlink(missing_ok=True)
                raise
            finally:
                if temporary_backup is not None:
                    temporary_backup.unlink(missing_ok=True)
            current_fingerprint = result.pop("current_fingerprint", None)
            migration_value = result.pop("migration_fingerprint", None)
            preview_id = None
            expires_at = None
            if result.get("can_apply"):
                preview_id = str(uuid4())
                expires_at = (_now() + _PREVIEW_TTL).isoformat()
                manifest["previews"][preview_id] = {
                    "backup_id": backup_id,
                    "database_name": row["database_name"],
                    "candidate": f"candidates/{candidate.name}",
                    "candidate_sha256": result["candidate_sha256"],
                    "current_fingerprint": current_fingerprint,
                    "migration_fingerprint": migration_value,
                    "mode": mode,
                    "conflict_policy": policy,
                    "table_policies": table_policies or {},
                    "user_id": user_id,
                    "expires_at": expires_at,
                }
                self._save(root, manifest)
            else:
                self._protected_path(root, f"candidates/{candidate.name}", "candidates").unlink(missing_ok=True)
            return {
                "preview_id": preview_id,
                "backup_id": backup_id,
                "database_name": row["database_name"],
                "mode": mode,
                "conflict_policy": policy,
                "tables": result.get("tables", []),
                "migrations_applied": result.get("migrations_applied", []),
                "warnings": result.get("warnings", []),
                "blockers": result.get("blockers", []),
                "can_apply": bool(result.get("can_apply")),
                "expires_at": expires_at,
            }

    @staticmethod
    def _preview_confined(
        files_dir: Path, root: Path, backup: str, database_name: str, candidate: str, mode: str, policy: str, table_policies: dict[str, str]
    ) -> dict[str, Any]:
        try:
            with open_directory(files_dir, ".ragtime/db") as source_fd, open_directory(root.parent, root.name) as destination_fd:
                for fd in (source_fd, destination_fd):
                    os.set_inheritable(fd, True)
                completed = run_admitted_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "runtime.worker.sqlite_history.capture_child",
                        "--source-fd",
                        str(source_fd),
                        "--destination-fd",
                        str(destination_fd),
                        "--source-name",
                        database_name,
                        "--destination-name",
                        candidate,
                        "--preview-backup",
                        backup,
                        "--preview-current",
                        database_name,
                        "--preview-migrations",
                        "migrations",
                        "--preview-candidate",
                        candidate,
                        "--mode",
                        mode,
                        "--conflict-policy",
                        policy,
                        "--table-policies",
                        json.dumps(table_policies),
                    ],
                    pass_fds=(source_fd, destination_fd),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
        except (OSError, SecureFileError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=503, detail="Secure SQLite restore confinement is unavailable") from exc
        if completed.returncode:
            logger.warning("Confined SQLite preview failed returncode=%s stderr=%s", completed.returncode, completed.stderr[:400])
            raise HTTPException(status_code=503, detail="Secure SQLite restore confinement is unavailable")
        try:
            result = json.loads(completed.stdout)
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Secure SQLite restore confinement is unavailable") from exc

    @staticmethod
    def _drift_confined(files_dir: Path, root: Path, database_name: str) -> dict[str, Any]:
        """Read live SQLite/migration fingerprints only in the Landlock child."""
        try:
            with open_directory(files_dir, ".ragtime/db") as source_fd, open_directory(root.parent, root.name) as destination_fd:
                completed = run_admitted_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "runtime.worker.sqlite_history.capture_child",
                        "--source-fd",
                        str(source_fd),
                        "--destination-fd",
                        str(destination_fd),
                        "--source-name",
                        database_name,
                        "--destination-name",
                        "candidates/.drift",
                        "--fingerprint-current",
                        database_name,
                        "--fingerprint-migrations",
                        "migrations",
                    ],
                    pass_fds=(source_fd, destination_fd),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
        except (OSError, SecureFileError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=503, detail="Secure SQLite drift confinement is unavailable") from exc
        if completed.returncode:
            raise HTTPException(status_code=503, detail="Secure SQLite drift confinement is unavailable")
        try:
            result = json.loads(completed.stdout)
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Secure SQLite drift confinement is unavailable") from exc

    def _completed_receipt(self, root: Path, workspace_id: str, preview_id: str, user_id: str) -> dict[str, Any] | None:
        """Read an owned terminal receipt without draining the runtime."""
        with _catalog_lock(root):
            manifest = self._load(root, workspace_id)
            operation = next(
                (
                    row
                    for row in manifest["operations"].values()
                    if row.get("preview_id") == preview_id and row.get("status") == "completed" and row.get("user_id") == user_id
                ),
                None,
            )
            if operation is None:
                return None
            result = operation.get("result")
            return result if isinstance(result, dict) else None

    @staticmethod
    def _publish_verified_candidate(root: Path, candidate: str, candidate_sha256: str, files_dir: Path, database_name: str, *, verification_error: str) -> None:
        """Publish one verified candidate and remove stale SQLite sidecars."""
        try:
            publish_regular_file(root, candidate, files_dir, f".ragtime/db/{database_name}")
            for suffix in ("-wal", "-shm", "-journal"):
                delete_file(files_dir, f".ragtime/db/{database_name}{suffix}")
        except SecureFileError as exc:
            raise HTTPException(status_code=409, detail="SQLite publication target is unsafe") from exc
        if sha256_regular_file(files_dir, f".ragtime/db/{database_name}") != candidate_sha256:
            raise RuntimeError(verification_error)

    async def apply(self, workspace_id: str, preview_id: str, *, user_id: str) -> dict[str, Any]:
        root = self._root(workspace_id)
        receipt = await run_sqlite_blocking(self._completed_receipt, root, workspace_id, preview_id, user_id)
        if receipt is not None:
            return receipt
        # sqlite_workspace_access already calls assert_sqlite_workspace_available internally.
        pre_intent_error: BaseException | None = None
        async with sqlite_workspace_access(workspace_id, maintenance=True) as files_dir:
            marker_payload = await run_sqlite_blocking(read_marker, root / "sqlite-maintenance-intent.json")
            lease_id = str(marker_payload.get("lease_id") or "") if marker_payload else ""
            if not lease_id:
                raise HTTPException(status_code=423, detail="SQLite maintenance fence cannot be verified")
            try:
                return await run_sqlite_blocking(self._apply_sync, workspace_id, root, files_dir, preview_id, user_id, lease_id)
            except _PreIntentFailure as exc:
                # These checks happen before any restore intent or publication
                # exists, so the live database was never touched.  Capture the
                # original error INSIDE the context and let maintenance exit
                # normally (releasing the fence/marker via the else branch);
                # mirror _guarded_code_restore's body_error pattern.  Publication
                # failures (raised after intent) are NOT _PreIntentFailure and
                # still cross the boundary fail-closed, retaining the marker.
                pre_intent_error = exc.error
        if pre_intent_error is not None:
            raise pre_intent_error

    def _apply_sync(self, workspace_id: str, root: Path, files_dir: Path, preview_id: str, user_id: str, lease_id: str) -> dict[str, Any]:
        with _catalog_lock(root):
            try:
                manifest = self._load(root, workspace_id)
                preview = manifest["previews"].get(preview_id)
                # Successful receipts outlive preview cleanup, but only the
                # authenticated creator may retrieve them.  Legacy receipts
                # without ownership are intentionally not inferred.
                operation = next(
                    (
                        row
                        for row in manifest["operations"].values()
                        if row.get("preview_id") == preview_id and row.get("status") == "completed" and row.get("user_id") == user_id
                    ),
                    None,
                )
                if operation:
                    return operation["result"]
                if not preview or preview["user_id"] != user_id:
                    raise HTTPException(status_code=409, detail="SQLite restore preview must be regenerated")
                if datetime.fromisoformat(preview["expires_at"]) <= _now():
                    raise HTTPException(status_code=409, detail="SQLite restore preview expired; regenerate it")
                drift = self._drift_confined(files_dir, root, preview["database_name"])
                if drift["current_fingerprint"] != preview["current_fingerprint"] or drift["migration_fingerprint"] != preview["migration_fingerprint"]:
                    raise HTTPException(status_code=409, detail="SQLite database changed; regenerate restore preview")
                candidate = self._protected_path(root, preview["candidate"], "candidates")
                if not candidate.is_file() or _sha256(candidate) != preview["candidate_sha256"]:
                    raise HTTPException(status_code=409, detail="SQLite restore candidate verification failed")
                # A missing live target has no state to preserve.  Merge has
                # already been rejected by the recovery engine in this case.
                safety = (
                    self._capture_one_locked(workspace_id, root, files_dir, preview["database_name"], "pre_restore", None, None)
                    if drift["current_fingerprint"] is not None
                    else None
                )
            except Exception as exc:
                raise _PreIntentFailure(exc) from exc
            # The mandatory capture atomically published its own catalog update.
            # Reload before recording the restore intent so it cannot be lost.
            manifest = self._load(root, workspace_id)
            op_id = str(uuid4())
            manifest["operations"][op_id] = {
                "preview_id": preview_id,
                "user_id": user_id,
                "status": "intent",
                "candidate": preview["candidate"],
                "safety_backup_id": safety["id"] if safety else None,
                "publication_state": "prepublication",
                "lease_id": lease_id,
                "created_at": _now().isoformat(),
            }
            self._save(root, manifest)
            # Durably cross to 'publishing' BEFORE the live database is replaced.
            # A crash after os.replace but before the terminal save then leaves
            # 'publishing' (not 'prepublication'), so abort is refused and only
            # completion (which re-publishes the idempotent candidate and clears
            # stale sidecars) can finish the interrupted restore.
            manifest["operations"][op_id]["publication_state"] = "publishing"
            self._save(root, manifest)
            self._publish_verified_candidate(
                root,
                str(preview["candidate"]),
                str(preview["candidate_sha256"]),
                files_dir,
                str(preview["database_name"]),
                verification_error="SQLite publication integrity verification failed",
            )
            manifest["operations"][op_id]["publication_state"] = "published"
            self._save(root, manifest)
            result = {
                "operation_id": op_id,
                "restored_backup_id": preview["backup_id"],
                "safety_backup_id": safety["id"] if safety else None,
                "runtime_stopped": True,
                "status": "completed",
            }
            manifest["operations"][op_id].update(status="completed", result=result, completed_at=_now().isoformat())
            self._save(root, manifest)
            return result

    async def download_path(self, workspace_id: str, backup_id: str) -> Path:
        root = self._root(workspace_id)

        def lookup() -> Path:
            with _catalog_lock(root):
                row = next((item for item in self._load(root, workspace_id)["backups"] if item["id"] == backup_id and item.get("status") == "ready"), None)
                if not row:
                    raise HTTPException(status_code=404, detail="SQLite backup not found")
                storage = row.get("storage")
                if not (isinstance(storage, dict) and storage.get("kind") == "restic"):
                    path = self._protected_path(root, row["blob"], "blobs")
                    if not path.is_file() or _sha256(path) != row["sha256"]:
                        raise HTTPException(status_code=409, detail="SQLite backup integrity verification failed")
                self._enforce_quota(root, self._load(root, workspace_id), int(row["size_bytes"]), protected_ids={backup_id})
                download_dir = _history_subdirectory(root, "downloads", create=True)
                copy = download_dir / f"{uuid4()}.sqlite3"
                try:
                    self._materialize_row_sync(root, row, copy)
                    with copy.open("rb") as handle:
                        os.fsync(handle.fileno())
                    self._enforce_quota(root, self._load(root, workspace_id), 0, protected_ids={backup_id})
                    if _sha256(copy) != row["sha256"]:
                        raise HTTPException(status_code=409, detail="SQLite backup download verification failed")
                except Exception:
                    self._protected_path(root, f"downloads/{copy.name}", "downloads").unlink(missing_ok=True)
                    raise
                return copy

        return await run_sqlite_blocking(lookup)

    async def download_to_path(self, workspace_id: str, backup_id: str, destination: Path) -> Path:
        """Materialize a verified image into a caller-owned private staging file."""
        root = self._root(validate_workspace_id(workspace_id))

        def materialize() -> Path:
            with _catalog_lock(root):
                row = next((item for item in self._load(root, workspace_id)["backups"] if item.get("id") == backup_id and item.get("status") == "ready"), None)
                if row is None:
                    raise HTTPException(status_code=404, detail="SQLite backup not found")
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._materialize_row_sync(root, row, destination)
                if destination.stat().st_size != int(row["size_bytes"]) or _sha256(destination) != row["sha256"]:
                    destination.unlink(missing_ok=True)
                    raise HTTPException(status_code=409, detail="SQLite backup download verification failed")
                return destination

        return await run_sqlite_blocking(materialize)

    async def delete(self, workspace_id: str, backup_id: str) -> None:
        root = self._root(workspace_id)

        def remove() -> None:
            with _catalog_lock(root):
                manifest = self._load(root, workspace_id)
                row = next((item for item in manifest["backups"] if item["id"] == backup_id), None)
                if not row:
                    raise HTTPException(status_code=404, detail="SQLite backup not found")
                if backup_id in self._protected_backup_ids(manifest):
                    raise HTTPException(status_code=409, detail="SQLite backup is protected and cannot be deleted")
                manifest["backups"].remove(row)
                self._queue_restic_tombstones(manifest, [row])
                self._save(root, manifest)
                if row.get("blob"):
                    self._unlink_unreferenced(root, manifest, {str(row["blob"])})

        await run_sqlite_blocking(remove)

    async def recover_operation(self, workspace_id: str, operation_id: str, *, action: Literal["complete", "abort"]) -> dict[str, Any]:
        """Finish or abandon a receipt left by an interrupted maintenance apply.

        The catalog receipt is made durably terminal BEFORE the runtime marker
        is released.  If marker release then fails (or the process crashes after
        it), a later retry with the same action finds the operation already
        terminal, skips republication, and safely re-attempts the idempotent
        release.  The opposite action against a terminal operation is rejected.
        This intentionally does not infer completion from elapsed time.
        """
        root = self._root(workspace_id)
        marker = root / "sqlite-maintenance-intent.json"

        # Read marker BEFORE acquiring catalog lock to avoid lock inversion.
        # Use hardened reader that rejects symlinks, directories, and malformed
        # payloads.  A missing marker is not fatal on its own: a prior attempt
        # may have already released it after persisting a terminal receipt, so
        # the catalog state below decides whether a retry can safely finish.
        def read_marker_safely() -> dict[str, object] | None:
            try:
                return read_marker(marker)
            except HTTPException:
                raise HTTPException(status_code=409, detail="SQLite maintenance fence cannot be verified")

        marker_payload = await run_sqlite_blocking(read_marker_safely)
        marker_lease_id = str(marker_payload.get("lease_id") or "") if marker_payload else ""
        if marker_payload is not None and not marker_lease_id:
            raise HTTPException(status_code=409, detail="SQLite maintenance fence cannot be verified")

        def _terminal_result(operation: dict[str, Any]) -> dict[str, Any]:
            stored = operation.get("result")
            if isinstance(stored, dict):
                return stored
            return {
                "operation_id": operation_id,
                "restored_backup_id": None,
                "safety_backup_id": operation.get("safety_backup_id"),
                "runtime_stopped": True,
                "status": operation.get("status"),
            }

        # Receipt replay does not need (and must not acquire) recovery
        # ownership if its original marker is already gone or belongs to a new
        # holder.  This is a read-only catalog lookup before any maintenance.
        def terminal_replay() -> dict[str, Any] | None:
            with _catalog_lock(root):
                operation = self._load(root, workspace_id)["operations"].get(operation_id)
                if operation is None or operation.get("status") not in {"completed", "aborted"}:
                    return None
                if (action == "abort") != (operation.get("status") == "aborted"):
                    raise HTTPException(status_code=409, detail="SQLite maintenance operation already finished with the opposite action")
                if marker_payload is None or operation.get("lease_id") != marker_lease_id:
                    return _terminal_result(operation)
                return None

        replay = await run_sqlite_blocking(terminal_replay)
        if replay is not None:
            return replay
        if marker_payload is None or not marker_lease_id:
            raise HTTPException(status_code=409, detail="SQLite maintenance fence cannot be verified")

        def recover_receipt() -> tuple[str, dict[str, Any]]:
            with _catalog_lock(root):
                manifest = self._load(root, workspace_id)
                operation = manifest["operations"].get(operation_id)

                # Retry safety: an operation already driven to a terminal state
                # by a previous attempt must not be republished.  The matching
                # action returns its stored receipt so the caller can re-run the
                # idempotent marker release; the opposite action is rejected.
                if operation is not None and operation.get("status") in {"completed", "aborted"}:
                    terminal = operation.get("status")
                    if (action == "abort") != (terminal == "aborted"):
                        raise HTTPException(status_code=409, detail="SQLite maintenance operation already finished with the opposite action")
                    # Release only the fence this terminal operation itself held.
                    # If the current marker belongs to a different lease (a new
                    # maintenance holder acquired the fence after this operation
                    # finished), this operation's fence is already gone: return an
                    # empty lease so the post-receipt release is skipped and the
                    # foreign marker is never released on its behalf.
                    own_lease = str(operation.get("lease_id") or "")
                    release_lease = own_lease if own_lease and own_lease == marker_lease_id else ""
                    return release_lease, _terminal_result(operation)

                # A missing marker with no terminal receipt cannot be verified.
                if marker_payload is None:
                    raise HTTPException(status_code=409, detail="SQLite maintenance fence cannot be verified")
                lease_id = marker_lease_id

                # A crash between acquiring the fence and recording an intent
                # has no candidate to publish.  Only the marker lease itself
                # can be safely aborted.
                if not operation:
                    if action != "abort" or operation_id != lease_id:
                        raise HTTPException(status_code=409, detail="SQLite maintenance operation is not recoverable")
                    return lease_id, {"operation_id": operation_id, "status": "aborted"}
                if operation.get("status") != "intent":
                    raise HTTPException(status_code=409, detail="SQLite maintenance operation is not recoverable")
                if action == "abort":
                    if operation.get("publication_state") != "prepublication":
                        raise HTTPException(status_code=409, detail="Published SQLite restore must be completed, not aborted")
                    # Persist the terminal abort receipt BEFORE the marker is
                    # released so a crash mid-release cannot orphan the intent.
                    abort_result: dict[str, Any] = {"operation_id": operation_id, "status": "aborted"}
                    operation.update(status="aborted", aborted_at=_now().isoformat(), result=abort_result, lease_id=lease_id)
                    self._save(root, manifest)
                    return lease_id, abort_result
                candidate = self._protected_path(root, str(operation.get("candidate") or ""), "candidates")
                preview = manifest["previews"].get(operation.get("preview_id"))
                if not preview or not candidate.is_file() or _sha256(candidate) != preview.get("candidate_sha256"):
                    raise HTTPException(status_code=409, detail="SQLite recovery candidate cannot be verified")
                # Mirror _apply_sync: durably cross to 'publishing' BEFORE the
                # live database is replaced, so an interrupted recovery-complete
                # can only be finished by another complete, never aborted.
                operation["publication_state"] = "publishing"
                self._save(root, manifest)
                self._publish_verified_candidate(
                    root,
                    str(operation["candidate"]),
                    str(preview["candidate_sha256"]),
                    self._files_dir_for_workspace(workspace_id),
                    str(preview["database_name"]),
                    verification_error="SQLite recovery publication verification failed",
                )
                completed_result: dict[str, Any] = {
                    "operation_id": operation_id,
                    "restored_backup_id": preview["backup_id"],
                    "safety_backup_id": operation.get("safety_backup_id"),
                    "runtime_stopped": True,
                    "status": "completed",
                }
                # Persist the terminal completed receipt (with published state)
                # BEFORE releasing the marker.  The database is already durably
                # published above, so this makes the catalog receipt terminal in
                # the same locked save, closing the crash window that previously
                # left an orphaned intent when release/finalize was interrupted.
                operation.update(
                    status="completed",
                    publication_state="published",
                    result=completed_result,
                    completed_at=_now().isoformat(),
                    lease_id=lease_id,
                )
                self._save(root, manifest)
                return lease_id, completed_result

        # The pre-read marker only supplies a lease candidate.  The recovery
        # context takes the exclusive non-blocking operation flock and verifies
        # that marker again before *any* receipt mutation/publication.
        async with sqlite_workspace_recovery(workspace_id, marker_lease_id):
            lease_id, result = await run_sqlite_blocking(recover_receipt)
            if lease_id:
                await recover_sqlite_workspace_maintenance(workspace_id, lease_id, action=action)
            return result


class RuntimeSqliteHistoryService(SqliteHistoryService):
    """Runtime owner preserving every legacy service method and manifest field.

    ``root`` is the runtime User Space root containing ``workspaces``.  The
    worker determines canonical versus active chroot files through its existing
    lease API; this class never guesses a source tree.
    """

    def __init__(self, root: Path, worker: Any, repository: Any | None = None) -> None:
        self._runtime = _RuntimeHistoryContext(Path(root), worker)
        self.repository = repository
        # A guarded restore deliberately owns its contexts in one task.  The
        # begin and finish HTTP requests are different tasks, so keeping an
        # asynccontextmanager here and exiting it from finish would reset a
        # ContextVar token in the wrong context (and strand its flocks).
        self._guarded_restores: dict[str, dict[str, Any]] = {}
        self._runtime_maintenance: Any | None = None
        super().__init__(lambda workspace_id: self._runtime.root / "workspaces" / workspace_id / "files")

    def _store_captured_image(self, workspace_id: str, operation_id: str, destination: Path, row: dict[str, Any]) -> dict[str, Any] | None:
        repository = self.repository
        if repository is None:
            return None
        from .storage import repository_gate

        async def ingest() -> ResticArtifact:
            return await repository.ingest(
                destination,
                workspace_id=workspace_id,
                operation_id=operation_id,
                sha256=str(row["sha256"]),
                size_bytes=int(row["size_bytes"]),
                pass_fds=_capture_pass_fds.get(),
            )

        # This function runs inside ``run_sqlite_blocking``.  It has no running
        # loop, so the narrow bridge cannot cross an application-loop lock.
        with repository_gate(self._runtime.root, exclusive=False):
            artifact = asyncio.run(ingest())
            # Restic's successful backup/ls metadata is not sufficient for a
            # destructive safety point. Verify the committed bytes before the
            # manifest can name the snapshot.
            verify_dir = destination.parent
            verify = verify_dir / f".ingest-verify-{uuid4()}.sqlite3"
            try:
                asyncio.run(repository.materialize(artifact, verify, pass_fds=_capture_pass_fds.get()))
            finally:
                verify.unlink(missing_ok=True)
        return {
            "kind": "restic",
            "repository_id": artifact.repository_id,
            "snapshot_id": artifact.snapshot_id,
            "path": artifact.path,
        }

    def _restic_row_is_materializable(self, root: Path, row: dict[str, Any]) -> bool:
        if self.repository is None:
            return False
        from .storage import repository_gate, restic_artifact

        storage = row.get("storage")
        if not isinstance(storage, dict):
            return False
        try:
            artifact = restic_artifact(storage, size_bytes=int(row["size_bytes"]), sha256=str(row["sha256"]))
            with repository_gate(self._runtime.root, exclusive=False):
                asyncio.run(self.repository.verify(artifact, pass_fds=_capture_pass_fds.get()))
            return True
        except Exception:
            return False

    def _materialize_row_sync(self, root: Path, row: dict[str, Any], destination: Path) -> None:
        storage = row.get("storage")
        if not isinstance(storage, dict) or storage.get("kind") != "restic":
            source = self._protected_path(root, str(row.get("blob") or ""), "blobs")
            shutil.copyfile(source, destination)
            if _sha256(destination) != row.get("sha256"):
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=409, detail="SQLite backup integrity verification failed")
            return
        if self.repository is None:
            raise HTTPException(status_code=503, detail="SQLite history repository is unavailable")
        from .storage import repository_gate, restic_artifact

        artifact = restic_artifact(storage, size_bytes=int(row["size_bytes"]), sha256=str(row["sha256"]))
        with repository_gate(self._runtime.root, exclusive=False):
            asyncio.run(self.repository.materialize(artifact, destination, pass_fds=_capture_pass_fds.get()))

    @staticmethod
    def _require_legacy_directory(directory: Path) -> None:
        if not directory.exists():
            directory.mkdir(mode=0o700)
        _require_regular_directory(directory)

    @staticmethod
    def _legacy_sha256(path: Path) -> str:
        return _sha256(path)

    @staticmethod
    @contextmanager
    def _catalog_lock_for_conversion(root: Path) -> Iterator[None]:
        with _catalog_lock(root):
            yield

    async def migrate_legacy_backups(
        self,
        workspace_id: str,
        *,
        pass_fds: tuple[int, ...] = (),
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> int:
        """Convert legacy rows, pausing only between whole legacy blobs.

        ``cancel_check`` is optional for the bootstrap runner; a true result
        leaves the durable per-blob conversion journal ready for a later resume.
        """
        workspace_id = validate_workspace_id(workspace_id)
        if self.repository is None:
            raise HTTPException(status_code=503, detail="SQLite history repository is unavailable")
        token = _capture_pass_fds.set(tuple(dict.fromkeys((*_capture_pass_fds.get(), *pass_fds))))
        try:
            # One-time migration owns the exclusive history gate so normal
            # cleanup/delete cannot race a prepared legacy blob. DB writers are
            # not fenced by this repository/catalog maintenance barrier.
            from .storage import AsyncRepositoryGate

            async with AsyncRepositoryGate(self._runtime.root, exclusive=True):
                async with self._installed():
                    from .conversion import LegacyHistoryConverter

                    return await LegacyHistoryConverter(self, workspace_id, pass_fds=_capture_pass_fds.get()).migrate(cancel_check)
        finally:
            _capture_pass_fds.reset(token)

    async def claim_due(self, workspace_ids: list[str]) -> list[dict[str, str]]:
        """Return durable occurrence IDs without dispatching a capture."""
        async with self._installed():
            claims: list[dict[str, str]] = []
            for workspace_id in workspace_ids[:100]:
                workspace_id = validate_workspace_id(workspace_id)
                root = self._root(workspace_id)
                claim = await run_sqlite_blocking(self._claim_scheduled_due_sync, root, workspace_id)
                if claim is not None:
                    claims.append({"workspace_id": workspace_id, "occurrence_id": claim})
            return claims

    async def ack_due(self, workspace_id: str, occurrence_id: str, job_id: str) -> bool:
        if not isinstance(job_id, str) or not job_id:
            raise HTTPException(status_code=400, detail="scheduled job ID is required")
        workspace_id = validate_workspace_id(workspace_id)
        async with self._installed():
            return await run_sqlite_blocking(self._complete_scheduled_attempt_sync, self._root(workspace_id), workspace_id, occurrence_id, True)

    async def forget_unreferenced_snapshots(self, snapshot_ids: list[str]) -> list[str]:
        """Forget only explicit IDs proven absent from catalog/receipt graph."""
        if self.repository is None:
            return []
        from .operations import OperationStore
        from .storage import AsyncRepositoryGate, eligible_snapshot_ids, repository_gate

        # Forget mutates repository reachability.  Take the exclusive gate before
        # the normal service-wide shared gate so it never attempts an upgrade.
        async with AsyncRepositoryGate(self._runtime.root, exclusive=True):
            async with self._installed():

                def eligible() -> list[str]:
                    receipts = OperationStore(self._runtime.root / "_sqlite_history").list_workspace(None)
                    return eligible_snapshot_ids(self._runtime.root, snapshot_ids, receipt_refs=receipts)

                identifiers = await run_sqlite_blocking(eligible)
                if not identifiers:
                    return []
                with repository_gate(self._runtime.root, exclusive=True):
                    await self.repository.forget(identifiers, pass_fds=_capture_pass_fds.get())
                return identifiers

    async def drain_restic_forget_tombstones(self) -> list[str]:
        """Forget persisted explicit candidates, then durably clear tombstones."""
        candidates: list[str] = []
        workspaces = self._runtime.root / "workspaces"
        if workspaces.is_dir():
            for entry in workspaces.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    workspace_id = entry.name
                    root = self._root(workspace_id)

                    def read(workspace_id: str = workspace_id, root: Path = root) -> list[str]:
                        with _catalog_lock(root):
                            return list(self._load(root, workspace_id).get("restic_forget_tombstones", []))

                    candidates.extend(await run_sqlite_blocking(read))
        forgotten = await self.forget_unreferenced_snapshots(candidates)
        if not forgotten:
            return []
        for entry in workspaces.iterdir() if workspaces.is_dir() else ():
            if not entry.is_dir() or entry.is_symlink():
                continue
            workspace_id = entry.name
            root = self._root(workspace_id)

            def clear(workspace_id: str = workspace_id, root: Path = root) -> None:
                with _catalog_lock(root):
                    manifest = self._load(root, workspace_id)
                    manifest["restic_forget_tombstones"] = [item for item in manifest.get("restic_forget_tombstones", []) if item not in forgotten]
                    self._save(root, manifest)

            await run_sqlite_blocking(clear)
        return forgotten

    async def reconcile_capture_operation(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Classify a restart-stale capture without inventing a ready backup.

        Restic tags are only an orphan-discovery aid: a snapshot cannot become a
        logical success unless the receipt already carries the durable image
        identity that was verified before catalog publication.  This deliberately
        turns unknown/duplicate tag results into ``interrupted`` rather than
        rerunning or adopting bytes whose source cannot be proven.
        """
        if receipt.get("kind") != "capture" or self.repository is None:
            return {"phase": "interrupted", "error": "Runtime capture recovery requires a durable repository receipt"}
        workspace_id = str(receipt.get("workspace_id") or "")
        operation_id = str(receipt.get("operation_id") or "")
        try:
            validate_workspace_id(workspace_id)
            from .operations import OperationStore
            from .storage import AsyncRepositoryGate

            payload = receipt.get("accepted_payload")
            names = payload.get("database_names") if isinstance(payload, dict) else None
            operation_ids = [OperationStore.suboperation_id(operation_id, str(name)) for name in names] if isinstance(names, list) and names else [operation_id]
            async with AsyncRepositoryGate(self._runtime.root, exclusive=False):
                discovered = {
                    candidate: await self.repository.list_operation_snapshot_ids(workspace_id=workspace_id, operation_id=candidate)
                    for candidate in operation_ids
                }
        except Exception:
            return {"phase": "interrupted", "error": "Runtime capture repository recovery is unavailable"}
        if any(len(snapshot_ids) != 1 for snapshot_ids in discovered.values()):
            return {"phase": "interrupted", "error": "Runtime capture repository result is absent or ambiguous"}

        root = self._root(workspace_id)

        def verified_catalog_boundary() -> bool:
            with _catalog_lock(root):
                manifest = self._load(root, workspace_id)
                rows_by_snapshot = {
                    str(storage.get("snapshot_id")): row
                    for row in manifest.get("backups", [])
                    if isinstance(row, dict)
                    for storage in [row.get("storage")]
                    if isinstance(storage, dict) and storage.get("kind") == "restic" and isinstance(storage.get("snapshot_id"), str)
                }
                snapshot_ids = {values[0] for values in discovered.values()}
                if len(snapshot_ids) != len(discovered) or not snapshot_ids.issubset(rows_by_snapshot):
                    return False
                for snapshot_id in snapshot_ids:
                    row = rows_by_snapshot[snapshot_id]
                    probe = _history_subdirectory(root, "downloads", create=True) / f".reconcile-{uuid4()}.sqlite3"
                    try:
                        self._materialize_row_sync(root, row, probe)
                    finally:
                        probe.unlink(missing_ok=True)
                return True

        try:
            verified = await run_sqlite_blocking(verified_catalog_boundary)
        except Exception:
            verified = False
        if not verified:
            return {"phase": "interrupted", "error": "Runtime capture repository result lacks a verified durable catalog identity"}
        # The coordinator can transition reconciling directly to completed only
        # when this exact catalog boundary has already been materialized and
        # verified.  Do not pass mutable outcome fields to that terminal phase.
        return {"phase": "completed"}

    async def begin_guarded_code_restore(self, workspace_id: str, operation_id: str, user_id: str) -> dict[str, Any]:
        """Acquire one exclusive lease and preserve verified DB images for Git.

        Trusted Git callers attach their own substeps to ``operation_id``; this
        method deliberately never releases the lease on an ambiguous failure.
        """
        workspace_id = validate_workspace_id(workspace_id)
        if not operation_id or not user_id:
            raise HTTPException(status_code=400, detail="guarded restore identity is required")
        held = self._guarded_restores.get(operation_id)
        if held is not None:
            if held["workspace_id"] != workspace_id or held["user_id"] != user_id:
                raise HTTPException(status_code=409, detail="guarded SQLite restore lease is unavailable")
            return await asyncio.shield(held["begun"])

        loop = asyncio.get_running_loop()
        held = {
            "workspace_id": workspace_id,
            "user_id": user_id,
            "begun": loop.create_future(),
            "finish": loop.create_future(),
        }
        self._guarded_restores[operation_id] = held
        task = asyncio.create_task(self._guarded_restore_lifecycle(operation_id, held), name=f"sqlite-guarded-restore-{operation_id}")
        held["task"] = task
        task.add_done_callback(lambda completed: self._guarded_restore_done(operation_id, held, completed))
        return await asyncio.shield(held["begun"])

    def _guarded_restore_done(self, operation_id: str, held: dict[str, Any], task: asyncio.Task[Any]) -> None:
        """Retain failed guards for explicit durable recovery; discard successes."""
        if task.cancelled():
            logger.error("SQLite guarded restore lifecycle cancelled operation_id=%s; durable recovery is required", operation_id)
            return
        error = task.exception()
        if error is not None:
            logger.error("SQLite guarded restore lifecycle failed operation_id=%s error_type=%s", operation_id, type(error).__name__)
            return
        if self._guarded_restores.get(operation_id) is held:
            self._guarded_restores.pop(operation_id, None)

    async def _guarded_restore_lifecycle(self, operation_id: str, held: dict[str, Any]) -> None:
        """Own the global gate, worker lease, and ContextVar tokens until finish."""
        workspace_id = str(held["workspace_id"])
        user_id = str(held["user_id"])
        begun: asyncio.Future[dict[str, Any]] = held["begun"]
        finish: asyncio.Future[tuple[str, str | None, asyncio.Future[dict[str, Any]]]] = held["finish"]
        completed: asyncio.Future[dict[str, Any]] | None = None
        completed_result: dict[str, Any] | None = None
        try:
            # This shared gate stays held while the caller's Git children run,
            # preventing an exporter's exclusive gate from cutting through the
            # preservation/republish window.
            async with self._installed():
                async with sqlite_workspace_access(workspace_id, maintenance=True) as files:
                    preserved = await self._capture_workspace_databases(
                        workspace_id,
                        trigger="pre_restore",
                        mandatory=True,
                        files_dir=files,
                        capture_job_id=f"guarded-{operation_id}",
                    )
                    root = self._root(workspace_id)
                    marker = await run_sqlite_blocking(read_marker, root / MARKER_NAME)
                    lease_id = str(marker.get("lease_id") or "") if marker else ""
                    if not lease_id:
                        raise HTTPException(status_code=423, detail="guarded SQLite restore lease cannot be verified")

                    def persist() -> None:
                        with _catalog_lock(root):
                            manifest = self._load(root, workspace_id)
                            manifest["operations"][operation_id] = {
                                "kind": "guarded_code_restore",
                                "status": "active",
                                "lease_id": lease_id,
                                "user_id": user_id,
                                "preserved": preserved,
                                "created_at": _now().isoformat(),
                            }
                            self._save(root, manifest)

                    await run_sqlite_blocking(persist)
                    response = {
                        "operation_id": operation_id,
                        "status": "active",
                        "lease_id": lease_id,
                        "authoritative_root": str(files),
                        "preserved": preserved,
                    }
                    held.update(lease_id=lease_id, files=files)
                    begun.set_result(response)
                    finishing_user, git_error, finished = await finish

                    def republish() -> None:
                        with _catalog_lock(root):
                            manifest = self._load(root, workspace_id)
                            operation = manifest["operations"].get(operation_id)
                            if not operation or operation.get("lease_id") != lease_id:
                                raise HTTPException(status_code=423, detail="guarded SQLite restore lease ownership was lost")
                            if operation.get("user_id") != finishing_user:
                                raise HTTPException(status_code=403, detail="guarded SQLite restore creator does not match")
                            for item in operation.get("preserved", []):
                                backup_id = item.get("id") if isinstance(item, dict) else None
                                row = next((entry for entry in manifest["backups"] if entry.get("id") == backup_id), None)
                                if row is None:
                                    raise HTTPException(status_code=409, detail="guarded SQLite preservation is unavailable")
                                candidate = _history_subdirectory(root, "candidates", create=True) / f".guarded-{uuid4()}.sqlite3"
                                try:
                                    self._materialize_row_sync(root, row, candidate)
                                    self._publish_verified_candidate(
                                        root,
                                        f"candidates/{candidate.name}",
                                        str(row["sha256"]),
                                        files,
                                        str(row["database_name"]),
                                        verification_error="guarded SQLite restore verification failed",
                                    )
                                finally:
                                    candidate.unlink(missing_ok=True)
                            operation.update(status="completed", completed_at=_now().isoformat(), git_error=git_error)
                            self._save(root, manifest)

                    await run_sqlite_blocking(republish)
                    completed = finished
                    completed_result = {"operation_id": operation_id, "status": "completed", "git_error": git_error}
            # Do not wake finish until its task has exited the worker lease and
            # global gate.  A successful HTTP response therefore never races a
            # still-held fence or a pending ContextVar reset.
            if completed is not None and completed_result is not None:
                completed.set_result(completed_result)
        except BaseException as exc:
            if not begun.done():
                begun.set_exception(exc)
            if finish.done():
                try:
                    _, _, finished = finish.result()
                except BaseException:
                    finished = None
                if finished is not None and not finished.done():
                    finished.set_exception(exc)
            raise

    async def verify_guarded_code_restore_lease(self, workspace_id: str, operation_id: str) -> dict[str, Any]:
        """Return the exact live authority a trusted runtime Git substep binds to."""
        workspace_id = validate_workspace_id(workspace_id)
        held = self._guarded_restores.get(operation_id)
        if held is None or held.get("workspace_id") != workspace_id:
            raise HTTPException(status_code=409, detail="guarded SQLite restore lease is unavailable")
        lease_id = held.get("lease_id")
        files = held.get("files")
        if not isinstance(lease_id, str) or not isinstance(files, Path):
            raise HTTPException(status_code=409, detail="guarded SQLite restore lease is unavailable")
        async with self._installed():
            marker = await run_sqlite_blocking(read_marker, self._root(workspace_id) / MARKER_NAME)
        if marker is None or marker.get("lease_id") != lease_id:
            raise HTTPException(status_code=423, detail="guarded SQLite restore lease ownership was lost")
        return {"operation_id": operation_id, "workspace_id": workspace_id, "lease_id": lease_id, "authoritative_root": str(files)}

    async def recover_guarded_code_restore(self, workspace_id: str, operation_id: str) -> dict[str, Any]:
        """Restart recovery: reattach the marker lease, republish, then release."""
        workspace_id = validate_workspace_id(workspace_id)
        held = self._guarded_restores.get(operation_id)
        if held is not None and held.get("workspace_id") == workspace_id and not held["task"].done():
            raise HTTPException(status_code=409, detail="guarded SQLite restore lease is active")
        root = self._root(workspace_id)
        async with self._installed():
            marker = await run_sqlite_blocking(read_marker, root / MARKER_NAME)
            lease_id = str(marker.get("lease_id") or "") if marker else ""
            if not lease_id:
                raise HTTPException(status_code=409, detail="guarded SQLite restore recovery is unavailable")
            async with sqlite_workspace_recovery(workspace_id, lease_id):
                result = await self._runtime.worker.acquire_sqlite_workspace_access(workspace_id, lease_id, maintenance=True)
                files = Path(str(result["authoritative_root"]))
                try:

                    def republish() -> None:
                        with _catalog_lock(root):
                            manifest = self._load(root, workspace_id)
                            operation = manifest["operations"].get(operation_id)
                            if (
                                not operation
                                or operation.get("kind") != "guarded_code_restore"
                                or operation.get("status") != "active"
                                or operation.get("lease_id") != lease_id
                            ):
                                raise HTTPException(status_code=409, detail="guarded SQLite restore is not recoverable")
                            for item in operation.get("preserved", []):
                                backup_id = item.get("id") if isinstance(item, dict) else None
                                row = next((entry for entry in manifest["backups"] if entry.get("id") == backup_id), None)
                                if row is None:
                                    raise HTTPException(status_code=409, detail="guarded SQLite preservation is unavailable")
                                candidate = _history_subdirectory(root, "candidates", create=True) / f".guarded-recover-{uuid4()}.sqlite3"
                                self._materialize_row_sync(root, row, candidate)
                                self._publish_verified_candidate(
                                    root,
                                    f"candidates/{candidate.name}",
                                    str(row["sha256"]),
                                    files,
                                    str(row["database_name"]),
                                    verification_error="guarded SQLite recovery verification failed",
                                )
                                candidate.unlink(missing_ok=True)
                            operation.update(status="completed", completed_at=_now().isoformat(), recovered=True)
                            self._save(root, manifest)

                    await run_sqlite_blocking(republish)
                finally:
                    await self._runtime.worker.release_sqlite_workspace_access(workspace_id, lease_id)
                await recover_sqlite_workspace_maintenance(workspace_id, lease_id, action="complete")
        return {"operation_id": operation_id, "status": "completed", "recovered": True}

    async def finish_guarded_code_restore(self, workspace_id: str, operation_id: str, user_id: str, *, git_error: str | None = None) -> dict[str, Any]:
        workspace_id = validate_workspace_id(workspace_id)
        held = self._guarded_restores.get(operation_id)
        if held is None or held.get("workspace_id") != workspace_id:
            raise HTTPException(status_code=409, detail="guarded SQLite restore lease is unavailable")
        if held.get("workspace_id") != workspace_id:
            raise HTTPException(status_code=409, detail="guarded SQLite restore lease is unavailable")
        task = held.get("task")
        if not isinstance(task, asyncio.Task) or task.done():
            raise HTTPException(status_code=409, detail="guarded SQLite restore lease is unavailable")
        begun: asyncio.Future[dict[str, Any]] = held["begun"]
        await asyncio.shield(begun)
        finish: asyncio.Future[tuple[str, str | None, asyncio.Future[dict[str, Any]]]] = held["finish"]
        if finish.done():
            raise HTTPException(status_code=409, detail="guarded SQLite restore finish is already in progress")
        completed: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        finish.set_result((user_id, git_error, completed))
        return await asyncio.shield(completed)

    @asynccontextmanager
    async def _installed(self) -> AsyncIterator[None]:
        # Every runtime-owned history operation shares one durable gate across
        # catalogs and repository work.  Export/import take its exclusive side,
        # producing a self-consistent repository/catalog cut without taking a
        # workspace fence while waiting for that global barrier.
        from .storage import AsyncRepositoryGate, held_repository_fds

        async with AsyncRepositoryGate(self._runtime.root, exclusive=False):
            if os.path.lexists(self._runtime.root / "_sqlite_history" / "transfers" / "import-journal.json"):
                raise HTTPException(status_code=503, detail="SQLite history import recovery is required")
            token = _runtime_context.set(self._runtime)
            fds_token = _capture_pass_fds.set(tuple(dict.fromkeys((*_capture_pass_fds.get(), *held_repository_fds()))))
            try:
                yield
            finally:
                _capture_pass_fds.reset(fds_token)
                _runtime_context.reset(token)

    async def list_backups(self, workspace_id: str, **filters: Any) -> list[dict[str, Any]]:
        async with self._installed():
            return await super().list_backups(workspace_id, **filters)

    async def delete(self, workspace_id: str, backup_id: str) -> None:
        async with self._installed():
            await super().delete(workspace_id, backup_id)

    async def download_path(self, workspace_id: str, backup_id: str) -> Path:
        async with self._installed():
            return await super().download_path(workspace_id, backup_id)

    async def download_to_path(self, workspace_id: str, backup_id: str, destination: Path) -> Path:
        """Materialize an export into a trusted runtime-private destination."""
        source = await self.download_path(workspace_id, backup_id)
        try:
            await run_sqlite_blocking(shutil.copyfile, source, destination)
        finally:
            source.unlink(missing_ok=True)
        return destination

    async def capture_workspace_databases(self, workspace_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        async with self._installed():
            return await super().capture_workspace_databases(workspace_id, **kwargs)

    async def preview(self, workspace_id: str, backup_id: str, **kwargs: Any) -> dict[str, Any]:
        async with self._installed():
            return await super().preview(workspace_id, backup_id, **kwargs)

    async def apply(self, workspace_id: str, preview_id: str, **kwargs: Any) -> dict[str, Any]:
        async with self._installed():
            return await super().apply(workspace_id, preview_id, **kwargs)

    async def recover_operation(self, workspace_id: str, operation_id: str, **kwargs: Any) -> dict[str, Any]:
        async with self._installed():
            root = self._root(validate_workspace_id(workspace_id))

            def guarded() -> bool:
                with _catalog_lock(root):
                    operation = self._load(root, workspace_id).get("operations", {}).get(operation_id)
                    return bool(isinstance(operation, dict) and operation.get("kind") == "guarded_code_restore" and operation.get("status") == "active")

            if await run_sqlite_blocking(guarded):
                return await self.recover_guarded_code_restore(workspace_id, operation_id)
            return await super().recover_operation(workspace_id, operation_id, **kwargs)

    async def interrupted_maintenance(self, workspace_id: str) -> dict[str, Any] | None:
        async with self._installed():
            return await super().interrupted_maintenance(workspace_id)

    async def run_maintenance_once(self) -> None:
        """Runtime-local cleanup; queue scheduling is not activated in Wave A."""
        async with self._installed():
            workspaces = self._runtime.root / "workspaces"
            if not workspaces.is_dir():
                return
            for entry in workspaces.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    root = self._root(entry.name)
                    await run_sqlite_blocking(self._cleanup_and_due_sync, root, entry.name)

    def start(self) -> None:
        """Start runtime-local cleanup plus repository maintenance lifecycle."""
        super().start()
        if self._runtime_maintenance is None:
            from .maintenance import RuntimeHistoryMaintenance

            self._runtime_maintenance = RuntimeHistoryMaintenance(self)
        self._runtime_maintenance.start()

    async def stop(self) -> None:
        active_guards = [
            operation_id for operation_id, held in self._guarded_restores.items() if isinstance(held.get("task"), asyncio.Task) and not held["task"].done()
        ]
        if active_guards:
            # Do not cancel a guard here: cancellation would release a live
            # worker fence without a finish receipt.  Its owning task remains
            # registered for the caller to finish; process loss is recovered
            # from the durable marker/catalog record.
            logger.warning("SQLite guarded restore lifecycles remain active during shutdown operation_ids=%s", active_guards)
        if self._runtime_maintenance is not None:
            await self._runtime_maintenance.stop()
        await super().stop()


_history_service: RuntimeSqliteHistoryService | None = None


def get_runtime_sqlite_history_service(root: Path, worker: Any, repository: Any | None = None) -> RuntimeSqliteHistoryService:
    """Explicit runtime construction; activation/wiring stays in the next wave."""
    return RuntimeSqliteHistoryService(root, worker, repository)
