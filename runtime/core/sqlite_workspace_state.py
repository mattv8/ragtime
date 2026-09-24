"""Runtime-safe durable SQLite workspace fencing and marker primitives.

The marker and operation lock names deliberately match the pre-runtime inspector
paths, so both owners coordinate through the same kernel locks.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

_WORKSPACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
MARKER_NAME = "sqlite-maintenance-intent.json"
OPERATION_LOCK_NAME = "sqlite-operation.lock"


class SqliteWorkspaceStateError(RuntimeError):
    pass


def validate_workspace_id(workspace_id: str) -> str:
    if not _WORKSPACE_ID.fullmatch(str(workspace_id or "")):
        raise SqliteWorkspaceStateError("invalid workspace ID")
    return workspace_id


def workspace_history_root(root: Path, workspace_id: str) -> Path:
    # Validate before composing any filesystem path.  Runtime callers receive
    # workspace IDs over authenticated HTTP, not from a trusted local iterator.
    return root / "workspaces" / validate_workspace_id(workspace_id) / "sqlite_backups"


def _mkdir_safe(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise SqliteWorkspaceStateError("SQLite workspace directory is unsafe")


def _open_lock(path: Path) -> int:
    _mkdir_safe(path.parent)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SqliteWorkspaceStateError("SQLite lock is unsafe")
    return fd


@contextmanager
def workspace_operation(root: Path, workspace_id: str, *, exclusive: bool, nonblocking: bool = False) -> Iterator[None]:
    fd = _open_lock(workspace_history_root(root, workspace_id) / OPERATION_LOCK_NAME)
    try:
        fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (fcntl.LOCK_NB if nonblocking else 0))
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def read_marker(path: Path) -> dict[str, Any] | None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(details.st_mode):
        raise SqliteWorkspaceStateError("SQLite maintenance marker is unsafe")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "r", encoding="utf-8") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise SqliteWorkspaceStateError("SQLite maintenance marker is unsafe")
            payload = json.load(source)
    except (OSError, ValueError) as exc:
        raise SqliteWorkspaceStateError("SQLite maintenance marker is unsafe") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("lease_id"), str):
        raise SqliteWorkspaceStateError("SQLite maintenance marker is unsafe")
    return payload


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def claim_marker(path: Path, payload: dict[str, Any]) -> None:
    _mkdir_safe(path.parent)
    lock = _open_lock(path.with_name(path.name + ".lock"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.lexists(path):
            raise SqliteWorkspaceStateError("SQLite workspace maintenance is active")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        with temporary.open("x", encoding="utf-8") as out:
            json.dump(payload, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def replace_owned_marker(path: Path, lease_id: str, payload: dict[str, Any]) -> None:
    if payload.get("lease_id") != lease_id:
        raise SqliteWorkspaceStateError("marker ownership lost")
    lock = _open_lock(path.with_name(path.name + ".lock"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (read_marker(path) or {}).get("lease_id") != lease_id:
            raise SqliteWorkspaceStateError("marker ownership lost")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        with temporary.open("x", encoding="utf-8") as out:
            json.dump(payload, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def remove_owned_marker(path: Path, lease_id: str) -> bool:
    lock = _open_lock(path.with_name(path.name + ".lock"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (read_marker(path) or {}).get("lease_id") != lease_id:
            return False
        path.unlink()
        _fsync_directory(path.parent)
        return True
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)
