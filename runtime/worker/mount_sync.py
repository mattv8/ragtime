"""Pinned-descriptor, confined rsync synchronization for copied mounts."""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

from . import mount_sync_launcher

_RSYNC = "/usr/bin/rsync"
_MAX_DIAGNOSTICS_BYTES = 16 * 1024


class MountSyncError(RuntimeError):
    """Rsync started, or launcher validation failed, without successful sync."""


class MountSyncUnavailable(RuntimeError):
    """Confinement or rsync is unavailable before a transfer begins."""


class MountSyncCancelled(MountSyncError):
    """The cooperative cancellation event stopped a started transfer."""


def mount_sync_available() -> bool:
    """Return whether this host can start the confined helper (no side effects)."""
    if sys.platform != "linux" or not (os.path.isfile(_RSYNC) and os.access(_RSYNC, os.X_OK)):
        return False
    try:
        mount_sync_launcher.trial_ruleset()
    except OSError:
        return False
    return True


def _validate(source_fd: int, destination_fd: int, protected_paths: Sequence[str], timeout_seconds: float) -> tuple[str, ...]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    for fd in (source_fd, destination_fd):
        if fd < 0 or not os.path.isdir(f"/proc/self/fd/{fd}"):
            raise ValueError(f"descriptor {fd} is not an open directory")
    source = os.fstat(source_fd)
    destination = os.fstat(destination_fd)
    if (source.st_dev, source.st_ino) == (destination.st_dev, destination.st_ino):
        raise ValueError("source and destination are identical")
    source_path = os.path.realpath(f"/proc/self/fd/{source_fd}")
    destination_path = os.path.realpath(f"/proc/self/fd/{destination_fd}")
    try:
        common = os.path.commonpath((source_path, destination_path))
    except ValueError:
        common = ""
    if common in {source_path, destination_path}:
        raise ValueError("source and destination roots overlap")
    result: list[str] = []
    for path in protected_paths:
        if not isinstance(path, str) or not path or path.startswith("/") or "\x00" in path:
            raise ValueError("protected paths must be non-empty relative paths")
        parts = path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("protected paths must be normalized relative paths")
        result.append(path)
    return tuple(result)


def _diagnostics(handle: BinaryIO) -> str:
    handle.flush()
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - _MAX_DIAGNOSTICS_BYTES))
    return handle.read().decode(errors="replace").strip()


def _require_linked_directory(fd: int, label: str) -> os.stat_result:
    directory_stat = os.fstat(fd)
    if directory_stat.st_nlink == 0:
        raise MountSyncError(f"{label} directory descriptor refers to a retired directory")
    return directory_stat


def _source_generation_changed(initial: os.stat_result, source_fd: int) -> bool:
    current = os.fstat(source_fd)
    return current.st_nlink == 0 or (current.st_dev, current.st_ino, current.st_ctime_ns) != (initial.st_dev, initial.st_ino, initial.st_ctime_ns)


def _active_group_members(pgid: int) -> tuple[int, ...]:
    """Return live (non-zombie) members of the helper's owned new session."""
    active: list[int] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(entry.path + "/stat", encoding="utf-8") as stat_file:
                fields = stat_file.read().rsplit(")", 1)[1].split()
            # Fields after comm are state, ppid, pgrp, then session.  Checking
            # both avoids treating an unrelated reused process-group ID as our
            # child group after the launcher has exited.
            if int(fields[2]) == pgid and int(fields[3]) == pgid and fields[0] != "Z":
                active.append(int(entry.name))
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return tuple(active)


def _wait_for_group_quiescence(pgid: int, deadline: float) -> bool:
    while _active_group_members(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _terminate_group(process: subprocess.Popen[bytes]) -> bool:
    """Stop and drain every live member of the helper's newly owned session."""
    pgid = process.pid  # ``start_new_session`` makes this an owned SID/PGID.
    if _active_group_members(pgid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
    term_deadline = time.monotonic() + 1.0
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=max(0.0, term_deadline - time.monotonic()))
    if _wait_for_group_quiescence(pgid, term_deadline):
        return True
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    kill_deadline = time.monotonic() + 2.0
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
    return _wait_for_group_quiescence(pgid, kill_deadline)


def _drain_unreapable_group(process: subprocess.Popen[bytes]) -> None:
    """Do not return while this helper group might still mutate the destination."""
    while not _terminate_group(process):
        time.sleep(0.1)


def sync_copied_mount(
    source_fd: int, destination_fd: int, *, protected_paths: Sequence[str] = (), cancel_event: threading.Event | None = None, timeout_seconds: float = 180.0
) -> None:
    """Synchronize pinned directories without closing their caller-owned FDs.

    Landlock confines content access. It intentionally cannot mediate metadata
    syscalls such as chmod, utimensat, or setxattr under a hostile writer.
    """
    protected = _validate(source_fd, destination_fd, protected_paths, timeout_seconds)
    initial_source = _require_linked_directory(source_fd, "source")
    _require_linked_directory(destination_fd, "destination")
    if cancel_event is not None and cancel_event.is_set():
        raise MountSyncCancelled("mount sync cancelled before launch")
    if not mount_sync_available():
        raise MountSyncUnavailable("rsync or Landlock ABI 3 confinement is unavailable")
    launcher_path = str(Path(mount_sync_launcher.__file__).resolve())
    argv = [sys.executable, launcher_path, "--source-fd", str(source_fd), "--destination-fd", str(destination_fd)]
    for path in protected:
        argv.append(f"--protected-path={path}")
    with tempfile.TemporaryFile(mode="w+b") as stderr:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            pass_fds=(source_fd, destination_fd),
            close_fds=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        reason: str | None = None
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                reason = "cancelled"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "timed out"
                break
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        if reason:
            if not _terminate_group(process):
                _drain_unreapable_group(process)
                if reason == "cancelled":
                    raise MountSyncCancelled("mount sync cancelled after extended process-group drain")
                raise MountSyncError(f"mount sync {reason}; process group required extended drain")
            if reason == "cancelled":
                raise MountSyncCancelled("mount sync cancelled")
            raise MountSyncError(f"mount sync timed out after {timeout_seconds:g} seconds")
        returncode = process.wait()
        lingering = not _wait_for_group_quiescence(process.pid, time.monotonic())
        if lingering and not _terminate_group(process):
            _drain_unreapable_group(process)
            raise MountSyncError("mount sync exited with a process group requiring extended drain")
        diagnostic = _diagnostics(stderr)
    if returncode == 0:
        if _source_generation_changed(initial_source, source_fd):
            raise MountSyncError("source directory changed generation during mount sync")
        if lingering:
            raise MountSyncError("mount sync exited successfully but left child processes running")
        return
    if returncode == 77:
        raise MountSyncUnavailable(diagnostic or "child confinement unavailable before rsync execution")
    detail = f": {diagnostic}" if diagnostic else ""
    raise MountSyncError(f"mount sync failed with exit status {returncode}{detail}")
