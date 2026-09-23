"""Safe ownership and collection helpers for private SQLite-history scratch."""

from __future__ import annotations

import fcntl
import os
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCRATCH_PREFIX = ".sqlite-history-"
_OWNER_LOCK_NAME = ".owner.lock"


@contextmanager
def scratch_owner_lock(scratch: Path) -> Iterator[None]:
    """Mark scratch as owned and keep its flock until its creator cleans up."""
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(scratch / _OWNER_LOCK_NAME, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError("SQLite scratch owner lock is unsafe")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _owner_lock(directory: Path, *, nonblocking: bool) -> int | None:
    if not directory.name.startswith(SCRATCH_PREFIX):
        return None
    fd: int | None = None
    try:
        if not stat.S_ISDIR(directory.lstat().st_mode) or directory.is_symlink():
            return None
        fd = os.open(directory / _OWNER_LOCK_NAME, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None
        fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        return fd
    except (FileNotFoundError, NotADirectoryError, OSError, BlockingIOError):
        if fd is not None:
            os.close(fd)
        return None


def is_managed_scratch(directory: Path) -> bool:
    """Recognize only directories created by the lock-owning implementation."""
    fd = _owner_lock(directory, nonblocking=True)
    if fd is None:
        # A held owner lock still proves this is our managed layout.
        try:
            details = (directory / _OWNER_LOCK_NAME).lstat()
            return directory.name.startswith(SCRATCH_PREFIX) and stat.S_ISREG(details.st_mode)
        except OSError:
            return False
    try:
        return True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def regular_tree_bytes(directory: Path) -> int:
    """Count private temporary files without traversing directory symlinks."""
    if directory.is_symlink():
        return 0
    total = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    details = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(details.st_mode):
                    total += details.st_size
                elif stat.S_ISDIR(details.st_mode) and not entry.is_symlink():
                    total += regular_tree_bytes(Path(entry.path))
    except OSError:
        pass
    return total


def managed_scratch_bytes(parent: Path) -> int:
    """Measure direct managed scratch children without following symlinks."""
    total = 0
    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                path = Path(entry.path)
                if is_managed_scratch(path):
                    total += regular_tree_bytes(path)
    except OSError:
        pass
    return total


def remove_orphaned_scratch(parent: Path) -> None:
    """Remove only unlocked, marker-owned scratch; unknown directories survive."""
    try:
        entries = list(os.scandir(parent))
    except OSError:
        return
    for entry in entries:
        directory = Path(entry.path)
        fd = _owner_lock(directory, nonblocking=True)
        if fd is None:
            continue
        try:
            # rmtree treats a symlink root as a link, and ownership was checked
            # through a no-follow marker before this point.
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
