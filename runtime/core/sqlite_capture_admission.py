"""Runtime-local, inherited-FD admission for confined SQLite children."""

from __future__ import annotations

import fcntl
import os
import stat
import subprocess
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

MAX_CONCURRENT_CAPTURE_SUBPROCESSES = 2
CAPTURE_SLOT_WAIT_SECONDS = 5.0
_inherited: ContextVar[tuple[int, ...]] = ContextVar("sqlite_capture_inherited_fds", default=())


@contextmanager
def inherit_capture_fds(fds: tuple[int, ...]) -> Iterator[None]:
    token = _inherited.set(tuple(dict.fromkeys(fd for fd in fds if fd >= 0)))
    try:
        yield
    finally:
        _inherited.reset(token)


def run_admitted_subprocess(root: Path, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    slots = root / "_sqlite_history" / "capture_slots"
    slots.mkdir(parents=True, exist_ok=True)
    if slots.is_symlink() or not slots.is_dir():
        raise RuntimeError("capture admission storage is unsafe")
    deadline = time.monotonic() + CAPTURE_SLOT_WAIT_SECONDS
    slot = -1
    while slot < 0:
        for index in range(MAX_CONCURRENT_CAPTURE_SUBPROCESSES):
            fd = os.open(slots / f"capture-{index}.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise RuntimeError("capture admission storage is unsafe")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                slot = fd
                break
            except BlockingIOError:
                os.close(fd)
        if slot < 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("SQLite capture capacity is busy")
            time.sleep(0.05)
    try:
        supplied = tuple(kwargs.pop("pass_fds", ()))
        kwargs["pass_fds"] = tuple(dict.fromkeys((*supplied, *_inherited.get(), slot)))
        return subprocess.run(command, **kwargs)
    finally:
        fcntl.flock(slot, fcntl.LOCK_UN)
        os.close(slot)
