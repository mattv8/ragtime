"""Bounded, process-safe admission for confined SQLite capture children.

The public-request counter prevents an application's executor from accumulating
unbounded capture work.  The flock slots additionally coordinate independent
application processes which share ``INDEX_DATA_PATH``.  They deliberately live
outside workspace-controlled files.
"""

from __future__ import annotations

import fcntl
import os
import stat
import subprocess
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import HTTPException

from ragtime.config import settings
from ragtime.core.logging import get_logger

# These are intentionally fixed defaults rather than administrator settings.
# Keep them module-level so focused tests can lower them without changing the
# production contract.
MAX_OUTSTANDING_CAPTURE_REQUESTS = 16
MAX_CONCURRENT_CAPTURE_SUBPROCESSES = 2
CAPTURE_SLOT_WAIT_SECONDS = 5.0

_SLOT_DIRECTORY = "sqlite_capture_slots"
_SLOT_PREFIX = "capture-"
_SLOT_SUFFIX = ".lock"
_admission_lock = threading.Lock()
_outstanding_requests = 0
_waiting_slot_callers = 0
logger = get_logger(__name__)
_capture_inherited_fds: ContextVar[tuple[int, ...]] = ContextVar("sqlite_capture_inherited_fds", default=())


@contextmanager
def inherit_capture_fds(fds: tuple[int, ...]) -> Any:
    """Keep controller liveness descriptors open in confined capture children."""
    token = _capture_inherited_fds.set(tuple(dict.fromkeys(fd for fd in fds if fd >= 0)))
    try:
        yield
    finally:
        _capture_inherited_fds.reset(token)


def _busy(detail: str = "SQLite capture capacity is busy; retry shortly") -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)


def _open_directory_chain(path: Path) -> int:
    """Open/create an absolute directory path without following any component."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    fd = os.open("/", _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child_fd = os.open(component, _directory_flags(), dir_fd=fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    # Another app process created it after our no-follow open.
                    # Re-open it below with the same no-follow validation.
                    pass
                child_fd = os.open(component, _directory_flags(), dir_fd=fd)
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                os.close(child_fd)
                raise OSError("capture slot parent is not a directory")
            os.close(fd)
            fd = child_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_slot_directory() -> int:
    """Return a descriptor for ``INDEX_DATA_PATH/_userspace/sqlite_capture_slots``."""
    root = Path(settings.index_data_path)
    root_fd = _open_directory_chain(root)
    try:
        userspace_fd = -1
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
                slot_fd = os.open(_SLOT_DIRECTORY, _directory_flags(), dir_fd=userspace_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(_SLOT_DIRECTORY, mode=0o700, dir_fd=userspace_fd)
                except FileExistsError:
                    pass
                slot_fd = os.open(_SLOT_DIRECTORY, _directory_flags(), dir_fd=userspace_fd)
            return slot_fd
        finally:
            if userspace_fd >= 0:
                os.close(userspace_fd)
    finally:
        os.close(root_fd)


def _try_acquire_slot(slot_directory_fd: int) -> int | None:
    """Acquire one regular no-follow flock slot, if immediately available."""
    flags = os.O_CREAT | os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    for index in range(MAX_CONCURRENT_CAPTURE_SUBPROCESSES):
        name = f"{_SLOT_PREFIX}{index}{_SLOT_SUFFIX}"
        fd = os.open(name, flags, 0o600, dir_fd=slot_directory_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("capture slot is not a regular file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            return fd
        except BaseException:
            # A nonregular or otherwise unsafe slot must fail closed rather
            # than silently trying another slot in a compromised directory.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    return None


def _release_slot(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@asynccontextmanager
async def capture_request_admission() -> AsyncIterator[None]:
    """Admit a public async capture request or immediately return HTTP 503.

    Callers must put this *outside* their workspace/catalog work.  Its finalizer
    runs only after ``run_sqlite_blocking`` has drained a cancelled worker.
    """
    global _outstanding_requests
    with _admission_lock:
        if _outstanding_requests >= MAX_OUTSTANDING_CAPTURE_REQUESTS:
            logger.info("SQLite capture request admission outcome=busy queue_wait_ms=0 duration_ms=0")
            raise _busy()
        _outstanding_requests += 1
    started = time.monotonic()
    try:
        yield
    finally:
        with _admission_lock:
            _outstanding_requests -= 1
        logger.info(
            "SQLite capture request admission outcome=released queue_wait_ms=%d duration_ms=%d",
            0,
            int((time.monotonic() - started) * 1000),
        )


def run_admitted_subprocess(command: list[str], **subprocess_kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Run one confined child while holding a cross-process capture slot.

    A caller waiting for a slot is bounded both in time and locally.  This
    function intentionally does not retry subprocesses and never includes the
    command, paths, or child output in logs.
    """
    global _waiting_slot_callers
    wait_started = time.monotonic()
    counted_waiter = False
    with _admission_lock:
        if _waiting_slot_callers >= MAX_OUTSTANDING_CAPTURE_REQUESTS:
            logger.info("SQLite capture subprocess outcome=busy queue_wait_ms=0 duration_ms=0")
            raise _busy()
        _waiting_slot_callers += 1
        counted_waiter = True

    slot_fd = -1
    try:
        try:
            slot_directory_fd = _open_slot_directory()
        except (OSError, ValueError) as exc:
            raise _busy("SQLite capture admission storage is unavailable") from exc
        try:
            deadline = time.monotonic() + CAPTURE_SLOT_WAIT_SECONDS
            while slot_fd < 0:
                try:
                    acquired_slot = _try_acquire_slot(slot_directory_fd)
                except OSError as exc:
                    raise _busy("SQLite capture admission storage is unavailable") from exc
                if acquired_slot is not None:
                    slot_fd = acquired_slot
                if slot_fd >= 0:
                    break
                if time.monotonic() >= deadline:
                    raise _busy()
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        finally:
            os.close(slot_directory_fd)

        with _admission_lock:
            _waiting_slot_callers -= 1
            counted_waiter = False
        duration_started = time.monotonic()
        try:
            supplied_fds = tuple(subprocess_kwargs.pop("pass_fds", ()))
            # ``pass_fds`` is POSIX-only, as is the flock admission mechanism.
            # Include the slot itself so a controller crash cannot release it
            # while its child is still copying a database.
            subprocess_kwargs["pass_fds"] = tuple(dict.fromkeys((*supplied_fds, *_capture_inherited_fds.get(), slot_fd)))
            completed = subprocess.run(command, **subprocess_kwargs)
        except BaseException:
            logger.info(
                "SQLite capture subprocess outcome=error queue_wait_ms=%d duration_ms=%d",
                int((duration_started - wait_started) * 1000),
                int((time.monotonic() - duration_started) * 1000),
            )
            raise
        logger.info(
            "SQLite capture subprocess outcome=completed queue_wait_ms=%d duration_ms=%d",
            int((duration_started - wait_started) * 1000),
            int((time.monotonic() - duration_started) * 1000),
        )
        return completed
    except HTTPException:
        logger.info(
            "SQLite capture subprocess outcome=busy queue_wait_ms=%d duration_ms=0",
            int((time.monotonic() - wait_started) * 1000),
        )
        raise
    finally:
        if slot_fd >= 0:
            _release_slot(slot_fd)
        if counted_waiter:
            with _admission_lock:
                _waiting_slot_callers -= 1
