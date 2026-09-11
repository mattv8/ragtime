"""Bounded, supervised subprocesses for resource-admitted indexing work.

Workers are intentionally one-shot.  This trades a little process-startup
throughput for a simple aggregate live-worker ceiling and, importantly, means
native allocations cannot remain resident in an idle job-owned pool.
"""

from __future__ import annotations

import asyncio
import ctypes
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import replace
from multiprocessing.process import BaseProcess
from typing import Any

from ragtime.core.logging import get_logger
from ragtime.indexer.resource_governor import ResourceRequest, resource_governor

logger = get_logger(__name__)

_POLL_SECONDS = 0.05
_TERM_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 0.5
# Task descriptors should be tiny.  Bulk text, chunks, and vectors belong in a
# task-private spool file, never on this pipe.
MAX_RESULT_IPC_BYTES = 4 * 1024 * 1024


class ResourceTaskError(RuntimeError):
    """A supervised worker failed without returning a usable descriptor."""


class ResourceTaskTimeout(ResourceTaskError):
    """The task exceeded its admitted deadline."""


class ResourceTaskMemoryExceeded(ResourceTaskError):
    """The task exceeded the peak memory envelope in its request."""

    def __init__(self, request: ResourceRequest, observed_peak_bytes: int) -> None:
        self.request = request
        self.observed_peak_bytes = max(0, int(observed_peak_bytes))
        super().__init__(f"{request.stage} task exceeded {request.estimated_peak_bytes} byte envelope (observed peak {self.observed_peak_bytes} bytes)")


class ResourceTaskResultTooLarge(ResourceTaskError):
    """A worker attempted to send corpus-sized data over IPC."""


def _install_parent_death_guard(expected_parent_pid: int) -> None:
    """Ensure Linux children die if their supervisor disappears.

    ``PR_SET_PDEATHSIG`` has a race: the parent can die between process start
    and prctl.  Checking the parent after installation closes that race.
    """

    def install_watchdog() -> None:
        # macOS/other supported platforms lack prctl.  It is also the fallback
        # when Linux prctl is unavailable or rejected by a sandbox.
        def watch_parent() -> None:
            while os.getppid() == expected_parent_pid:
                time.sleep(0.1)
            os._exit(1)

        threading.Thread(target=watch_parent, name="index-worker-parent-watch", daemon=True).start()

    if not sys.platform.startswith("linux"):
        install_watchdog()
        return
    if os.name != "posix" or not hasattr(signal, "SIGKILL"):
        install_watchdog()
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            install_watchdog()
            return
        if os.getppid() != expected_parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)
    except Exception:
        install_watchdog()
        return


def _send_result(connection: Any, payload: tuple[str, Any]) -> None:
    try:
        encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        if len(encoded) > MAX_RESULT_IPC_BYTES:
            connection.send_bytes(pickle.dumps(("result_too_large", None)))
        else:
            connection.send_bytes(encoded)
    except Exception:
        # The parent will report an abnormal worker exit if this fallback fails.
        try:
            connection.send_bytes(pickle.dumps(("result_serialization_failed", None)))
        except Exception:
            pass


def _worker_entry(connection: Any, parent_pid: int, function: Callable[..., Any], args: tuple[Any, ...]) -> None:
    _install_parent_death_guard(parent_pid)
    try:
        result = function(*args)
        _send_result(connection, ("ok", result))
    except BaseException as exc:
        _send_result(connection, ("exception", (type(exc).__name__, str(exc), traceback.format_exc(limit=20))))
    finally:
        connection.close()


def _terminate_process(process: BaseProcess) -> None:
    if not process.is_alive():
        process.join(timeout=0)
        return
    process.terminate()
    process.join(timeout=_TERM_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=_KILL_GRACE_SECONDS)


def _receive_bounded(connection: Any) -> tuple[str, Any]:
    """Read a complete framed result off-loop without accepting oversized IPC."""
    try:
        payload = connection.recv_bytes(MAX_RESULT_IPC_BYTES)
    except OSError as exc:
        raise ResourceTaskResultTooLarge("worker result exceeded bounded IPC limit") from exc
    try:
        return pickle.loads(payload)
    except (EOFError, pickle.UnpicklingError) as exc:
        raise ResourceTaskError("worker returned a truncated result") from exc


async def _reap_process(process: BaseProcess) -> None:
    """Join/kill outside the API loop and finish reaping despite cancellation."""
    cleanup = asyncio.create_task(asyncio.to_thread(_terminate_process, process))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        # A cancelled caller must not release its governor charge while a
        # native child still exists.
        await asyncio.shield(cleanup)
        raise


def _rss_bytes(pid: int) -> int:
    try:
        import psutil

        return int(psutil.Process(pid).memory_info().rss)
    except Exception:
        return 0


async def run_resource_task(
    request: ResourceRequest,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    timeout_seconds: float = 300.0,
) -> Any:
    """Run one serializable task under one governor lease.

    The lease covers the whole child lifetime.  We deliberately do not retain
    idle executors: a second job cannot quietly multiply resident processes,
    and cancelled work can be reaped independently of another job.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    # A sampled native-parser cold start can exceed a conservative initial
    # estimate.  Retry exactly once with the measured peak plus headroom.  Each
    # pass has a separate lease, so the larger attempt is admitted normally.
    current_request = request
    for attempt in range(2):
        try:
            return await _run_resource_task_once(current_request, function, args, timeout_seconds=timeout_seconds)
        except ResourceTaskMemoryExceeded as error:
            if attempt:
                raise
            measured = max(error.observed_peak_bytes, current_request.estimated_peak_bytes)
            headroom = max(64 * 1024 * 1024, measured // 5)
            current_request = replace(current_request, estimated_peak_bytes=measured + headroom)


async def _run_resource_task_once(
    request: ResourceRequest,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    timeout_seconds: float,
) -> Any:
    async with resource_governor.acquire(request) as lease:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_worker_entry, args=(sender, os.getpid(), function, args))
        started = time.monotonic()
        peak_rss = 0
        failed = True
        reaped = False
        try:
            process.start()
            sender.close()
            process_pid = process.pid
            if process_pid is None:
                raise ResourceTaskError("worker did not report a process id after start")
            resource_governor.track_worker(process_pid, request.job_id)
            resource_governor.set_worker_active(process_pid, True)

            while True:
                elapsed = time.monotonic() - started
                peak_rss = max(peak_rss, _rss_bytes(process_pid))
                lease.report_resident_bytes(peak_rss)
                lease.report_peak_bytes(peak_rss)
                if request.estimated_peak_bytes and peak_rss > request.estimated_peak_bytes:
                    await _reap_process(process)
                    reaped = True
                    raise ResourceTaskMemoryExceeded(request, peak_rss)
                if elapsed > timeout_seconds:
                    await _reap_process(process)
                    reaped = True
                    raise ResourceTaskTimeout(f"{request.stage} task exceeded {timeout_seconds:.1f}s deadline")
                if receiver.poll():
                    try:
                        payload = await asyncio.to_thread(_receive_bounded, receiver)
                    except ResourceTaskResultTooLarge:
                        await _reap_process(process)
                        reaped = True
                        raise
                    except (EOFError, OSError, ResourceTaskError) as exc:
                        # A killed child makes Pipe.poll() true and recv_bytes()
                        # raise EOFError. Reap before raising so the lease stays
                        # charged until the child is definitely gone.
                        await _reap_process(process)
                        reaped = True
                        raise ResourceTaskError(f"worker exited with code {process.exitcode} before returning a usable result") from exc
                    await _reap_process(process)
                    reaped = True
                    kind, value = payload
                    if kind == "ok":
                        failed = False
                        return value
                    if kind == "result_too_large":
                        raise ResourceTaskResultTooLarge("worker result exceeded bounded IPC limit")
                    if kind == "exception":
                        name, message, details = value
                        raise ResourceTaskError(f"worker {name}: {message}\n{details}")
                    raise ResourceTaskError(str(value))
                if not process.is_alive():
                    await _reap_process(process)
                    reaped = True
                    raise ResourceTaskError(f"worker exited with code {process.exitcode} before returning a result")
                await asyncio.sleep(_POLL_SECONDS)
        except asyncio.CancelledError:
            if process.pid and not reaped:
                await _reap_process(process)
                reaped = True
            raise
        finally:
            sender.close()
            receiver.close()
            if process.pid:
                if not reaped:
                    await _reap_process(process)
                resource_governor.set_worker_active(process.pid, False)
                resource_governor.untrack_worker(process.pid)
            resource_governor.record_outcome(
                stage=request.stage,
                elapsed_seconds=time.monotonic() - started,
                peak_bytes=peak_rss,
                failed=failed,
            )
            if failed:
                logger.warning(
                    "Resource task failed: job=%s stage=%s source=%s peak_rss_bytes=%d",
                    request.job_id,
                    request.stage,
                    request.diagnostic_source or "unknown",
                    peak_rss,
                )
