from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

from runtime.worker import sandbox

try:
    WorkerService = importlib.import_module("runtime.worker.service").WorkerService
except ImportError:
    WorkerService = None


@unittest.skipUnless(sys.platform == "linux", "process-group assertions require Linux")
class RuntimeStartupProcessCleanupTests(unittest.IsolatedAsyncioTestCase):
    class _BlockedProcess:
        returncode = None

        def __init__(self) -> None:
            self.communicate_started = asyncio.Event()

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_started.set()
            await asyncio.Future()
            return b"", b""

    async def test_exited_leader_with_live_descendant_is_not_signalled_by_unverified_pgid(self) -> None:
        """A dead leader's PGID alone must not target a possibly reused group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_pid_path = Path(tmpdir) / "child.pid"
            process = await asyncio.create_subprocess_exec(
                "sh",
                "-c",
                f"sleep 30 & printf %s $! > {child_pid_path}; exit 0",
                start_new_session=True,
            )
            sandbox._capture_process_group_ownership(process)
            await process.wait()
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            try:
                await sandbox.terminate_process_group(process, timeout=0.2)
                stat_path = Path(f"/proc/{child_pid}/stat")
                if stat_path.exists():
                    self.assertNotEqual(stat_path.read_text(encoding="utf-8").split()[2], "Z")
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child_pid, signal.SIGKILL)

    async def test_extinct_recorded_group_is_not_signalled(self) -> None:
        process = SimpleNamespace(pid=424242, returncode=0, _ragtime_owned_process_group=424242)
        with mock.patch.object(sandbox.os, "killpg") as killpg:
            await sandbox.terminate_process_group(cast(asyncio.subprocess.Process, process))
        killpg.assert_not_called()

    async def test_readiness_fails_when_process_exits_before_http_serves(self) -> None:
        service_class = WorkerService or importlib.import_module("runtime.worker.service").WorkerService
        service = service_class()
        service._devserver_start_timeout_seconds = 10
        process = SimpleNamespace(returncode=1)
        with mock.patch("runtime.worker.service.httpx.AsyncClient") as client_type:
            self.assertFalse(await service._wait_devserver_ready(43123, process))
        client_type.return_value.__aenter__.return_value.get.assert_not_awaited()

    async def test_bootstrap_communication_timeout_reaps_real_child(self) -> None:
        service_class = WorkerService or importlib.import_module("runtime.worker.service").WorkerService
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "sleep 30",
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        sandbox._capture_process_group_ownership(process)
        with self.assertRaises(asyncio.TimeoutError):
            await service_class()._communicate_or_terminate(process, timeout=0.01)
        self.assertIsNotNone(process.returncode)

    async def test_bootstrap_and_dependency_cancellation_wait_for_process_drain(self) -> None:
        """Both wrapper paths retain cancellation until their child cleanup ends."""
        service_class = WorkerService or importlib.import_module("runtime.worker.service").WorkerService
        service = service_class()
        drain_started = asyncio.Event()
        allow_drain = asyncio.Event()

        async def terminate(_process: object) -> None:
            drain_started.set()
            await allow_drain.wait()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            files = root / "files"
            files.mkdir()
            session = SimpleNamespace(
                workspace_files_path=files,
                sandbox_spec=SimpleNamespace(rootfs_path=root / "rootfs"),
            )
            for wrapper, setup in (
                (
                    service._run_workspace_bootstrap_if_needed,
                    lambda: (
                        mock.patch.object(service, "_runtime_bootstrap_config_digest", new=mock.AsyncMock(return_value="new")),
                        mock.patch.object(service, "_read_runtime_bootstrap_config", new=mock.AsyncMock(return_value=[{"run": "sleep 30"}])),
                        mock.patch.object(service, "_sync_missing_bootstrap_watch_paths_to_sandbox_sync"),
                    ),
                ),
                (
                    service._ensure_entrypoint_dependencies,
                    lambda: (mock.patch.object(service, "_read_runtime_entrypoint_config", return_value={"framework": "flask"}),),
                ),
            ):
                process = self._BlockedProcess()
                patches = setup()
                with contextlib.ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    stack.enter_context(mock.patch("runtime.worker.service.spawn_sandboxed", new=mock.AsyncMock(return_value=process)))
                    stack.enter_context(mock.patch("runtime.worker.service.terminate_process_group", side_effect=terminate))
                    task = asyncio.create_task(wrapper(session))
                    await asyncio.wait_for(process.communicate_started.wait(), timeout=1)
                    task.cancel()
                    await asyncio.wait_for(drain_started.wait(), timeout=1)
                    self.assertFalse(task.done())
                    allow_drain.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                allow_drain.clear()
                drain_started.clear()

    async def test_timing_entries_are_removed_for_missing_pipeline_and_stop(self) -> None:
        service_class = WorkerService or importlib.import_module("runtime.worker.service").WorkerService
        service = service_class()
        service._operation_monotonic_starts[("missing", "old-op")] = 1.0
        await service._run_startup_pipeline("missing", "old-op")
        self.assertNotIn(("missing", "old-op"), service._operation_monotonic_starts)
