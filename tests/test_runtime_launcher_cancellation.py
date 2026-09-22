from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from runtime.worker import sandbox

_FAKE_LAUNCHER = r"""
import os
import signal
import struct
import sys

spec_fd = int(sys.argv[1])
status_fd = int(sys.argv[2])
header = os.read(spec_fd, 4)
if len(header) != 4:
    raise SystemExit(21)
remaining = struct.unpack(">I", header)[0]
while remaining:
    chunk = os.read(spec_fd, remaining)
    if not chunk:
        raise SystemExit(22)
    remaining -= len(chunk)
os.close(spec_fd)
os.write(1, b"R")
while True:
    signal.pause()
"""


class SandboxLauncherCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_keeps_reader_fd_open_until_real_launcher_is_reaped(self) -> None:
        """Closing the reader or releasing startup before cleanup finishes is the bug."""
        real_create_subprocess_exec = asyncio.create_subprocess_exec
        real_read_status = sandbox._read_launch_status_from_fd
        real_cleanup = sandbox._cleanup_failed_launcher_process
        loop = asyncio.get_running_loop()
        reader_started = asyncio.Event()
        reader_finished = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        spawned: asyncio.subprocess.Process | None = None
        startup_task: asyncio.Task[asyncio.subprocess.Process] | None = None
        reader_fd: int | None = None

        async def create_fake_launcher(*_args, **kwargs):
            nonlocal spawned
            spec_fd, status_fd = kwargs["pass_fds"]
            kwargs["stdout"] = asyncio.subprocess.PIPE
            spawned = await real_create_subprocess_exec(
                sys.executable,
                "-c",
                _FAKE_LAUNCHER,
                str(spec_fd),
                str(status_fd),
                **kwargs,
            )
            return spawned

        def read_status(fd: int):
            nonlocal reader_fd
            reader_fd = fd
            loop.call_soon_threadsafe(reader_started.set)
            try:
                return real_read_status(fd)
            finally:
                loop.call_soon_threadsafe(reader_finished.set)

        async def blocked_cleanup(process: asyncio.subprocess.Process) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()
            await real_cleanup(process)

        capabilities = sandbox.SandboxCapabilities(mode="chroot")
        spec = sandbox.SandboxSpec(
            workspace_id="launcher-cancellation",
            workspace_files_path=Path("/tmp/launcher-cancellation/files"),
            rootfs_path=Path("/tmp/launcher-cancellation/rootfs"),
            mode="chroot",
        )

        try:
            with (
                mock.patch.object(sandbox, "detect_capabilities", return_value=capabilities),
                mock.patch.object(asyncio, "create_subprocess_exec", side_effect=create_fake_launcher),
                mock.patch.object(sandbox, "_read_launch_status_from_fd", side_effect=read_status),
                mock.patch.object(sandbox, "_cleanup_failed_launcher_process", side_effect=blocked_cleanup),
            ):
                startup_task = asyncio.create_task(sandbox.spawn_sandboxed(spec, ["true"], ensure_ready=False))
                await asyncio.wait_for(reader_started.wait(), timeout=2.0)
                self.assertIsNotNone(spawned)
                self.assertIsNotNone(reader_fd)
                spawned_process = cast(asyncio.subprocess.Process, spawned)
                status_read_fd = cast(int, reader_fd)
                stdout = spawned_process.stdout
                assert stdout is not None
                self.assertEqual(await asyncio.wait_for(stdout.readexactly(1), timeout=2.0), b"R")

                startup_task.cancel()
                await asyncio.wait_for(cleanup_started.wait(), timeout=2.0)
                self.assertFalse(startup_task.done())
                os.fstat(status_read_fd)

                startup_task.cancel()
                checkpoint = asyncio.Event()
                loop.call_soon(checkpoint.set)
                await asyncio.wait_for(checkpoint.wait(), timeout=1.0)
                self.assertFalse(startup_task.done())
                os.fstat(status_read_fd)

                allow_cleanup.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(startup_task), timeout=3.0)

                await asyncio.wait_for(reader_finished.wait(), timeout=1.0)
                self.assertIsNotNone(spawned_process.returncode)
                with self.assertRaises(OSError):
                    os.fstat(status_read_fd)
        finally:
            allow_cleanup.set()
            if spawned is not None and spawned.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    spawned.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(spawned.wait(), timeout=2.0)
            if startup_task is not None and not startup_task.done():
                startup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(asyncio.shield(startup_task), timeout=2.0)
            if reader_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(reader_fd)


if __name__ == "__main__":
    unittest.main()
