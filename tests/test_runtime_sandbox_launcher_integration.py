from __future__ import annotations

# Privileged local coverage command:
# docker run --rm --privileged --entrypoint python -e PYTHONPATH=/repo -v "/Users/elfy/GitRepos/ragtime:/repo" -w /repo docker-runtime tests/test_runtime_sandbox_launcher_integration.py -v
import asyncio
import contextlib
import json
import os
import signal
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from runtime.worker import sandbox


class RuntimeSandboxLauncherIntegrationTests(unittest.IsolatedAsyncioTestCase):
    _caps: sandbox.SandboxCapabilities

    @classmethod
    def setUpClass(cls) -> None:
        if sys.platform != "linux":
            raise unittest.SkipTest("sandbox launcher integration requires Linux")
        if os.geteuid() != 0:
            raise unittest.SkipTest("sandbox launcher integration requires root")

        sandbox._capabilities_cache.clear()
        cls._caps = sandbox.detect_capabilities()
        if not cls._caps.has_cap_sys_admin:
            raise unittest.SkipTest("sandbox launcher integration requires CAP_SYS_ADMIN")
        if not cls._caps.pid_namespace:
            raise unittest.SkipTest("sandbox launcher integration requires PID namespace support")

    def setUp(self) -> None:
        sandbox._capabilities_cache.clear()

    def tearDown(self) -> None:
        sandbox._capabilities_cache.clear()

    def _build_spec(self, workspace_root: Path, *, mode: str = "chroot") -> sandbox.SandboxSpec:
        files = workspace_root / "files"
        files.mkdir(parents=True, exist_ok=True)
        return sandbox.SandboxSpec(
            workspace_id=f"launcher-int-{uuid.uuid4().hex[:12]}",
            workspace_files_path=files,
            rootfs_path=workspace_root / "rootfs",
            mode=mode,
        )

    async def _cleanup_spec(self, spec: sandbox.SandboxSpec) -> None:
        await asyncio.to_thread(sandbox.cleanup_sandbox, spec)

    async def _wait_for_cgroup_empty(self, spec: sandbox.SandboxSpec, *, timeout: float = 5.0) -> bool:
        cgroup_path = sandbox._sandbox_cgroup_path(spec, self._caps)
        if cgroup_path is None:
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not sandbox._read_cgroup_process_ids(cgroup_path):
                return True
            await asyncio.sleep(0.05)
        self.fail(f"sandbox cgroup still contains processes: {sandbox._read_cgroup_process_ids(cgroup_path)}")

    async def _wait_for_pid_exit(self, pid: int, *, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            with contextlib.suppress(OSError, ValueError):
                stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
                if len(stat_fields) >= 3 and stat_fields[2] == "Z":
                    return
            await asyncio.sleep(0.05)
        self.fail(f"process {pid} did not exit")

    async def _read_pty_until(self, master_fd: int, expected: bytes, *, timeout: float = 10.0) -> bytes:
        os.set_blocking(master_fd, False)
        deadline = time.monotonic() + timeout
        chunks = bytearray()
        while time.monotonic() < deadline:
            try:
                chunk = os.read(master_fd, 4096)
            except BlockingIOError:
                chunk = b""
            except OSError:
                break
            if chunk:
                chunks.extend(chunk)
                if expected in chunks:
                    return bytes(chunks)
            await asyncio.sleep(0.05)
        self.fail(f"PTY output never contained {expected!r}; got {bytes(chunks)!r}")

    async def test_spawn_sandboxed_runs_under_tini_pid_namespace_and_handles_64_sequential_forks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._build_spec(Path(tmpdir))
            process = await sandbox.spawn_sandboxed(
                spec,
                ["bash", "-lc", "for i in $(seq 1 64); do /bin/true || exit 1; done; cat /proc/1/comm; echo $$"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await process.communicate()
                self.assertEqual(process.returncode, 0, stderr.decode())
                lines = stdout.decode().splitlines()
                self.assertGreaterEqual(len(lines), 2, stdout.decode())
                self.assertEqual(lines[0], "tini")
                self.assertGreaterEqual(int(lines[1]), 2)
                self.assertNotIn("Cannot allocate memory", stderr.decode())
            finally:
                await self._cleanup_spec(spec)

    async def test_spawn_sandboxed_pivot_root_runs_under_tini_pid_namespace_and_handles_64_sequential_forks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._build_spec(Path(tmpdir), mode="pivot_root")
            process = await sandbox.spawn_sandboxed(
                spec,
                ["bash", "-lc", "for i in $(seq 1 64); do /bin/true || exit 1; done; cat /proc/1/comm; echo $$"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await process.communicate()
                self.assertEqual(process.returncode, 0, stderr.decode())
                lines = stdout.decode().splitlines()
                self.assertGreaterEqual(len(lines), 2, stdout.decode())
                self.assertEqual(lines[0], "tini")
                self.assertGreaterEqual(int(lines[1]), 2)
                self.assertNotIn("Cannot allocate memory", stderr.decode())
            finally:
                await self._cleanup_spec(spec)

    async def test_spawn_sandboxed_maps_missing_executable_to_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._build_spec(Path(tmpdir))
            try:
                with self.assertRaises(FileNotFoundError):
                    await sandbox.spawn_sandboxed(
                        spec,
                        ["definitely-not-a-real-executable-ragtime"],
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
            finally:
                await self._cleanup_spec(spec)

    async def test_spawn_sandboxed_pty_ctrl_c_keeps_shell_reusable(self) -> None:
        master_fd, slave_fd = os.openpty()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                spec = self._build_spec(Path(tmpdir))
                process = await sandbox.spawn_sandboxed(
                    spec,
                    ["bash", "--noprofile", "--norc", "-i"],
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    pty=True,
                    env={"PS1": "", "PROMPT_COMMAND": ""},
                )
                try:
                    os.write(master_fd, b"sleep 30\n")
                    await asyncio.sleep(0.3)
                    os.write(master_fd, b"\x03")
                    await asyncio.sleep(0.2)
                    os.write(master_fd, b"echo PTY_REUSED\nexit\n")
                    output = await self._read_pty_until(master_fd, b"PTY_REUSED")
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                    self.assertEqual(process.returncode, 0, output.decode(errors="replace"))
                finally:
                    with contextlib.suppress(Exception):
                        await sandbox.terminate_process_group(process)
                    await self._cleanup_spec(spec)
        finally:
            os.close(master_fd)
            os.close(slave_fd)

    async def test_launcher_parent_death_sigkill_cleans_up_launcher_and_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._build_spec(Path(tmpdir))
            helper_code = """
import asyncio
import json
from pathlib import Path
from runtime.worker import sandbox

async def main() -> None:
    spec = sandbox.SandboxSpec(
        workspace_id={workspace_id!r},
        workspace_files_path=Path({workspace_files_path!r}),
        rootfs_path=Path({rootfs_path!r}),
        mode={mode!r},
    )
    process = await sandbox.spawn_sandboxed(
        spec,
        ["bash", "-lc", "echo ready >/workspace/ready.txt; sleep 30"],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    info = {{
        "launcher_pid": process.pid,
        "cgroup_path": str(sandbox._sandbox_cgroup_path(spec, sandbox.detect_capabilities())),
    }}
    print(json.dumps(info), flush=True)
    await asyncio.sleep(30)

asyncio.run(main())
""".format(
                workspace_id=spec.workspace_id,
                workspace_files_path=str(spec.workspace_files_path),
                rootfs_path=str(spec.rootfs_path),
                mode=spec.mode,
            )
            helper = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                helper_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            try:
                assert helper.stdout is not None
                line = await asyncio.wait_for(helper.stdout.readline(), timeout=15.0)
                self.assertTrue(line, "helper produced no launcher metadata")
                info = json.loads(line.decode())
                launcher_pid = int(info["launcher_pid"])

                os.kill(helper.pid, signal.SIGKILL)
                await asyncio.wait_for(helper.wait(), timeout=10.0)
                await self._wait_for_pid_exit(launcher_pid)
                await self._wait_for_cgroup_empty(spec)
            finally:
                if helper.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        helper.kill()
                    with contextlib.suppress(Exception):
                        await helper.wait()
                await self._cleanup_spec(spec)

    async def test_sandbox_cgroup_is_empty_after_process_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._build_spec(Path(tmpdir))
            if sandbox._sandbox_cgroup_path(spec, self._caps) is None:
                self.skipTest("sandbox cgroup pid tracking unavailable")
            process = await sandbox.spawn_sandboxed(
                spec,
                ["bash", "-lc", "sleep 0.2"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await process.communicate()
                self.assertEqual(process.returncode, 0, stderr.decode())
                self.assertEqual(stdout, b"")
                await self._wait_for_cgroup_empty(spec)
            finally:
                await self._cleanup_spec(spec)

    async def test_spawn_sandboxed_communicate_closes_stdin_with_eof(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._build_spec(Path(tmpdir))
            process = await sandbox.spawn_sandboxed(
                spec,
                [
                    "python3",
                    "-c",
                    "import sys; data = sys.stdin.buffer.read(); sys.stdout.write(str(len(data)) + '\\n')",
                ],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await process.communicate(b"abcdef")
                self.assertEqual(process.returncode, 0, stderr.decode())
                self.assertEqual(stdout.decode().strip(), "6")
            finally:
                await self._cleanup_spec(spec)


if __name__ == "__main__":
    unittest.main()
