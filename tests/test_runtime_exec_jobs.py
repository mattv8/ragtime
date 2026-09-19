from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from runtime.worker import service as worker_service


class _Reader:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)

    async def read(self, _size: int) -> bytes:
        return next(self._chunks)


class _Process:
    def __init__(self, chunks: list[bytes]) -> None:
        self.stdout = _Reader(chunks)
        self.returncode = 0

    async def wait(self) -> int:
        return self.returncode


class _BlockingSpawn:
    def __init__(self, process: _Process) -> None:
        self.process = process
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, *_args, **_kwargs) -> _Process:
        self.entered.set()
        await self.release.wait()
        return self.process


class _BlockingReader(_Reader):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def read(self, _size: int) -> bytes:
        self.entered.set()
        await self.release.wait()
        return b""


class RuntimeExecJobTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, root: Path) -> worker_service.WorkerSession:
        return worker_service.WorkerSession(
            id="worker-session",
            workspace_id="workspace",
            provider_session_id="provider",
            workspace_root=root,
            workspace_files_path=root / "files",
            sandbox_spec=worker_service.SandboxSpec(workspace_id="workspace", workspace_files_path=root / "files", rootfs_path=root / "rootfs"),
            pty_access_token="token",
            workspace_env={},
            workspace_env_visibility={},
            workspace_mounts=[],
            mount_targets_to_clear=set(),
            state="running",
            devserver_running=False,
            devserver_port=None,
            devserver_command=None,
            launch_framework=None,
            launch_cwd=None,
            last_error=None,
            runtime_operation_id=None,
            runtime_operation_phase=None,
            runtime_operation_started_at=None,
            runtime_operation_updated_at=None,
            updated_at=worker_service.utc_now(),
        )

    async def test_job_is_addressable_and_reads_incremental_bounded_output(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            with mock.patch.object(worker_service, "spawn_sandboxed", new=mock.AsyncMock(return_value=_Process([b"first-", b"second", b""]))):
                started = await service.start_exec_job(session.id, "echo test", user_id="user", credential_id="credential")
                await service._exec_job_tasks[started.id]
            first = await service.get_exec_job(session.id, started.id, cursor=0, limit=6)
            second = await service.get_exec_job(session.id, started.id, cursor=first.next_cursor, limit=64)

        self.assertEqual(first.status, "completed")
        self.assertEqual(first.output, "first-")
        self.assertEqual(second.output, "second")
        self.assertEqual(second.user_id, "user")
        self.assertEqual(second.credential_id, "credential")

    async def test_start_job_rejects_session_still_starting(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            session.state = "starting"
            service._sessions[session.id] = session

            with self.assertRaises(HTTPException) as error:
                await service.start_exec_job(session.id, "echo test")

        self.assertEqual(error.exception.status_code, 409)

    async def test_workspace_running_quota_is_two_jobs(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            first = await service.start_exec_job(session.id, "sleep")
            second = await service.start_exec_job(session.id, "sleep")
            with self.assertRaises(HTTPException) as error:
                await service.start_exec_job(session.id, "sleep")
            for job_id in (first.id, second.id):
                service._exec_job_tasks[job_id].cancel()
            await asyncio.gather(*service._exec_job_tasks.values(), return_exceptions=True)

        self.assertEqual(error.exception.status_code, 429)

    async def test_immediately_cancelled_queued_job_never_spawns(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            spawn = mock.AsyncMock()
            with mock.patch.object(worker_service, "spawn_sandboxed", new=spawn):
                started = await service.start_exec_job(session.id, "echo test")
                cancelled = await service.cancel_exec_job(session.id, started.id)
                await service._exec_job_tasks[started.id]

        self.assertEqual(cancelled.status, "cancelled")
        spawn.assert_not_awaited()

    async def test_cancel_during_spawn_terminates_process_before_it_can_run(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            spawn = _BlockingSpawn(_Process([b""]))
            terminate = mock.AsyncMock()
            with (
                mock.patch.object(worker_service, "spawn_sandboxed", new=spawn),
                mock.patch.object(worker_service, "terminate_process_group", new=terminate),
            ):
                started = await service.start_exec_job(session.id, "echo test")
                await spawn.entered.wait()
                cancelled = await service.cancel_exec_job(session.id, started.id)
                spawn.release.set()
                await service._exec_job_tasks[started.id]

        self.assertEqual(cancelled.status, "cancelled")
        terminate.assert_awaited_once()

    async def test_cancel_running_job_terminates_its_process_group(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            process = _Process([])
            reader = _BlockingReader()
            process.stdout = reader

            async def terminate(running_process: _Process) -> None:
                running_process.returncode = -15
                reader.release.set()

            with (
                mock.patch.object(worker_service, "spawn_sandboxed", new=mock.AsyncMock(return_value=process)),
                mock.patch.object(worker_service, "terminate_process_group", new=mock.AsyncMock(side_effect=terminate)) as terminate_mock,
            ):
                started = await service.start_exec_job(session.id, "echo test")
                await reader.entered.wait()
                cancelled = await service.cancel_exec_job(session.id, started.id)
                await service._exec_job_tasks[started.id]

        self.assertEqual(cancelled.status, "cancelled")
        terminate_mock.assert_awaited_once_with(process)

    async def test_stop_during_spawn_terminates_process_after_spawn_race(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            spawn = _BlockingSpawn(_Process([b""]))
            terminate = mock.AsyncMock()
            with (
                mock.patch.object(worker_service, "spawn_sandboxed", new=spawn),
                mock.patch.object(worker_service, "terminate_process_group", new=terminate),
                mock.patch.object(worker_service, "cleanup_sandbox"),
            ):
                started = await service.start_exec_job(session.id, "echo test")
                await spawn.entered.wait()
                await service.stop_session(session.id)
                spawn.release.set()
                await service._exec_job_tasks[started.id]

        self.assertEqual(service._exec_jobs[started.id].status, "interrupted")
        terminate.assert_awaited_once()

    async def test_timeout_includes_sandbox_spawn(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            spawn = _BlockingSpawn(_Process([b""]))
            with mock.patch.object(worker_service, "spawn_sandboxed", new=spawn):
                started = await service.start_exec_job(session.id, "echo test", timeout_seconds=1)
                await spawn.entered.wait()
                await service._exec_job_tasks[started.id]

        self.assertEqual(service._exec_jobs[started.id].status, "timed_out")

    async def test_job_output_is_bounded_and_reports_truncation_cursor(self) -> None:
        service = worker_service.WorkerService()
        service._EXEC_JOB_MAX_OUTPUT_BYTES = 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            with mock.patch.object(worker_service, "spawn_sandboxed", new=mock.AsyncMock(return_value=_Process([b"abcdef", b""]))):
                started = await service.start_exec_job(session.id, "echo test")
                await service._exec_job_tasks[started.id]
                result = await service.get_exec_job(session.id, started.id, cursor=0, limit=64)

        self.assertEqual(result.output, "cdef")
        self.assertEqual(result.truncated_before, 2)

    async def test_job_output_redacts_a_secret_split_across_chunks_and_ledger(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            session.workspace_env = {"API_KEY": "very-secret"}
            service._sessions[session.id] = session
            with mock.patch.object(
                worker_service,
                "spawn_sandboxed",
                new=mock.AsyncMock(return_value=_Process([b"API_KEY=very-", b"secret\n", b""])),
            ):
                started = await service.start_exec_job(session.id, "echo test")
                await service._exec_job_tasks[started.id]
                result = await service.get_exec_job(session.id, started.id, cursor=0, limit=64)
            ledger = (root / ".runtime-exec-jobs.json").read_text(encoding="utf-8")

        self.assertNotIn("very-secret", result.output)
        self.assertIn("API_KEY=*****", result.output)
        self.assertNotIn("very-secret", ledger)

    async def test_job_output_preserves_unicode_split_across_chunks(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "files").mkdir()
            session = self._session(root)
            service._sessions[session.id] = session
            with mock.patch.object(
                worker_service,
                "spawn_sandboxed",
                new=mock.AsyncMock(return_value=_Process([b"price: \xe2\x82", b"\xac\n", b""])),
            ):
                started = await service.start_exec_job(session.id, "echo test")
                await service._exec_job_tasks[started.id]
                result = await service.get_exec_job(session.id, started.id, cursor=0, limit=64)

        self.assertEqual(result.output, "price: €\n")
