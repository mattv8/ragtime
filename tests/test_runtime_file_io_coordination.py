from __future__ import annotations

import asyncio
import importlib
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fastapi import HTTPException

worker_service: Any = importlib.import_module("runtime.worker.service")


class RuntimeFileCoordinationTests(unittest.IsolatedAsyncioTestCase):
    def _install(self, service: Any, root: Path, session_id: str, workspace_id: str) -> Any:
        files = root / "files"
        files.mkdir()
        session = worker_service.WorkerSession(
            id=session_id,
            workspace_id=workspace_id,
            provider_session_id=session_id,
            workspace_root=root,
            workspace_files_path=files,
            sandbox_spec=worker_service.SandboxSpec(workspace_id=workspace_id, workspace_files_path=files, rootfs_path=root / "rootfs"),
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
            runtime_operation_id="op",
            runtime_operation_phase="ready",
            runtime_operation_started_at=None,
            runtime_operation_updated_at=None,
            updated_at=worker_service.utc_now(),
        )
        service._sessions[session_id] = session
        return session

    async def test_same_workspace_serializes_but_other_workspace_remains_responsive(self) -> None:
        service = worker_service.WorkerService()
        started = threading.Event()
        release_first = threading.Event()
        active_by_root: dict[Path, int] = {}
        peak_for_first = 0

        def blocked_write(root: Path, rel: str, content: str) -> None:
            nonlocal peak_for_first
            active_by_root[root] = active_by_root.get(root, 0) + 1
            if root.name == "files" and root.parent == Path(first):
                peak_for_first = max(peak_for_first, active_by_root[root])
            if root.parent == Path(first):
                started.set()
                release_first.wait(timeout=2)
            (root / rel).write_text(content)
            active_by_root[root] -= 1

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = self._install(service, Path(first), "one", "workspace-one")
            two = self._install(service, Path(second), "two", "workspace-two")
            with mock.patch("runtime.worker.service.secure_write_text", side_effect=blocked_write):
                first_write = asyncio.create_task(service.write_file(one.id, "a.txt", "one"))
                await asyncio.to_thread(started.wait, 1)
                second_same = asyncio.create_task(service.write_file(one.id, "b.txt", "two"))
                await asyncio.wait_for(service.write_file(two.id, "free.txt", "free"), timeout=0.25)
                self.assertFalse(release_first.is_set())
                self.assertFalse(second_same.done())
                release_first.set()
                await asyncio.gather(first_write, second_same)
        self.assertEqual(peak_for_first, 1)

    async def test_cancelled_file_io_drains_before_mount_refresh_materializes(self) -> None:
        service = worker_service.WorkerService()
        started = threading.Event()
        release = threading.Event()

        def blocked_write(root: Path, rel: str, content: str) -> None:
            started.set()
            release.wait(timeout=2)

        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            with (
                mock.patch("runtime.worker.service.secure_write_text", side_effect=blocked_write),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()) as materialize,
            ):
                write = asyncio.create_task(service.write_file(session.id, "a.txt", "one"))
                await asyncio.to_thread(started.wait, 1)
                write.cancel()
                refresh = asyncio.create_task(
                    service.refresh_mounts(session.id, [{"target_path": "/workspace/data", "source_local_path": str(Path(tmp) / "mount")}])
                )
                await asyncio.sleep(0)
                self.assertEqual(materialize.await_count, 0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await write
                await refresh
                materialize.assert_awaited_once()

    async def test_queued_write_recaptures_readonly_mount_after_file_lock(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_mount = root / "old"
            new_mount = root / "new"
            old_mount.mkdir()
            new_mount.mkdir()
            session = self._install(
                service,
                root,
                "one",
                "workspace-one",
            )
            session.workspace_mounts = [{"target_path": "/workspace/data", "source_local_path": str(old_mount), "read_only": False}]
            file_lock = service._workspace_file_lock(session.workspace_id)
            await file_lock.acquire()
            try:
                write = asyncio.create_task(service.write_file(session.id, "data/file.txt", "blocked"))
                await asyncio.sleep(0)
                async with service._lock:
                    session.workspace_mounts = [{"target_path": "/workspace/data", "source_local_path": str(new_mount), "read_only": True}]
                file_lock.release()
                with self.assertRaises(HTTPException) as error:
                    await write
            finally:
                if file_lock.locked():
                    file_lock.release()
            self.assertEqual(error.exception.status_code, 403)
            self.assertFalse((old_mount / "file.txt").exists())
            self.assertFalse((new_mount / "file.txt").exists())

    async def test_stop_cancels_tracked_startup_without_cleanup_lock_deadlock(self) -> None:
        """A stop barrier must not wait on startup while startup owns its lock."""
        service = worker_service.WorkerService()
        startup_entered = asyncio.Event()
        startup_cancelled = asyncio.Event()
        followup_started = asyncio.Event()

        async def blocked_bootstrap(_session: Any) -> str | None:
            startup_entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                startup_cancelled.set()
                raise
            raise AssertionError("blocked bootstrap future unexpectedly completed")

        async def followup_bootstrap(_session: Any) -> str | None:
            followup_started.set()
            return "expected follow-up stop"

        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            with (
                mock.patch("runtime.worker.service.ensure_sandbox_ready"),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", side_effect=blocked_bootstrap),
            ):
                async with service._lock:
                    service._schedule_startup_locked(session)
                await asyncio.wait_for(startup_entered.wait(), timeout=0.5)
                response = await asyncio.wait_for(service.stop_session(session.id), timeout=0.5)

            self.assertTrue(startup_cancelled.is_set())
            self.assertEqual(response.state, "stopped")
            self.assertNotIn(session.workspace_id, service._workspace_cleanup_tasks)

            with (
                mock.patch("runtime.worker.service.ensure_sandbox_ready"),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", side_effect=followup_bootstrap),
            ):
                async with service._lock:
                    service._schedule_startup_locked(session)
                    followup = service._startup_tasks[session.id]
                await asyncio.wait_for(followup_started.wait(), timeout=0.5)
                await asyncio.wait_for(followup, timeout=0.5)
