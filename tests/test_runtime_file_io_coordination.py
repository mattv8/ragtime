from __future__ import annotations

import asyncio
import hashlib
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
            workspace_env={
                "RAGTIME_BRIDGE_TOKEN_FILE": worker_service.RUNTIME_BRIDGE_TOKEN_FILE_PATH,
            },
            workspace_env_visibility={},
            bridge_token_file_initial_token="test-bridge-token",
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

    async def test_write_uses_worker_authoritative_tree_and_enforces_compare_and_swap(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            mirror = session.sandbox_spec.rootfs_path / "workspace"
            mirror.mkdir(parents=True)
            mirror.joinpath("app.txt").write_text("shell", encoding="utf-8")

            with mock.patch("runtime.worker.service.workspace_mirror_required", return_value=True):
                read = await service.read_file(session.id, "app.txt")
                self.assertEqual(read.content, "shell")
                self.assertEqual(read.content_hash, hashlib.sha256(b"shell").hexdigest())
                self.assertEqual(read.updated_at, read.actual_updated_at)
                self.assertIsNone(read.artifact_metadata)

                with self.assertRaises(HTTPException) as error:
                    await service.write_file(session.id, "app.txt", "api", expected_content_hash="stale", require_content_hash=True)
                self.assertEqual(error.exception.status_code, 409)

                written = await service.write_file(
                    session.id,
                    "app.txt",
                    "api",
                    expected_content_hash=read.content_hash,
                    require_content_hash=True,
                    artifact_metadata={"artifact_type": "module_ts", "live_data_connections": [{"component_id": "tool"}]},
                )

            self.assertEqual(written.content, "api")
            self.assertEqual(written.artifact_metadata, {"artifact_type": "module_ts", "live_data_connections": [{"component_id": "tool"}]})
            self.assertEqual(mirror.joinpath("app.txt").read_text(encoding="utf-8"), "api")
            self.assertEqual(
                mirror.joinpath("app.txt.artifact.json").read_text(encoding="utf-8"),
                '{"artifact_type":"module_ts","live_data_connections":[{"component_id":"tool"}]}',
            )
            self.assertFalse(session.workspace_files_path.joinpath("app.txt").exists())

    async def test_require_content_hash_with_null_requires_absent_file(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            session.workspace_files_path.joinpath("app.txt").write_text("present", encoding="utf-8")
            with self.assertRaises(HTTPException) as error:
                await service.write_file(session.id, "app.txt", "new", require_content_hash=True)
            self.assertEqual(error.exception.status_code, 409)

    async def test_move_and_delete_preserve_active_root_and_sidecar(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            mirror = session.sandbox_spec.rootfs_path / "workspace"
            mirror.mkdir(parents=True)
            with mock.patch("runtime.worker.service.workspace_mirror_required", return_value=True):
                await service.write_file(session.id, "old.txt", "shell", artifact_metadata={"artifact_type": "module_ts"})
                moved = await service.move_file(session.id, "old.txt", "new.txt")
                self.assertTrue(moved["success"])
                read = await service.read_file(session.id, "new.txt")
                self.assertEqual(read.content, "shell")
                self.assertEqual(read.artifact_metadata, {"artifact_type": "module_ts"})
                await service.delete_file(session.id, "new.txt")
                self.assertFalse((mirror / "new.txt").exists())
                self.assertFalse((mirror / "new.txt.artifact.json").exists())

    async def test_read_holds_workspace_lock_through_metadata_capture(self) -> None:
        service = worker_service.WorkerService()
        stat_started = threading.Event()
        release_stat = threading.Event()
        original_stat = worker_service.secure_stat_file

        def blocked_stat(root: Path, rel_path: str) -> Any:
            stat_started.set()
            release_stat.wait(timeout=2)
            return original_stat(root, rel_path)

        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            session.workspace_files_path.joinpath("app.txt").write_text("before", encoding="utf-8")
            with mock.patch("runtime.worker.service.secure_stat_file", side_effect=blocked_stat):
                read = asyncio.create_task(service.read_file(session.id, "app.txt"))
                await asyncio.to_thread(stat_started.wait, 1)
                write = asyncio.create_task(service.write_file(session.id, "app.txt", "after"))
                await asyncio.sleep(0)
                self.assertEqual(session.workspace_files_path.joinpath("app.txt").read_text(encoding="utf-8"), "before")
                release_stat.set()
                response = await read
                await write

        self.assertIsNotNone(response.actual_updated_at)

    async def test_metadata_timestamp_is_optional_when_descriptor_stat_is_unavailable(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            session.workspace_files_path.joinpath("app.txt").write_text("content", encoding="utf-8")
            with mock.patch("runtime.worker.service.secure_stat_file", return_value=None):
                response = await service.read_file(session.id, "app.txt")

        self.assertIsNone(response.actual_updated_at)
        self.assertEqual(response.updated_at, session.updated_at)

    async def test_active_non_utf8_file_is_reported_as_existing_non_text(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            session.workspace_files_path.joinpath("binary.bin").write_bytes(b"\xff")
            response = await service.read_file(session.id, "binary.bin")

        self.assertTrue(response.exists)
        self.assertFalse(response.is_utf8_text)

    async def test_active_mirror_snapshots_store_git_objects_in_canonical_metadata(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            service._root = Path(tmp)
            (service._root / "workspaces" / "workspace-one").mkdir(parents=True)
            session = self._install(service, service._root / "workspaces" / "workspace-one", "one", "workspace-one")
            mirror = session.sandbox_spec.rootfs_path / "workspace"
            mirror.mkdir(parents=True)
            with mock.patch("runtime.worker.service.workspace_mirror_required", return_value=True):
                await service.run_workspace_git_command(session.workspace_id, args=["init"])
                await service.run_workspace_git_command(session.workspace_id, args=["config", "user.email", "test@example.test"])
                await service.run_workspace_git_command(session.workspace_id, args=["config", "user.name", "Test"])
                mirror.joinpath("app.txt").write_text("one", encoding="utf-8")
                await service.run_workspace_git_command(session.workspace_id, args=["add", "app.txt"])
                await service.run_workspace_git_command(session.workspace_id, args=["commit", "-m", "one"])
                first = await service.run_workspace_git_command(session.workspace_id, args=["rev-parse", "HEAD"])
                mirror.joinpath("app.txt").write_text("two", encoding="utf-8")
                await service.run_workspace_git_command(session.workspace_id, args=["add", "app.txt"])
                await service.run_workspace_git_command(session.workspace_id, args=["commit", "-m", "two"])

            service._sessions.pop(session.id)
            first_commit = __import__("base64").b64decode(first.stdout_b64).decode("utf-8").strip()
            restored = await service.run_workspace_git_command(session.workspace_id, args=["show", f"{first_commit}:app.txt"])
            self.assertTrue(session.workspace_files_path.joinpath(".git", "objects").is_dir())

        self.assertEqual(restored.returncode, 0)
        self.assertEqual(__import__("base64").b64decode(restored.stdout_b64), b"one")

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

        def provision_sandbox(spec: Any) -> None:
            spec.rootfs_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp), "one", "workspace-one")
            with (
                mock.patch("runtime.worker.service.ensure_sandbox_ready", side_effect=provision_sandbox),
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
                mock.patch("runtime.worker.service.ensure_sandbox_ready", side_effect=provision_sandbox),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", side_effect=followup_bootstrap),
            ):
                async with service._lock:
                    service._schedule_startup_locked(session)
                    followup = service._startup_tasks[session.id]
                await asyncio.wait_for(followup_started.wait(), timeout=0.5)
                await asyncio.wait_for(followup, timeout=0.5)
