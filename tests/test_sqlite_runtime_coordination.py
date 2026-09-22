from __future__ import annotations

import asyncio
import json
import multiprocessing
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.sqlite_runtime import _acquire_operation_lock, _release_operation_lock


def _hold_shared_sqlite_operation_lock(path: str, ready, release) -> None:
    """Child-process fixture for advisory flock exclusion."""
    fd = _acquire_operation_lock(Path(path), exclusive=False)
    try:
        ready.set()
        release.wait()
    finally:
        _release_operation_lock(fd)


class SqliteRuntimeCoordinationTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_shared_process_lock_excludes_exclusive_maintenance(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            ready = context.Event()
            release = context.Event()
            lock_path = Path(temporary) / "sqlite_backups" / "sqlite-operation.lock"
            process = context.Process(target=_hold_shared_sqlite_operation_lock, args=(str(lock_path), ready, release))
            process.start()
            try:
                self.assertTrue(await asyncio.to_thread(ready.wait, 5), "child never acquired the shared lock")
                with self.assertRaises(BlockingIOError):
                    sqlite_runtime._acquire_operation_lock(lock_path, exclusive=True, nonblocking=True)
            finally:
                release.set()
                await asyncio.to_thread(process.join, 5)
                if process.is_alive():
                    process.kill()
                    await asyncio.to_thread(process.join, 5)
                self.assertFalse(process.is_alive(), "child did not exit after lock release")
            self.assertEqual(process.exitcode, 0)

    async def test_offline_shared_access_cannot_upgrade_to_maintenance(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=False),
            ):
                async with sqlite_runtime.sqlite_workspace_access("workspace-1"):
                    with self.assertRaises(HTTPException) as blocked:
                        async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                            pass
            self.assertEqual(blocked.exception.status_code, 423)

    async def test_recovery_rejects_an_active_shared_operation_before_marker_release(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            marker = data / "_userspace" / "workspaces" / "workspace-1" / "sqlite_backups" / "sqlite-maintenance-intent.json"
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            marker.parent.mkdir()
            marker.write_text('{"workspace_id":"workspace-1","lease_id":"lease-1","state":"active","origin":"online"}')
            request = mock.AsyncMock()
            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", request),
            ):
                # Use the raw operation context because ordinary access correctly
                # rejects the durable marker before admitting a new reader.
                async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=False):
                    self.assertTrue(await sqlite_runtime.sqlite_workspace_operation_active("workspace-1"))
                    with self.assertRaises(HTTPException) as blocked:
                        await sqlite_runtime.recover_sqlite_workspace_maintenance("workspace-1", "lease-1", action="abort")
            self.assertEqual(blocked.exception.status_code, 423)
            request.assert_not_awaited()
            self.assertTrue(marker.exists())

    async def test_cancelled_blocking_work_drains_before_cancellation_propagates(self) -> None:
        from ragtime.userspace.sqlite_runtime import run_sqlite_blocking

        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()

        def blocking() -> None:
            started.set()
            release.wait()
            completed.set()

        task = asyncio.create_task(run_sqlite_blocking(blocking))
        await asyncio.to_thread(started.wait)
        task.cancel()
        task.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(completed.is_set())

    async def test_cancelled_failing_work_logs_failure_and_reraises_cancellation(self) -> None:
        from ragtime.userspace.sqlite_runtime import run_sqlite_blocking

        started = threading.Event()
        release = threading.Event()

        def failing() -> None:
            started.set()
            release.wait()
            raise RuntimeError("drained failure")

        task = asyncio.create_task(run_sqlite_blocking(failing))
        await asyncio.to_thread(started.wait)
        task.cancel()
        task.cancel()
        release.set()
        with self.assertLogs("ragtime.userspace.sqlite_runtime", level="ERROR") as logs, self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIn("Cancelled SQLite blocking operation completed with an error", logs.output[0])

    async def test_cancelled_operation_acquire_releases_lock_after_thread_drain(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            workspace = data / "_userspace" / "workspaces" / "workspace-1"
            (workspace / "files").mkdir(parents=True)
            lock_path = workspace / "sqlite_backups" / "sqlite-operation.lock"
            shared_fd = sqlite_runtime._acquire_operation_lock(lock_path, exclusive=False)
            entered = threading.Event()
            original = sqlite_runtime._acquire_operation_lock

            def acquire(*args, **kwargs):
                entered.set()
                return original(*args, **kwargs)

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "_acquire_operation_lock", side_effect=acquire),
            ):
                task = asyncio.create_task(sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=True).__aenter__())
                await asyncio.to_thread(entered.wait)
                task.cancel()
                task.cancel()
                sqlite_runtime._release_operation_lock(shared_fd)
                with self.assertRaises(asyncio.CancelledError):
                    await task
            fd = sqlite_runtime._acquire_operation_lock(lock_path, exclusive=True, nonblocking=True)
            sqlite_runtime._release_operation_lock(fd)

    async def test_inherited_context_from_finished_owner_reacquires_a_real_lock(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            proceed = asyncio.Event()
            original = sqlite_runtime._acquire_operation_lock
            calls = 0

            def acquire(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs)

            async def child() -> None:
                await proceed.wait()
                async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=False):
                    self.assertTrue(await sqlite_runtime.sqlite_workspace_operation_active("workspace-1"))

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "_acquire_operation_lock", side_effect=acquire),
            ):
                async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=False):
                    child_task = asyncio.create_task(child())
                proceed.set()
                await child_task
            self.assertEqual(calls, 2)

    async def test_same_task_depth_three_reuses_one_lock_and_unwinds_cleanly(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            original = sqlite_runtime._acquire_operation_lock
            calls = 0

            def acquire(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs)

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "_acquire_operation_lock", side_effect=acquire),
            ):
                async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=False):
                    async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=False):
                        async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=False):
                            self.assertTrue(await sqlite_runtime.sqlite_workspace_operation_active("workspace-1"))
                    async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=False):
                        pass
                self.assertNotIn("workspace-1", sqlite_runtime._operation_held.get())
            self.assertEqual(calls, 1)

    async def test_exclusive_operation_allows_a_nested_shared_operation(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            with mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)):
                async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=True):
                    async with sqlite_runtime._sqlite_workspace_operation("workspace-1", exclusive=False):
                        self.assertTrue(await sqlite_runtime.sqlite_workspace_operation_active("workspace-1"))

    async def test_inherited_recovery_context_cannot_bypass_fresh_ownership(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            marker = data / "_userspace" / "workspaces" / "workspace-1" / "sqlite_backups" / "sqlite-maintenance-intent.json"
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            marker.parent.mkdir()
            marker.write_text('{"workspace_id":"workspace-1","lease_id":"lease-1","state":"active","origin":"online"}')
            proceed = asyncio.Event()

            async def child() -> None:
                await proceed.wait()
                await sqlite_runtime.recover_sqlite_workspace_maintenance("workspace-1", "lease-1", action="abort")

            request = mock.AsyncMock()
            original = sqlite_runtime.sqlite_workspace_recovery
            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", request),
                mock.patch.object(sqlite_runtime, "sqlite_workspace_recovery", wraps=original) as recovery,
            ):
                async with sqlite_runtime.sqlite_workspace_recovery("workspace-1", "lease-1"):
                    child_task = asyncio.create_task(child())
                proceed.set()
                await child_task
            self.assertEqual(recovery.call_count, 2)
            self.assertFalse(marker.exists())

    async def test_lock_symlinks_and_unsafe_parent_are_rejected(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external"
            external.mkdir()
            backups = root / "sqlite_backups"
            backups.symlink_to(external, target_is_directory=True)
            with self.assertRaises(HTTPException) as parent_blocked:
                sqlite_runtime._acquire_operation_lock(backups / "sqlite-operation.lock", exclusive=False)
            self.assertEqual(parent_blocked.exception.status_code, 423)
            backups.unlink()
            backups.mkdir()
            target = external / "lock-target"
            (backups / "sqlite-operation.lock").symlink_to(target)
            with self.assertRaises(HTTPException) as leaf_blocked:
                sqlite_runtime._acquire_operation_lock(backups / "sqlite-operation.lock", exclusive=False)
            self.assertEqual(leaf_blocked.exception.status_code, 423)
            self.assertFalse(target.exists())

    async def test_maintenance_pins_canonical_root_and_removes_marker_after_release(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            files = data / "_userspace" / "workspaces" / "workspace-1" / "files"
            files.mkdir(parents=True)

            async def request(method, path, **kwargs):
                if method == "POST":
                    return {"authoritative_root": str(files)}
                self.assertEqual(method, "DELETE")
                return {}

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=request),
            ):
                async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True) as root:
                    self.assertEqual(root, files.resolve())
                    self.assertTrue(sqlite_runtime._marker_path("workspace-1").exists())

            self.assertFalse(sqlite_runtime._marker_path("workspace-1").exists())

    async def test_interrupted_maintenance_is_fail_closed(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            files = data / "_userspace" / "workspaces" / "workspace-1" / "files"
            files.mkdir(parents=True)

            async def request(method, path, **kwargs):
                if method == "POST":
                    return {"authoritative_root": str(files)}
                return {}

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=request),
            ):
                with self.assertRaises(RuntimeError):
                    async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                        raise RuntimeError("publication interrupted")
                with self.assertRaises(HTTPException) as blocked:
                    await sqlite_runtime.assert_sqlite_workspace_available("workspace-1")

            self.assertEqual(blocked.exception.status_code, 423)

    async def test_runtime_unavailable_does_not_fall_back_when_manager_is_enabled(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=HTTPException(status_code=502, detail="unavailable")),
            ):
                with self.assertRaises(HTTPException) as unavailable:
                    async with sqlite_runtime.sqlite_workspace_access("workspace-1"):
                        pass

            self.assertEqual(unavailable.exception.status_code, 502)

    async def test_disabled_runtime_maintenance_writes_and_removes_marker(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            files = data / "_userspace" / "workspaces" / "workspace-1" / "files"
            files.mkdir(parents=True)
            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=False),
            ):
                async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True) as root:
                    self.assertEqual(root, files)
                    self.assertTrue(sqlite_runtime._marker_path("workspace-1").exists())
            self.assertFalse(sqlite_runtime._marker_path("workspace-1").exists())

    async def test_disabled_runtime_ambiguous_rootfs_leaves_fail_closed_marker(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            workspace = data / "_userspace" / "workspaces" / "workspace-1"
            (workspace / "files").mkdir(parents=True)
            mirror = workspace / "rootfs" / "workspace"
            mirror.mkdir(parents=True)
            (mirror / "active.txt").write_text("runtime evidence")
            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=False),
            ):
                with self.assertRaises(HTTPException) as blocked:
                    async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                        pass
                self.assertTrue(sqlite_runtime._marker_path("workspace-1").exists())
            self.assertEqual(blocked.exception.status_code, 503)

    async def test_worker_marker_blocks_admission_after_worker_restart(self) -> None:
        from runtime.worker.service import WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            marker = Path(temporary) / "workspaces" / "workspace-1" / "sqlite_backups" / "sqlite-maintenance-intent.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}")
            worker = WorkerService()
            with self.assertRaises(HTTPException) as blocked:
                worker._ensure_workspace_available_locked("workspace-1")
            self.assertEqual(blocked.exception.status_code, 423)

    async def test_worker_release_is_idempotent_only_without_an_active_lease(self) -> None:
        from runtime.worker.service import WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            worker = WorkerService()
            await worker.release_sqlite_workspace_access("workspace-1", "lease-1")
            worker._workspace_maintenance["workspace-1"] = {"lease-1": (False, mock.Mock())}
            with self.assertRaises(HTTPException) as wrong_lease:
                await worker.release_sqlite_workspace_access("workspace-1", "wrong-lease")
            self.assertEqual(wrong_lease.exception.status_code, 409)

    async def test_worker_allows_two_shared_leases_and_releases_exactly_one(self) -> None:
        from runtime.worker.service import WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            worker = WorkerService()
            await worker.acquire_sqlite_workspace_access("workspace-1", "lease-1", maintenance=False)
            await worker.acquire_sqlite_workspace_access("workspace-1", "lease-2", maintenance=False)
            await worker.release_sqlite_workspace_access("workspace-1", "lease-1")
            self.assertIn("lease-2", worker._workspace_maintenance["workspace-1"])
            with self.assertRaises(HTTPException) as blocked:
                await worker.acquire_sqlite_workspace_access("workspace-1", "maintenance", maintenance=True)
            self.assertEqual(blocked.exception.status_code, 409)
            await worker.release_sqlite_workspace_access("workspace-1", "lease-2")
            self.assertNotIn("workspace-1", worker._workspace_maintenance)

    async def test_worker_shared_lease_allows_pty_but_blocks_public_stop(self) -> None:
        from runtime.worker.service import WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            worker = WorkerService()
            worker._sessions["session-1"] = mock.Mock(id="session-1", workspace_id="workspace-1")
            await worker.acquire_sqlite_workspace_access("workspace-1", "lease-1", maintenance=False)
            await worker.assert_pty_available("session-1")
            with self.assertRaises(HTTPException) as blocked:
                await worker.stop_session("session-1")
            self.assertEqual(blocked.exception.status_code, 423)

    async def test_worker_maintenance_drain_bypass_requires_matching_regular_marker(self) -> None:
        from runtime.worker.service import WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            worker = WorkerService()
            worker._workspace_maintenance["workspace-1"] = {"lease-1": (True, mock.Mock())}
            marker = Path(temporary) / "workspaces" / "workspace-1" / "sqlite_backups" / "sqlite-maintenance-intent.json"
            marker.parent.mkdir(parents=True)
            marker.write_text('{"lease_id":"lease-1"}')
            worker._ensure_workspace_available_locked("workspace-1", require_full_release=True, maintenance_lease_id="lease-1")
            marker.write_text('{"lease_id":"foreign"}')
            with self.assertRaises(HTTPException) as blocked:
                worker._ensure_workspace_available_locked("workspace-1", require_full_release=True, maintenance_lease_id="lease-1")
            self.assertEqual(blocked.exception.status_code, 423)

    async def test_worker_active_chroot_without_mirror_returns_503_and_cleans_failed_lease(self) -> None:
        from runtime.worker import service as worker_service_module
        from runtime.worker.service import WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            worker = WorkerService()
            canonical = Path(temporary) / "workspaces" / "workspace-1" / "files"
            canonical.mkdir(parents=True)
            spec = mock.Mock(workspace_id="workspace-1", workspace_files_path=canonical)
            spec.rootfs_path = canonical.parent / "rootfs"
            spec.sandbox_workspace = "/workspace"
            worker._sessions["session-1"] = mock.Mock(workspace_id="workspace-1", state="running", sandbox_spec=spec)
            with (
                mock.patch.object(worker, "_resolve_workspace_root", return_value=(canonical.parent, canonical, spec)),
                mock.patch.object(worker_service_module, "workspace_mirror_required", return_value=True),
            ):
                with self.assertRaises(HTTPException) as unavailable:
                    await worker.acquire_sqlite_workspace_access("workspace-1", "lease-1", maintenance=False)
            self.assertEqual(unavailable.exception.status_code, 503)
            self.assertNotIn("workspace-1", worker._workspace_maintenance)

    async def test_clean_409_maintenance_acquire_removes_marker(self) -> None:
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=HTTPException(status_code=409, detail="busy")),
            ):
                with self.assertRaises(HTTPException) as busy:
                    async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                        pass
            self.assertEqual(busy.exception.status_code, 409)
            self.assertFalse(sqlite_runtime._marker_path("workspace-1").exists())

    async def test_competing_marker_claims_keep_the_first_owner_marker(self) -> None:
        """A second maintenance caller must not replace a claimed durable intent."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "sqlite_backups" / "sqlite-maintenance-intent.json"
            first: dict[str, object] = {"workspace_id": "workspace-1", "lease_id": "first", "state": "acquiring"}
            second: dict[str, object] = {"workspace_id": "workspace-1", "lease_id": "second", "state": "acquiring"}

            results = await asyncio.gather(
                asyncio.to_thread(sqlite_runtime._claim_marker, marker, first),
                asyncio.to_thread(sqlite_runtime._claim_marker, marker, second),
                return_exceptions=True,
            )

            self.assertEqual(sum(result is None for result in results), 1)
            conflict = next(result for result in results if isinstance(result, HTTPException))
            self.assertEqual(conflict.status_code, 423)
            payload = json.loads(marker.read_text())
            self.assertIn(payload["lease_id"], {"first", "second"})

    async def test_clean_409_preserves_a_foreign_marker(self) -> None:
        """Conflict cleanup may remove only the caller's own durable marker."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)

            async def request(method, path, **kwargs):
                marker = sqlite_runtime._marker_path("workspace-1")
                marker.write_text('{"workspace_id":"workspace-1","lease_id":"foreign","state":"acquiring"}')
                raise HTTPException(status_code=409, detail="busy")

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=request),
            ):
                with self.assertRaises(HTTPException) as busy:
                    async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                        pass
                retained_marker = sqlite_runtime._marker_path("workspace-1")

            self.assertEqual(busy.exception.status_code, 409)
            self.assertEqual("foreign", json.loads(retained_marker.read_text())["lease_id"])

    async def test_owned_marker_can_update_state_and_be_removed(self) -> None:
        """The owning lease alone may advance and clear its durable marker."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "sqlite_backups" / "sqlite-maintenance-intent.json"
            sqlite_runtime._claim_marker(marker, {"workspace_id": "workspace-1", "lease_id": "lease-1", "state": "acquiring"})
            sqlite_runtime._update_owned_marker(marker, "lease-1", {"workspace_id": "workspace-1", "lease_id": "lease-1", "state": "active"})

            self.assertEqual("active", json.loads(marker.read_text())["state"])
            self.assertTrue(sqlite_runtime._remove_owned_marker(marker, "lease-1"))
            self.assertFalse(marker.exists())

    async def test_owned_marker_update_cannot_change_lease_owner(self) -> None:
        """A state transition must retain the lease that proved marker ownership."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "sqlite_backups" / "sqlite-maintenance-intent.json"
            sqlite_runtime._claim_marker(marker, {"workspace_id": "workspace-1", "lease_id": "lease-1", "state": "acquiring"})

            with self.assertRaises(HTTPException) as blocked:
                sqlite_runtime._update_owned_marker(marker, "lease-1", {"workspace_id": "workspace-1", "lease_id": "foreign", "state": "active"})

            self.assertEqual(blocked.exception.status_code, 423)
            self.assertEqual("lease-1", json.loads(marker.read_text())["lease_id"])

    async def test_malformed_and_symlink_markers_fail_closed(self) -> None:
        """Unsafe marker shapes must block maintenance rather than be replaced."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            marker = data / "_userspace" / "workspaces" / "workspace-1" / "sqlite_backups" / "sqlite-maintenance-intent.json"
            (data / "_userspace" / "workspaces" / "workspace-1" / "files").mkdir(parents=True)
            marker.parent.mkdir()

            for unsafe_marker in ("not json",):
                marker.write_text(unsafe_marker)
                with mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)):
                    with self.assertRaises(HTTPException) as blocked:
                        await sqlite_runtime.assert_sqlite_workspace_available("workspace-1")
                self.assertEqual(blocked.exception.status_code, 423)

            marker.unlink()
            target = marker.with_name("foreign-marker.json")
            target.write_text('{"lease_id":"foreign"}')
            marker.symlink_to(target)
            with mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)):
                with self.assertRaises(HTTPException) as blocked:
                    await sqlite_runtime.assert_sqlite_workspace_available("workspace-1")
            self.assertEqual(blocked.exception.status_code, 423)
            self.assertTrue(marker.is_symlink())

    async def test_recovery_requires_the_expected_marker_owner(self) -> None:
        """Recovery must not release or delete a marker belonging to another lease."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            marker = data / "_userspace" / "workspaces" / "workspace-1" / "sqlite_backups" / "sqlite-maintenance-intent.json"
            marker.parent.mkdir(parents=True)
            marker.write_text('{"workspace_id":"workspace-1","lease_id":"foreign","state":"active"}')
            request = mock.AsyncMock()

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", request),
            ):
                with self.assertRaises(HTTPException) as blocked:
                    await sqlite_runtime.recover_sqlite_workspace_maintenance("workspace-1", "lease-1", action="abort")

            self.assertEqual(blocked.exception.status_code, 423)
            self.assertTrue(marker.exists())
            request.assert_not_awaited()

    async def test_workspace_pty_drain_evicts_all_registered_workspace_processes(self) -> None:
        from runtime.worker import api

        first = mock.Mock(returncode=None)
        second = mock.Mock(returncode=None)
        api._pty_processes.clear()
        api._pty_master_fds.clear()
        api._pty_workspace_ids.clear()
        api._pty_processes.update({"one": first, "two": second})
        api._pty_workspace_ids.update({"one": "workspace-1", "two": "workspace-2"})
        with mock.patch.object(api, "_terminate_pty_process", new=mock.AsyncMock()) as terminate:
            await api.evict_workspace_ptys("workspace-1")
        terminate.assert_awaited_once_with(first)
        self.assertNotIn("one", api._pty_processes)
        self.assertIn("two", api._pty_processes)
        api._pty_processes.clear()
        api._pty_master_fds.clear()
        api._pty_workspace_ids.clear()

    async def test_maintenance_drains_pty_then_processes_before_reconcile_and_archives_on_release(self) -> None:
        from runtime.worker import api as worker_api
        from runtime.worker import service as worker_service_module
        from runtime.worker.service import WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            worker = WorkerService()
            canonical = Path(temporary) / "workspaces" / "workspace-1" / "files"
            canonical.mkdir(parents=True)
            spec = mock.Mock(workspace_id="workspace-1", workspace_files_path=canonical)
            events: list[str] = []
            worker._sessions["session-1"] = mock.Mock(id="session-1", workspace_id="workspace-1", state="running")
            caps = mock.Mock(mode="chroot")
            with (
                mock.patch.object(worker, "_resolve_workspace_root", return_value=(canonical.parent, canonical, spec)),
                mock.patch.object(worker_service_module, "detect_capabilities", return_value=caps),
                mock.patch.object(worker_service_module, "workspace_mirror_required", return_value=True),
                mock.patch.object(worker_service_module, "reconcile_stopped_workspace_mirror", side_effect=lambda _spec: events.append("reconcile")),
                mock.patch.object(worker_service_module, "archive_workspace_mirror", side_effect=lambda _spec: events.append("archive")),
                mock.patch.object(worker_api, "evict_workspace_ptys", new=mock.AsyncMock(side_effect=lambda _workspace: events.append("pty"))),
                mock.patch.object(worker, "stop_session", new=mock.AsyncMock(side_effect=lambda _session, **_kwargs: events.append("stop"))),
            ):
                await worker.acquire_sqlite_workspace_access("workspace-1", "lease-1", maintenance=True)
                await worker.release_sqlite_workspace_access("workspace-1", "lease-1")

            self.assertEqual(events, ["pty", "stop", "reconcile", "archive"])

    async def test_assert_maintenance_held_fails_outside_maintenance_context(self) -> None:
        """Assertion should fail when no maintenance context is active."""
        from ragtime.userspace import sqlite_runtime

        with self.assertRaises(HTTPException) as outside:
            await sqlite_runtime.assert_sqlite_workspace_maintenance_held("workspace-1")

        self.assertEqual(outside.exception.status_code, 423)

    async def test_assert_maintenance_held_succeeds_inside_manager_maintenance_context(self) -> None:
        """Assertion should succeed inside an active maintenance context (manager-enabled)."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            files = data / "_userspace" / "workspaces" / "workspace-1" / "files"
            files.mkdir(parents=True)

            async def request(method, path, **kwargs):
                if method == "POST":
                    return {"authoritative_root": str(files)}
                return {}

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=request),
            ):
                async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                    # Inside the context, assertion should succeed
                    await sqlite_runtime.assert_sqlite_workspace_maintenance_held("workspace-1")

    async def test_assert_maintenance_held_succeeds_inside_offline_maintenance_context(self) -> None:
        """Assertion should succeed inside an active maintenance context (offline)."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            files = data / "_userspace" / "workspaces" / "workspace-1" / "files"
            files.mkdir(parents=True)

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=False),
            ):
                async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                    # Inside the context, assertion should succeed
                    await sqlite_runtime.assert_sqlite_workspace_maintenance_held("workspace-1")

    async def test_assert_maintenance_held_fails_outside_context_after_exit(self) -> None:
        """After exiting maintenance context, assertion should fail again."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            files = data / "_userspace" / "workspaces" / "workspace-1" / "files"
            files.mkdir(parents=True)

            async def request(method, path, **kwargs):
                if method == "POST":
                    return {"authoritative_root": str(files)}
                return {}

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=request),
            ):
                async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                    pass

            # After context exit, assertion should fail
            with self.assertRaises(HTTPException) as outside:
                await sqlite_runtime.assert_sqlite_workspace_maintenance_held("workspace-1")

            self.assertEqual(outside.exception.status_code, 423)

    async def test_non_maintenance_access_does_not_mark_held(self) -> None:
        """Non-maintenance read access should not mark workspace as maintenance-held."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            files = data / "_userspace" / "workspaces" / "workspace-1" / "files"
            files.mkdir(parents=True)

            async def request(method, path, **kwargs):
                if method == "POST":
                    return {"authoritative_root": str(files)}
                return {}

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=request),
            ):
                async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=False):
                    # Even inside a non-maintenance context, assertion should fail
                    with self.assertRaises(HTTPException) as outside:
                        await sqlite_runtime.assert_sqlite_workspace_maintenance_held("workspace-1")

                    self.assertEqual(outside.exception.status_code, 423)

    async def test_assert_maintenance_held_context_cleared_on_error(self) -> None:
        """Context should be cleared even if an error occurs during maintenance."""
        from ragtime.userspace import sqlite_runtime

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            files = data / "_userspace" / "workspaces" / "workspace-1" / "files"
            files.mkdir(parents=True)

            async def request(method, path, **kwargs):
                if method == "POST":
                    return {"authoritative_root": str(files)}
                return {}

            with (
                mock.patch.object(sqlite_runtime.settings, "index_data_path", str(data)),
                mock.patch.object(sqlite_runtime, "runtime_manager_enabled", return_value=True),
                mock.patch.object(sqlite_runtime, "runtime_manager_request", side_effect=request),
            ):
                with self.assertRaises(RuntimeError):
                    async with sqlite_runtime.sqlite_workspace_access("workspace-1", maintenance=True):
                        raise RuntimeError("simulated error")

            # After error, context should be cleared
            with self.assertRaises(HTTPException) as outside:
                await sqlite_runtime.assert_sqlite_workspace_maintenance_held("workspace-1")

            self.assertEqual(outside.exception.status_code, 423)

    async def test_worker_shared_lease_stop_leaves_session_unchanged_on_blocked_stop(self) -> None:
        """WorkerService.stop_session must check shared lease before mutating session state."""
        from runtime.worker.service import SandboxSpec, WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            worker = WorkerService()
            workspace_root = Path(temporary) / "workspaces" / "workspace-1"
            workspace_files = workspace_root / "files"
            rootfs_path = workspace_root / "rootfs"
            workspace_files.mkdir(parents=True)
            (rootfs_path / "workspace").mkdir(parents=True)
            session_mock = mock.Mock(
                id="session-1",
                workspace_id="workspace-1",
                state="running",
                runtime_operation_id="op-1",
                sandbox_spec=SandboxSpec(
                    workspace_id="workspace-1",
                    workspace_files_path=workspace_files,
                    rootfs_path=rootfs_path,
                ),
            )
            worker._sessions["session-1"] = session_mock
            startup_task = mock.Mock()
            worker._startup_tasks["session-1"] = startup_task
            worker._devserver_processes["session-1"] = mock.Mock()
            worker._active_execs["session-1"] = {1: mock.Mock()}

            # Acquire a shared (non-maintenance) lease
            await worker.acquire_sqlite_workspace_access("workspace-1", "lease-1", maintenance=False)

            # Attempt to stop the session; should fail due to shared lease
            with self.assertRaises(HTTPException) as blocked:
                await worker.stop_session("session-1")
            self.assertEqual(blocked.exception.status_code, 423)

            # Verify session state and resources are unchanged
            self.assertEqual(session_mock.runtime_operation_id, "op-1")
            self.assertIn("session-1", worker._startup_tasks)
            self.assertIn("session-1", worker._devserver_processes)
            self.assertIn("session-1", worker._active_execs)

    async def test_manager_failed_idempotent_workspace_acquire_preserves_existing_lease(self) -> None:
        from runtime.manager.models import RuntimeWorkspaceMaintenanceRequest
        from runtime.manager.service import SessionManager

        manager = SessionManager()
        manager._workspace_sqlite_maintenance["workspace-1"] = {"lease-1": False}
        manager._worker_service.health = mock.AsyncMock(return_value=mock.Mock(metadata={"runtime_capabilities": {"sqlite_workspace_maintenance": True}}))
        manager._worker_service.acquire_sqlite_workspace_access = mock.AsyncMock(side_effect=RuntimeError("simulated failure"))

        with self.assertRaises(RuntimeError):
            await manager.acquire_sqlite_workspace_maintenance(
                "workspace-1",
                RuntimeWorkspaceMaintenanceRequest(lease_id="lease-1", maintenance=False),
            )

        self.assertEqual({"lease-1": False}, manager._workspace_sqlite_maintenance["workspace-1"])

    async def test_worker_shared_lease_idempotent_acquire_cleanup_on_failure_only_if_new(self) -> None:
        """Worker acquire_sqlite_workspace_access must only cleanup leases it newly created."""
        from runtime.worker.service import WorkerService

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RUNTIME_WORKSPACE_ROOT": temporary}, clear=False):
            worker = WorkerService()
            canonical = Path(temporary) / "workspaces" / "workspace-1" / "files"
            canonical.mkdir(parents=True)
            spec = mock.Mock(workspace_id="workspace-1", workspace_files_path=canonical)

            # First acquire a shared lease successfully
            with mock.patch.object(worker, "_resolve_workspace_root", return_value=(canonical.parent, canonical, spec)):
                await worker.acquire_sqlite_workspace_access("workspace-1", "lease-1", maintenance=False)

            self.assertIn("lease-1", worker._workspace_maintenance.get("workspace-1", {}))

            # Now attempt idempotent acquire of the same lease but force it to fail in the try block
            with (
                mock.patch.object(worker, "_resolve_workspace_root", return_value=(canonical.parent, canonical, spec)),
                mock.patch.object(worker, "_workspace_startup_lock", side_effect=RuntimeError("simulated failure")),
            ):
                with self.assertRaises(RuntimeError):
                    await worker.acquire_sqlite_workspace_access("workspace-1", "lease-1", maintenance=False)

            # Verify the lease was NOT cleaned up (it pre-existed, so shouldn't be removed)
            self.assertIn("lease-1", worker._workspace_maintenance.get("workspace-1", {}))
