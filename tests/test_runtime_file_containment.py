from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi import HTTPException

worker_service: Any = importlib.import_module("runtime.worker.service")


class RuntimeFileContainmentTests(unittest.IsolatedAsyncioTestCase):
    def _install(self, service: Any, root: Path, mounts: list[dict[str, Any]] | None = None) -> Any:
        files = root / "files"
        files.mkdir()
        session = worker_service.WorkerSession(
            id="session",
            workspace_id="workspace",
            provider_session_id="provider",
            workspace_root=root,
            workspace_files_path=files,
            sandbox_spec=worker_service.SandboxSpec(workspace_id="workspace", workspace_files_path=files, rootfs_path=root / "rootfs"),
            pty_access_token="token",
            workspace_env={},
            workspace_env_visibility={},
            workspace_mounts=mounts or [],
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
        service._sessions[session.id] = session
        return session

    async def test_symlinked_parent_and_leaf_cannot_escape_writes_or_reads(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = self._install(service, root)
            outside = root / "outside"
            outside.mkdir()
            secret = outside / "secret.txt"
            secret.write_text("secret")
            (session.workspace_files_path / "escape").symlink_to(outside, target_is_directory=True)
            (session.workspace_files_path / "leaf").symlink_to(secret)

            self.assertFalse((await service.read_file(session.id, "escape/secret.txt")).exists)
            self.assertFalse((await service.read_file(session.id, "leaf")).exists)
            with self.assertRaises(HTTPException) as parent_error:
                await service.write_file(session.id, "escape/new.txt", "bad")
            self.assertEqual(parent_error.exception.status_code, 403)
            with self.assertRaises(HTTPException):
                await service.write_file(session.id, "leaf", "bad")
            self.assertEqual(secret.read_text(), "secret")

    async def test_readonly_matching_mount_cannot_fall_back_after_escape(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mount = root / "mount"
            outside = root / "outside"
            mount.mkdir()
            outside.mkdir()
            (mount / "escape").symlink_to(outside, target_is_directory=True)
            session = self._install(service, root, [{"target_path": "/workspace/data", "source_local_path": str(mount), "read_only": True}])

            with self.assertRaises(HTTPException) as error:
                await service.write_file(session.id, "data/escape/new.txt", "bad")
            self.assertEqual(error.exception.status_code, 403)
            self.assertFalse((session.workspace_files_path / "data" / "escape" / "new.txt").exists())
            self.assertFalse((outside / "new.txt").exists())

    async def test_regular_mounted_write_missing_delete_and_leaf_link_delete(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mount = root / "mount"
            mount.mkdir()
            session = self._install(service, root, [{"target_path": "/workspace/data", "source_local_path": str(mount), "read_only": False}])

            response = await service.write_file(session.id, "data/nested/file.txt", "ok")
            self.assertTrue(response.exists)
            self.assertEqual((mount / "nested" / "file.txt").read_text(), "ok")
            self.assertTrue((await service.delete_file(session.id, "data/missing.txt"))["success"])
            target = mount / "target.txt"
            target.write_text("keep")
            (mount / "link.txt").symlink_to(target)
            await service.delete_file(session.id, "data/link.txt")
            self.assertTrue(target.exists())
            self.assertFalse((mount / "link.txt").exists())

    async def test_fifo_is_never_opened_blocking_or_truncated(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp))
            fifo = session.workspace_files_path / "pipe"
            os.mkfifo(fifo)

            response = await asyncio.wait_for(service.read_file(session.id, "pipe"), timeout=0.5)
            self.assertFalse(response.exists)
            with self.assertRaises(HTTPException) as error:
                await asyncio.wait_for(service.write_file(session.id, "pipe", "bad"), timeout=0.5)
            self.assertEqual(error.exception.status_code, 403)
            self.assertTrue(fifo.exists())

    async def test_nul_path_is_rejected_before_filesystem_io(self) -> None:
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmp:
            session = self._install(service, Path(tmp))
            with self.assertRaises(HTTPException) as error:
                await service.write_file(session.id, "bad\x00name", "bad")
            self.assertEqual(error.exception.status_code, 400)
