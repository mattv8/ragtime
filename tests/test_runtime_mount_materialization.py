import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from runtime.worker import sandbox
from runtime.worker.service import WorkerService


class MountMaterializationTests(unittest.TestCase):
    def _case(self):
        tempdir = tempfile.TemporaryDirectory()
        root = Path(tempdir.name)
        source = root / "source"
        source.mkdir()
        rootfs = root / "rootfs"
        rootfs.mkdir()
        files = root / "files"
        files.mkdir()
        return tempdir, source, rootfs, sandbox.SandboxSpec("workspace", files, rootfs)

    def _no_mount_caps(self):
        return sandbox.SandboxCapabilities(mode="chroot", can_mount=False)

    def test_incremental_sync_keeps_target_and_passes_minimal_nested_protection(self):
        tempdir, source, rootfs, spec = self._case()
        self.addCleanup(tempdir.cleanup)
        (source / "current.txt").write_text("new", encoding="utf-8")
        target = rootfs / "workspace" / "mount"
        target.mkdir(parents=True)
        old = target / "unchanged.txt"
        old.write_text("old", encoding="utf-8")
        old_inode = old.stat().st_ino

        protected_calls = []

        def sync(source_fd, destination_fd, **kwargs):
            self.assertEqual(old.stat().st_ino, old_inode)
            protected_calls.append(kwargs["protected_paths"])

        mounts = [
            {"source_local_path": str(source), "target_path": "/workspace/mount"},
            {"source_local_path": str(source), "target_path": "/workspace/mount/nested"},
            {"source_local_path": str(source), "target_path": "/workspace/mount/nested/deeper"},
        ]
        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=self._no_mount_caps()),
            mock.patch.object(sandbox, "mount_sync_available", return_value=True),
            mock.patch.object(sandbox, "sync_copied_mount", side_effect=sync),
        ):
            sandbox.materialize_mounts(spec, mounts)
        self.assertEqual(old.stat().st_ino, old_inode)
        self.assertEqual(protected_calls[0], ("nested",))

    def test_unavailable_helper_uses_copy_fallback_without_executing_sync(self):
        tempdir, source, rootfs, spec = self._case()
        self.addCleanup(tempdir.cleanup)
        (source / "fresh.txt").write_text("fresh", encoding="utf-8")
        stale = rootfs / "workspace" / "mount" / "stale.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("stale", encoding="utf-8")
        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=self._no_mount_caps()),
            mock.patch.object(sandbox, "mount_sync_available", return_value=False),
            mock.patch.object(sandbox, "sync_copied_mount") as sync,
        ):
            sandbox.materialize_mounts(spec, [{"source_local_path": str(source), "target_path": "/workspace/mount"}])
        sync.assert_not_called()
        self.assertFalse(stale.exists())
        self.assertEqual((rootfs / "workspace" / "mount" / "fresh.txt").read_text(encoding="utf-8"), "fresh")

    def test_sync_failure_propagates_instead_of_becoming_copy_warning(self):
        tempdir, source, _rootfs, spec = self._case()
        self.addCleanup(tempdir.cleanup)
        failure = RuntimeError("rsync failed")
        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=self._no_mount_caps()),
            mock.patch.object(sandbox, "mount_sync_available", return_value=True),
            mock.patch.object(sandbox, "sync_copied_mount", side_effect=failure),
        ):
            with self.assertRaisesRegex(RuntimeError, "rsync failed"):
                sandbox.materialize_mounts(spec, [{"source_local_path": str(source), "target_path": "/workspace/mount"}])

    def test_replaced_source_generation_after_pin_fails_without_success(self):
        tempdir, source, rootfs, spec = self._case()
        self.addCleanup(tempdir.cleanup)
        destination = rootfs / "workspace" / "mount"
        destination.mkdir(parents=True)
        (destination / "preserved.txt").write_text("preserved", encoding="utf-8")

        def replace_source(*_args, **_kwargs):
            retired = source.parent / "retired"
            source.rename(retired)
            source.mkdir()

        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=self._no_mount_caps()),
            mock.patch.object(sandbox, "mount_sync_available", return_value=True),
            mock.patch.object(sandbox, "sync_copied_mount", side_effect=replace_source),
        ):
            with self.assertRaises(sandbox.MountSyncError):
                sandbox.materialize_mounts(spec, [{"source_local_path": str(source), "target_path": "/workspace/mount"}])
        self.assertEqual((destination / "preserved.txt").read_text(encoding="utf-8"), "preserved")

    def test_replaced_source_generation_does_not_trigger_unavailable_fallback(self):
        tempdir, source, rootfs, spec = self._case()
        self.addCleanup(tempdir.cleanup)
        destination = rootfs / "workspace" / "mount"
        destination.mkdir(parents=True)
        (destination / "preserved.txt").write_text("preserved", encoding="utf-8")

        def replace_then_unavailable(*_args, **_kwargs):
            source.rename(source.parent / "retired")
            source.mkdir()
            raise sandbox.MountSyncUnavailable()

        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=self._no_mount_caps()),
            mock.patch.object(sandbox, "mount_sync_available", return_value=True),
            mock.patch.object(sandbox, "sync_copied_mount", side_effect=replace_then_unavailable),
        ):
            with self.assertRaises(sandbox.MountSyncError):
                sandbox.materialize_mounts(spec, [{"source_local_path": str(source), "target_path": "/workspace/mount"}])
        self.assertEqual((destination / "preserved.txt").read_text(encoding="utf-8"), "preserved")

    def test_fallback_orders_parent_before_child_when_input_is_reversed(self):
        tempdir, parent_source, rootfs, spec = self._case()
        self.addCleanup(tempdir.cleanup)
        child_source = parent_source.parent / "child-source"
        child_source.mkdir()
        (parent_source / "child").mkdir()
        (parent_source / "child" / "parent.txt").write_text("parent", encoding="utf-8")
        (child_source / "child.txt").write_text("child", encoding="utf-8")
        mounts = [
            {"source_local_path": str(child_source), "target_path": "/workspace/mount/child"},
            {"source_local_path": str(parent_source), "target_path": "/workspace/mount"},
        ]
        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=self._no_mount_caps()),
            mock.patch.object(sandbox, "mount_sync_available", return_value=False),
        ):
            sandbox.materialize_mounts(spec, mounts)
        child_target = rootfs / "workspace" / "mount" / "child"
        self.assertEqual((child_target / "child.txt").read_text(encoding="utf-8"), "child")
        self.assertFalse((child_target / "parent.txt").exists())

    def test_sync_replaces_file_or_symlink_target_with_real_directory(self):
        for make_target in (
            lambda target: target.write_text("old", encoding="utf-8"),
            lambda target: target.symlink_to("elsewhere"),
        ):
            tempdir, source, rootfs, spec = self._case()
            self.addCleanup(tempdir.cleanup)
            target = rootfs / "workspace" / "mount"
            target.parent.mkdir(parents=True)
            make_target(target)
            with (
                mock.patch.object(sandbox, "detect_capabilities", return_value=self._no_mount_caps()),
                mock.patch.object(sandbox, "mount_sync_available", return_value=True),
                mock.patch.object(sandbox, "sync_copied_mount"),
            ):
                sandbox.materialize_mounts(spec, [{"source_local_path": str(source), "target_path": "/workspace/mount"}])
            self.assertTrue(target.is_dir())

    def test_live_bind_does_not_call_sync(self):
        tempdir, source, _rootfs, spec = self._case()
        self.addCleanup(tempdir.cleanup)
        caps = sandbox.SandboxCapabilities(mode="pivot_root", can_mount=True)
        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
            mock.patch.object(sandbox, "_syscall_mount"),
            mock.patch.object(sandbox, "mount_sync_available") as available,
            mock.patch.object(sandbox, "sync_copied_mount") as sync,
        ):
            sandbox.materialize_mounts(spec, [{"source_local_path": str(source), "target_path": "/workspace/mount", "runtime_mount_mode": "live_bind"}])
        available.assert_not_called()
        sync.assert_not_called()

    def test_live_bind_without_mount_authority_does_not_probe_sync(self):
        tempdir, source, _rootfs, spec = self._case()
        self.addCleanup(tempdir.cleanup)
        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=self._no_mount_caps()),
            mock.patch.object(sandbox, "mount_sync_available") as available,
        ):
            with self.assertRaises(PermissionError):
                sandbox.materialize_mounts(
                    spec,
                    [
                        {
                            "source_local_path": str(source),
                            "target_path": "/workspace/mount",
                            "runtime_mount_mode": "live_bind",
                        }
                    ],
                )
        available.assert_not_called()


class MountMaterializationCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_drains_materialization_thread_before_returning(self):
        service = WorkerService()
        entered = threading.Event()
        release = threading.Event()
        drained = threading.Event()
        session = mock.Mock(workspace_mounts=[{"target_path": "/workspace/mount"}])
        session.mount_targets_to_clear = set()
        session.sandbox_spec = mock.sentinel.spec

        def blocking_materialize(*_args, cancel_event, **_kwargs):
            entered.set()
            self.assertTrue(cancel_event.wait(2))
            self.assertTrue(release.wait(2))
            drained.set()

        with mock.patch("runtime.worker.service.materialize_mounts", side_effect=blocking_materialize):
            task = asyncio.create_task(service._materialize_workspace_mounts(session))
            await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(drained.is_set())
