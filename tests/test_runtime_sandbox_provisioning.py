import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from runtime.core.shared import RUNTIME_BOOTSTRAP_CONFIG_PATH
from runtime.worker import sandbox
from runtime.worker.service import WorkerService


class SandboxProvisioningTests(unittest.TestCase):
    def tearDown(self) -> None:
        sandbox._capabilities_cache.clear()

    def test_provision_rootfs_skips_workspace_copy_when_bind_mount_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            files.mkdir()
            (files / "app.py").write_text("print('hello')\n", encoding="utf-8")
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="pivot_root",
            )
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=True,
                can_pivot_root=True,
                can_mount=True,
                mode="pivot_root",
            )

            with (
                mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
                mock.patch.object(
                    sandbox,
                    "_provision_etc",
                ),
                mock.patch.object(sandbox, "_provision_dev"),
                mock.patch.object(sandbox.shutil, "copytree") as copytree,
            ):
                sandbox.provision_rootfs(spec)

            self.assertTrue((rootfs / "workspace").is_dir())
            copytree.assert_not_called()

    def test_provision_rootfs_reconciles_legacy_workspace_copy_before_bind_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            legacy_workspace = rootfs / "workspace"
            files.mkdir()
            (legacy_workspace / ".ragtime" / "db" / "app.sqlite3").parent.mkdir(parents=True)
            (legacy_workspace / ".ragtime" / "db" / "app.sqlite3").write_bytes(b"sqlite-data")

            preserved_paths = [
                "dashboard/main.ts",
                "node_modules/left-pad/index.js",
                ".venv/lib/python3.12/site-packages/flask/__init__.py",
                "vendor/package/index.js",
                "dist/main.js",
                "build/asset.txt",
                ".next/server/app.js",
                ".nuxt/dist/server.js",
            ]
            skipped_paths = [
                ".git/objects/sentinel",
                ".pytest_cache/sentinel",
            ]
            for relative_path in preserved_paths + skipped_paths:
                path = legacy_workspace / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{relative_path}\n", encoding="utf-8")
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="pivot_root",
            )
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=True,
                can_pivot_root=True,
                can_mount=True,
                mode="pivot_root",
            )

            with (
                mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
                mock.patch.object(sandbox, "_provision_etc"),
                mock.patch.object(sandbox, "_provision_dev"),
                mock.patch.object(sandbox.shutil, "copytree") as copytree,
            ):
                sandbox.provision_rootfs(spec)

            self.assertEqual((files / ".ragtime" / "db" / "app.sqlite3").read_bytes(), b"sqlite-data")
            for relative_path in preserved_paths:
                self.assertEqual((files / relative_path).read_text(encoding="utf-8"), f"{relative_path}\n")
            for relative_path in skipped_paths:
                self.assertFalse((files / relative_path).exists())
            archives = list((tmp / sandbox._WORKSPACE_LEGACY_RECOVERY_DIR).glob("chroot-workspace-*"))
            self.assertEqual(len(archives), 1)
            self.assertTrue((archives[0] / "node_modules" / "left-pad" / "index.js").exists())
            self.assertEqual(list((rootfs / "workspace").iterdir()), [])
            copytree.assert_not_called()

    def test_provision_rootfs_mirrors_workspace_for_no_mount_chroot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            files.mkdir()
            (files / "app.py").write_text("print('hello')\n", encoding="utf-8")
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="chroot",
            )
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=False,
                can_user_ns=False,
                can_mount=False,
                mode="chroot",
            )

            with (
                mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
                mock.patch.object(
                    sandbox,
                    "_provision_etc",
                ),
                mock.patch.object(sandbox, "_provision_dev"),
                mock.patch.object(sandbox.shutil, "copytree") as copytree,
            ):
                sandbox.provision_rootfs(spec)

            copytree.assert_called_once()
            self.assertEqual(
                copytree.call_args.args[:2],
                (str(files), str(rootfs / "workspace")),
            )

    def test_cleanup_sandbox_reconciles_no_mount_chroot_workspace_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            legacy_workspace = rootfs / "workspace"
            files.mkdir()
            legacy_workspace.mkdir(parents=True)
            canonical_file = files / "dashboard" / "main.ts"
            canonical_file.parent.mkdir()
            canonical_file.write_text("old\n", encoding="utf-8")
            legacy_file = legacy_workspace / "dashboard" / "main.ts"
            legacy_file.parent.mkdir()
            legacy_file.write_text("new\n", encoding="utf-8")
            os.utime(canonical_file, (1000, 1000))
            os.utime(legacy_file, (2000, 2000))
            (legacy_workspace / ".ragtime" / "db").mkdir(parents=True)
            (legacy_workspace / ".ragtime" / "db" / "app.sqlite3").write_bytes(b"sqlite-data")
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="chroot",
            )
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=False,
                can_user_ns=False,
                can_mount=False,
                mode="chroot",
            )

            with mock.patch.object(sandbox, "detect_capabilities", return_value=caps):
                sandbox.cleanup_sandbox(spec)

            self.assertEqual(canonical_file.read_text(encoding="utf-8"), "new\n")
            self.assertEqual((files / ".ragtime" / "db" / "app.sqlite3").read_bytes(), b"sqlite-data")
            archives = list((tmp / sandbox._WORKSPACE_LEGACY_RECOVERY_DIR).glob("chroot-workspace-cleanup-*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(list((rootfs / "workspace").iterdir()), [])

    def test_terminate_sandbox_cgroup_processes_sends_sigterm_to_lingering_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cgroup_parent = tmp / "cgroups"
            cgroup_path = cgroup_parent / "workspace-1"
            cgroup_path.mkdir(parents=True)
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=tmp / "files",
                rootfs_path=tmp / "rootfs",
                mode="pivot_root",
            )
            caps = sandbox.SandboxCapabilities(
                cgroup_pids_available=True,
                cgroup_pids_parent=str(cgroup_parent),
                cgroup_pids_max=1024,
                mode="pivot_root",
            )

            with (
                mock.patch.object(
                    sandbox,
                    "_read_cgroup_process_ids",
                    side_effect=[[123, 456], []],
                ),
                mock.patch.object(sandbox.os, "kill") as kill_process,
            ):
                sandbox._terminate_sandbox_cgroup_processes(spec, caps)

            self.assertEqual(
                kill_process.mock_calls,
                [
                    mock.call(123, sandbox.signal.SIGTERM),
                    mock.call(456, sandbox.signal.SIGTERM),
                ],
            )

    def test_list_workspace_files_uses_canonical_files_for_active_pivot_root_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workspace_root = tmp / "workspaces" / "workspace-1"
            files = workspace_root / "files"
            rootfs = workspace_root / "rootfs"
            files.mkdir(parents=True)
            (files / "dashboard").mkdir()
            (files / "dashboard" / "main.ts").write_text("export const value = 1;\n", encoding="utf-8")
            (rootfs / "workspace").mkdir(parents=True)
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="pivot_root",
            )
            service = WorkerService()
            service._root = tmp
            service._sessions["session-1"] = self._worker_session("session-1", spec, files, rootfs)
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=True,
                can_pivot_root=True,
                can_mount=True,
                mode="pivot_root",
            )

            with mock.patch("runtime.worker.service.detect_capabilities", return_value=caps):
                response = asyncio.run(service.list_workspace_files("workspace-1", include_dirs=True))

            paths = {entry.path for entry in response.files}
            self.assertIn("dashboard", paths)
            self.assertIn("dashboard/main.ts", paths)

    def test_list_workspace_files_uses_active_chroot_mirror_without_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workspace_root = tmp / "workspaces" / "workspace-1"
            files = workspace_root / "files"
            rootfs = workspace_root / "rootfs"
            files.mkdir(parents=True)
            (files / "canonical.txt").write_text("canonical\n", encoding="utf-8")
            rootfs_workspace = rootfs / "workspace"
            rootfs_workspace.mkdir(parents=True)
            (rootfs_workspace / "active-chroot.txt").write_text("active\n", encoding="utf-8")
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="chroot",
            )
            service = WorkerService()
            service._root = tmp
            service._sessions["session-1"] = self._worker_session("session-1", spec, files, rootfs)
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=False,
                can_user_ns=False,
                can_mount=False,
                mode="chroot",
            )

            with mock.patch("runtime.worker.service.detect_capabilities", return_value=caps):
                response = asyncio.run(service.list_workspace_files("workspace-1", include_dirs=True))

            paths = {entry.path for entry in response.files}
            self.assertIn("active-chroot.txt", paths)
            self.assertNotIn("canonical.txt", paths)

    def test_runtime_file_read_uses_canonical_files_for_active_pivot_root_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workspace_root = tmp / "workspaces" / "workspace-1"
            files = workspace_root / "files"
            rootfs = workspace_root / "rootfs"
            files.mkdir(parents=True)
            (files / "note.txt").write_text("canonical\n", encoding="utf-8")
            rootfs_workspace = rootfs / "workspace"
            rootfs_workspace.mkdir(parents=True)
            (rootfs_workspace / "note.txt").write_text("stale-rootfs\n", encoding="utf-8")
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="pivot_root",
            )
            service = WorkerService()
            service._root = tmp
            service._sessions["session-1"] = self._worker_session("session-1", spec, files, rootfs)
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=True,
                can_pivot_root=True,
                can_mount=True,
                mode="pivot_root",
            )

            with mock.patch("runtime.worker.service.detect_capabilities", return_value=caps):
                response = asyncio.run(service.read_file("session-1", "note.txt"))

            self.assertTrue(response.exists)
            self.assertEqual(response.content, "canonical\n")

    def test_runtime_file_write_uses_active_chroot_mirror_without_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workspace_root = tmp / "workspaces" / "workspace-1"
            files = workspace_root / "files"
            rootfs = workspace_root / "rootfs"
            files.mkdir(parents=True)
            rootfs_workspace = rootfs / "workspace"
            rootfs_workspace.mkdir(parents=True)
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="chroot",
            )
            service = WorkerService()
            service._root = tmp
            service._sessions["session-1"] = self._worker_session("session-1", spec, files, rootfs)
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=False,
                can_user_ns=False,
                can_mount=False,
                mode="chroot",
            )

            with mock.patch("runtime.worker.service.detect_capabilities", return_value=caps):
                response = asyncio.run(service.write_file("session-1", "note.txt", "active\n"))

            self.assertTrue(response.exists)
            self.assertEqual((rootfs_workspace / "note.txt").read_text(encoding="utf-8"), "active\n")
            self.assertFalse((files / "note.txt").exists())

    def test_ensure_sandbox_ready_syncs_chroot_system_dirs_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=tmp / "files",
                rootfs_path=tmp / "rootfs",
                mode="chroot",
            )
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=False,
                can_user_ns=False,
                can_mount=False,
                mode="chroot",
            )

            with (
                mock.patch.object(sandbox, "provision_rootfs") as provision,
                mock.patch.object(
                    sandbox,
                    "detect_capabilities",
                    return_value=caps,
                ),
                mock.patch.object(sandbox, "_sync_system_dirs_for_chroot") as sync,
            ):
                sandbox.ensure_sandbox_ready(spec)

            provision.assert_called_once_with(spec)
            sync.assert_called_once_with(spec)

    def test_pivot_root_unshare_flags_skip_user_namespace_with_sys_admin(self) -> None:
        caps = sandbox.SandboxCapabilities(
            has_cap_sys_admin=True,
            can_pivot_root=True,
            can_user_ns=True,
            can_mount=True,
            mode="pivot_root",
        )

        flags = sandbox._pivot_root_unshare_flags(caps)

        self.assertTrue(flags & sandbox.CLONE_NEWNS)
        self.assertTrue(flags & sandbox.CLONE_NEWPID)
        self.assertFalse(flags & sandbox.CLONE_NEWUSER)

    def test_pivot_root_unshare_flags_include_user_namespace_without_sys_admin(self) -> None:
        caps = sandbox.SandboxCapabilities(
            has_cap_sys_admin=False,
            can_pivot_root=False,
            can_user_ns=True,
            can_mount=False,
            mode="pivot_root",
        )

        flags = sandbox._pivot_root_unshare_flags(caps)

        self.assertTrue(flags & sandbox.CLONE_NEWUSER)

    def test_detect_capabilities_degrades_to_chroot_when_pid_namespace_rejected(self) -> None:
        def can_unshare(flags: int) -> bool:
            if flags == sandbox.CLONE_NEWUSER:
                return False
            if flags & sandbox.CLONE_NEWPID:
                return False
            return True

        with (
            mock.patch.object(sandbox, "has_cap_sys_admin", return_value=True),
            mock.patch.object(sandbox.os, "geteuid", return_value=0),
            mock.patch.object(sandbox, "_can_unshare_flags", side_effect=can_unshare),
            mock.patch.object(sandbox, "_detect_cgroup_pids_limit", return_value=(False, None, None)),
        ):
            caps = sandbox.detect_capabilities()

        self.assertEqual(caps.mode, "chroot")
        self.assertTrue(caps.can_mount)
        self.assertFalse(caps.can_pivot_root)
        self.assertFalse(caps.pid_namespace)
        self.assertTrue(caps.mount_namespace)
        self.assertTrue(caps.dropped_unshare_flags & sandbox.CLONE_NEWPID)

    def test_calculate_sandbox_pids_max_scales_up_on_resource_rich_host(self) -> None:
        pids_max = sandbox._calculate_sandbox_pids_max(
            total_pids_limit=77064,
            memory_limit_bytes=64 * 1024 * 1024 * 1024,
            cpu_count=16,
            max_sessions=12,
        )

        self.assertEqual(pids_max, 1024)

    def test_calculate_sandbox_pids_max_scales_down_on_small_host(self) -> None:
        pids_max = sandbox._calculate_sandbox_pids_max(
            total_pids_limit=None,
            memory_limit_bytes=2 * 1024 * 1024 * 1024,
            cpu_count=2,
            max_sessions=12,
        )

        self.assertEqual(pids_max, 256)

    def test_calculate_sandbox_pids_max_respects_constrained_pids_budget(self) -> None:
        pids_max = sandbox._calculate_sandbox_pids_max(
            total_pids_limit=1000,
            memory_limit_bytes=64 * 1024 * 1024 * 1024,
            cpu_count=16,
            max_sessions=12,
        )

        self.assertEqual(pids_max, 72)

    def test_recommended_startup_concurrency_stays_single_without_pid_namespace(self) -> None:
        caps = sandbox.SandboxCapabilities(pid_namespace=False, mode="chroot")

        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
            mock.patch.object(sandbox.os, "cpu_count", return_value=16),
            mock.patch.object(sandbox, "_detect_memory_limit_bytes", return_value=16 * 1024 * 1024 * 1024),
            mock.patch.dict(os.environ, {"RUNTIME_MAX_SESSIONS": "12"}, clear=False),
        ):
            concurrency = sandbox.recommended_startup_concurrency()

        self.assertEqual(concurrency, 1)

    def test_recommended_startup_concurrency_scales_with_isolated_resources(self) -> None:
        caps = sandbox.SandboxCapabilities(pid_namespace=True, mode="pivot_root")

        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
            mock.patch.object(sandbox.os, "cpu_count", return_value=16),
            mock.patch.object(sandbox, "_detect_memory_limit_bytes", return_value=16 * 1024 * 1024 * 1024),
            mock.patch.dict(os.environ, {"RUNTIME_MAX_SESSIONS": "12"}, clear=False),
        ):
            concurrency = sandbox.recommended_startup_concurrency()

        self.assertEqual(concurrency, 4)

    def test_recommended_startup_concurrency_respects_session_capacity(self) -> None:
        caps = sandbox.SandboxCapabilities(pid_namespace=True, mode="pivot_root")

        with (
            mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
            mock.patch.object(sandbox.os, "cpu_count", return_value=16),
            mock.patch.object(sandbox, "_detect_memory_limit_bytes", return_value=16 * 1024 * 1024 * 1024),
            mock.patch.dict(os.environ, {"RUNTIME_MAX_SESSIONS": "2"}, clear=False),
        ):
            concurrency = sandbox.recommended_startup_concurrency()

        self.assertEqual(concurrency, 2)

    def test_setup_sandbox_mounts_skips_proc_without_pid_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            files.mkdir()
            for relative in ("bin", "usr", "lib", "sbin", "workspace", "dev", "dev/pts", "dev/shm", "proc"):
                (rootfs / relative).mkdir(parents=True, exist_ok=True)
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="chroot",
            )

            with mock.patch.object(sandbox, "_syscall_mount") as mount_call:
                sandbox._setup_sandbox_mounts(spec, mount_proc=False)

        self.assertFalse(any(call.args[2] == "proc" for call in mount_call.call_args_list))

    def test_setup_sandbox_mounts_mounts_proc_with_pid_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            files.mkdir()
            for relative in ("bin", "usr", "lib", "sbin", "workspace", "dev", "dev/pts", "dev/shm", "proc"):
                (rootfs / relative).mkdir(parents=True, exist_ok=True)
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="pivot_root",
            )

            with mock.patch.object(sandbox, "_syscall_mount") as mount_call:
                sandbox._setup_sandbox_mounts(spec, mount_proc=True)

        self.assertTrue(any(call.args[2] == "proc" for call in mount_call.call_args_list))

    def test_materialize_mounts_live_binds_filesystem_source_when_mount_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source"
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            source.mkdir()
            files.mkdir()
            (source / "ledger.txt").write_text("live\n", encoding="utf-8")
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="pivot_root",
            )
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=True,
                can_pivot_root=True,
                can_mount=True,
                mode="pivot_root",
            )

            with (
                mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
                mock.patch.object(sandbox, "_syscall_mount") as mount_call,
                mock.patch.object(sandbox.shutil, "copytree") as copytree,
            ):
                sandbox.materialize_mounts(
                    spec,
                    [
                        {
                            "source_local_path": str(source),
                            "target_path": "/workspace/reconciliations",
                            "runtime_mount_mode": "live_bind",
                            "read_only": True,
                        }
                    ],
                )

            copytree.assert_not_called()
            self.assertTrue((files / "reconciliations").is_dir())
            self.assertEqual(mount_call.call_args_list[0].args[:2], (str(source), str(files / "reconciliations")))

    def test_materialize_mounts_uses_canonical_files_for_mount_capable_sync_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source"
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            source.mkdir()
            files.mkdir()
            (source / "ledger.txt").write_text("synced\n", encoding="utf-8")
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="pivot_root",
            )
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=True,
                can_pivot_root=True,
                can_mount=True,
                mode="pivot_root",
            )

            with (
                mock.patch.object(sandbox, "detect_capabilities", return_value=caps),
                mock.patch.object(sandbox, "_syscall_mount") as mount_call,
                mock.patch.object(sandbox.shutil, "copytree") as copytree,
            ):
                sandbox.materialize_mounts(
                    spec,
                    [
                        {
                            "source_local_path": str(source),
                            "target_path": "/workspace/reconciliations",
                            "read_only": True,
                        }
                    ],
                )

            copytree.assert_not_called()
            self.assertEqual(mount_call.call_args_list[0].args[:2], (str(source), str(files / "reconciliations")))
            self.assertFalse((rootfs / "workspace" / "reconciliations").exists())

    def test_materialize_mounts_live_bind_missing_source_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            files.mkdir()
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="chroot",
            )

            with self.assertRaises(FileNotFoundError):
                sandbox.materialize_mounts(
                    spec,
                    [
                        {
                            "source_local_path": str(tmp / "missing"),
                            "target_path": "/workspace/reconciliations",
                            "runtime_mount_mode": "live_bind",
                            "read_only": True,
                        }
                    ],
                )

    def test_materialize_mounts_live_bind_requires_mount_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source"
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            source.mkdir()
            files.mkdir()
            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="chroot",
            )
            caps = sandbox.SandboxCapabilities(
                has_cap_sys_admin=False,
                can_user_ns=False,
                can_mount=False,
                mode="chroot",
            )

            with mock.patch.object(sandbox, "detect_capabilities", return_value=caps):
                with self.assertRaises(PermissionError):
                    sandbox.materialize_mounts(
                        spec,
                        [
                            {
                                "source_local_path": str(source),
                                "target_path": "/workspace/reconciliations",
                                "runtime_mount_mode": "live_bind",
                                "read_only": True,
                            }
                        ],
                    )

    def test_runtime_mount_file_resolution_prefers_deepest_mount_prefix(self) -> None:
        resolved = WorkerService._resolve_workspace_mount_file_path(
            [
                {
                    "source_local_path": "/mnt/accounting",
                    "target_path": "/workspace/reconciliations",
                    "read_only": True,
                },
                {
                    "source_local_path": "/mnt/accounting/special",
                    "target_path": "/workspace/reconciliations/special",
                    "read_only": False,
                },
            ],
            "reconciliations/special/may.xlsx",
        )

        self.assertEqual(resolved, (Path("/mnt/accounting/special/may.xlsx"), False))

    def test_bootstrap_watch_sync_removes_stale_sandbox_lockfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            files = tmp / "files"
            rootfs = tmp / "rootfs"
            rootfs_workspace = rootfs / "workspace"
            files.mkdir()
            rootfs_workspace.mkdir(parents=True)
            (files / "package.json").write_text('{"devDependencies":{"esbuild":"^0.25.0"}}\n', encoding="utf-8")
            (rootfs_workspace / "package.json").write_text('{"devDependencies":{"esbuild":"^0.25.0"}}\n', encoding="utf-8")
            (rootfs_workspace / "package-lock.json").write_text(
                '{"packages":{"":{"dependencies":{"esbuild":"^0.28.0"}}}}\n',
                encoding="utf-8",
            )
            config_path = files / RUNTIME_BOOTSTRAP_CONFIG_PATH
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                '{"watch_paths":["package.json","package-lock.json"],"commands":[]}\n',
                encoding="utf-8",
            )

            spec = sandbox.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=files,
                rootfs_path=rootfs,
                mode="chroot",
            )
            session = self._worker_session("session-1", spec, files, rootfs)

            WorkerService()._sync_missing_bootstrap_watch_paths_to_sandbox_sync(session)

            self.assertFalse((rootfs_workspace / "package-lock.json").exists())
            self.assertTrue((rootfs_workspace / "package.json").exists())

    @staticmethod
    def _worker_session(session_id: str, spec: sandbox.SandboxSpec, files: Path, rootfs: Path):
        from runtime.worker.service import WorkerSession

        return WorkerSession(
            id=session_id,
            workspace_id=spec.workspace_id,
            provider_session_id="provider-session-1",
            workspace_root=rootfs.parent,
            workspace_files_path=files,
            sandbox_spec=spec,
            pty_access_token="pty-token",
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
            updated_at=datetime.now(timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
