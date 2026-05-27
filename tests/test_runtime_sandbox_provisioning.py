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
