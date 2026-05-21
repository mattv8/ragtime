import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime.worker import sandbox


class SandboxProvisioningTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
