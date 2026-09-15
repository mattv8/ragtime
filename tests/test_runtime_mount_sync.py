from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from runtime.worker import mount_sync, mount_sync_launcher


class _CompletedProcess:
    pid = 12345

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


class _WaitingProcess:
    pid = 4242

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def wait(self, timeout: float | None = None) -> int:
        assert timeout is not None
        self.timeouts.append(timeout)
        raise subprocess.TimeoutExpired("mount-sync", timeout)


class MountSyncTests(unittest.TestCase):
    def _directory_fds(self) -> tuple[tempfile.TemporaryDirectory[str], int, int]:
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        source = root / "source"
        destination = root / "destination"
        source.mkdir()
        destination.mkdir()
        return directory, os.open(source, os.O_RDONLY), os.open(destination, os.O_RDONLY)

    def test_launcher_uses_pinned_fd_contents_and_fixed_flags(self) -> None:
        argv = mount_sync_launcher._rsync_argv(12, 13, ("nested/[literal]*",))
        self.assertEqual(
            argv[:9],
            [
                "/usr/bin/rsync",
                "--recursive",
                "--links",
                "--perms",
                "--times",
                "--xattrs",
                "--checksum",
                "--modify-window=-1",
                "--delete-delay",
            ],
        )
        self.assertEqual(argv[-3:], ["--", "/proc/self/fd/12/", "/proc/self/fd/13/"])
        self.assertIn("/nested/\\[literal\\]\\*", argv)
        self.assertIn("/nested/\\[literal\\]\\*/***", argv)
        self.assertNotIn("--archive", argv)
        self.assertNotIn("--inplace", argv)

    def test_rejects_identical_and_overlapping_roots_before_spawn(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        nested_fd = -1
        try:
            with self.assertRaisesRegex(ValueError, "identical"):
                mount_sync.sync_copied_mount(source_fd, source_fd)
            Path(os.path.realpath(f"/proc/self/fd/{source_fd}"), "child").mkdir()
            nested_fd = os.open(f"/proc/self/fd/{source_fd}/child", os.O_RDONLY)
            with self.assertRaisesRegex(ValueError, "overlap"):
                mount_sync.sync_copied_mount(source_fd, nested_fd)
        finally:
            if nested_fd >= 0:
                os.close(nested_fd)
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    def test_preset_cancel_event_aborts_before_spawn(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        cancel_event = threading.Event()
        cancel_event.set()
        try:
            with mock.patch.object(mount_sync.subprocess, "Popen") as popen:
                with self.assertRaises(mount_sync.MountSyncCancelled):
                    mount_sync.sync_copied_mount(source_fd, destination_fd, cancel_event=cancel_event)
            popen.assert_not_called()
        finally:
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    def test_special_creation_rights_are_handled_but_not_granted_to_destination(self) -> None:
        special = mount_sync_launcher._MAKE_CHAR | mount_sync_launcher._MAKE_BLOCK | mount_sync_launcher._MAKE_SOCK | mount_sync_launcher._MAKE_FIFO
        self.assertEqual(mount_sync_launcher._DESTINATION_RIGHTS & special, 0)
        self.assertEqual(mount_sync_launcher._HANDLED_RIGHTS & special, special)

    def test_launcher_rejects_malformed_protected_path(self) -> None:
        with self.assertRaises(ValueError):
            mount_sync_launcher._validate_protected_paths(("../child",))

    def test_post_transfer_retired_source_fd_is_a_visible_error(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        source_path = Path(os.path.realpath(f"/proc/self/fd/{source_fd}"))
        try:

            def retire_source(*_args: object, **_kwargs: object) -> _CompletedProcess:
                source_path.rmdir()
                return _CompletedProcess()

            with (
                mock.patch.object(mount_sync, "mount_sync_available", return_value=True),
                mock.patch.object(mount_sync.subprocess, "Popen", side_effect=retire_source),
            ):
                with self.assertRaisesRegex(mount_sync.MountSyncError, "changed generation"):
                    mount_sync.sync_copied_mount(source_fd, destination_fd)
        finally:
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    def test_termination_phases_share_each_monotonic_deadline(self) -> None:
        process = _WaitingProcess()
        with (
            mock.patch.object(mount_sync, "_active_group_members", return_value=(process.pid,)),
            mock.patch.object(mount_sync, "_wait_for_group_quiescence", side_effect=(False, False)),
            mock.patch.object(mount_sync.os, "killpg"),
            mock.patch.object(mount_sync.time, "monotonic", side_effect=(0.0, 0.25, 1.0, 1.5)),
        ):
            self.assertFalse(mount_sync._terminate_group(cast(subprocess.Popen[bytes], process)))
        self.assertEqual(process.timeouts, [0.75, 1.5])

    @unittest.skipUnless(mount_sync.mount_sync_available(), "confined rsync is unavailable")
    def test_protected_root_excludes_source_file_and_literal_metacharacter_name(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        literal = "child*?[]\\"
        leading_dash = "-child"
        source_root = Path(os.path.realpath(f"/proc/self/fd/{source_fd}"))
        destination_root = Path(os.path.realpath(f"/proc/self/fd/{destination_fd}"))
        try:
            (source_root / "child").write_text("parent-file", encoding="utf-8")
            (source_root / literal).write_text("parent-file", encoding="utf-8")
            (source_root / leading_dash).write_text("parent-file", encoding="utf-8")
            (source_root / "copied").write_text("copied", encoding="utf-8")
            for protected in ("child", literal, leading_dash):
                target = destination_root / protected
                target.mkdir()
                (target / "nested-mount-marker").write_text("keep", encoding="utf-8")
            mount_sync.sync_copied_mount(source_fd, destination_fd, protected_paths=("child", literal, leading_dash))
            self.assertEqual((destination_root / "child" / "nested-mount-marker").read_text(), "keep")
            self.assertEqual((destination_root / literal / "nested-mount-marker").read_text(), "keep")
            self.assertEqual((destination_root / leading_dash / "nested-mount-marker").read_text(), "keep")
            self.assertEqual((destination_root / "copied").read_text(), "copied")
        finally:
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    @unittest.skipUnless(mount_sync.mount_sync_available(), "confined rsync is unavailable")
    def test_retired_source_fd_never_deletes_destination_after_path_replacement(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        source_path = Path(os.path.realpath(f"/proc/self/fd/{source_fd}"))
        destination_root = Path(os.path.realpath(f"/proc/self/fd/{destination_fd}"))
        try:
            sentinel = destination_root / "destination-sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            source_path.rmdir()
            source_path.mkdir()
            (source_path / "fresh-source").write_text("fresh", encoding="utf-8")
            with self.assertRaisesRegex(mount_sync.MountSyncError, "retired"):
                mount_sync.sync_copied_mount(source_fd, destination_fd)
            self.assertEqual(sentinel.read_text(), "keep")
            self.assertEqual((source_path / "fresh-source").read_text(), "fresh")
        finally:
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    @unittest.skipUnless(sys.platform == "linux", "process groups require Linux /proc")
    def test_timeout_and_cancel_drain_a_real_child_process_group(self) -> None:
        real_popen = subprocess.Popen
        groups: list[int] = []

        def launch_sleeping_group(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
            process = real_popen(
                [sys.executable, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); time.sleep(30)"],
                start_new_session=True,
            )
            groups.append(process.pid)
            return process

        directory, source_fd, destination_fd = self._directory_fds()
        try:
            with (
                mock.patch.object(mount_sync, "mount_sync_available", return_value=True),
                mock.patch.object(mount_sync.subprocess, "Popen", side_effect=launch_sleeping_group),
            ):
                with self.assertRaisesRegex(mount_sync.MountSyncError, "timed out"):
                    mount_sync.sync_copied_mount(source_fd, destination_fd, timeout_seconds=0.05)

            cancel_event = threading.Event()
            trigger = threading.Timer(0.05, cancel_event.set)
            trigger.start()
            try:
                with (
                    mock.patch.object(mount_sync, "mount_sync_available", return_value=True),
                    mock.patch.object(mount_sync.subprocess, "Popen", side_effect=launch_sleeping_group),
                ):
                    with self.assertRaises(mount_sync.MountSyncCancelled):
                        mount_sync.sync_copied_mount(source_fd, destination_fd, cancel_event=cancel_event, timeout_seconds=5)
            finally:
                trigger.cancel()
            time.sleep(0.05)
            self.assertTrue(all(not mount_sync._active_group_members(pgid) for pgid in groups))
        finally:
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    def test_passes_only_pinned_descriptors_and_does_not_close_them(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        try:
            with (
                mock.patch.object(mount_sync, "mount_sync_available", return_value=True),
                mock.patch.object(mount_sync.subprocess, "Popen", return_value=_CompletedProcess()) as popen,
            ):
                mount_sync.sync_copied_mount(source_fd, destination_fd, protected_paths=("-child",))
            self.assertEqual(popen.call_args.args[0][1], str(Path(mount_sync_launcher.__file__).resolve()))
            self.assertNotIn("-m", popen.call_args.args[0])
            self.assertIn("--protected-path=-child", popen.call_args.args[0])
            self.assertEqual(popen.call_args.kwargs["pass_fds"], (source_fd, destination_fd))
            self.assertTrue(popen.call_args.kwargs["close_fds"])
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertTrue(os.path.isdir(f"/proc/self/fd/{source_fd}"))
            self.assertTrue(os.path.isdir(f"/proc/self/fd/{destination_fd}"))
        finally:
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    @unittest.skipUnless(mount_sync.mount_sync_available(), "confined rsync is unavailable")
    def test_real_helper_runs_after_cwd_change_without_pythonpath(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        original_cwd = os.getcwd()
        source_root = Path(os.path.realpath(f"/proc/self/fd/{source_fd}"))
        destination_root = Path(os.path.realpath(f"/proc/self/fd/{destination_fd}"))
        try:
            (source_root / "from-standalone-launcher").write_text("ok", encoding="utf-8")
            with tempfile.TemporaryDirectory() as unrelated:
                os.chdir(unrelated)
                mount_sync.sync_copied_mount(source_fd, destination_fd)
                (source_root / "direct-launcher").write_text("also-ok", encoding="utf-8")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(mount_sync_launcher.__file__).resolve()),
                        "--source-fd",
                        str(source_fd),
                        "--destination-fd",
                        str(destination_fd),
                    ],
                    capture_output=True,
                    check=False,
                    close_fds=True,
                    cwd=unrelated,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                    pass_fds=(source_fd, destination_fd),
                )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertNotIn("RUNTIME_AUTH_TOKEN", result.stderr.decode(errors="replace"))
            self.assertEqual((destination_root / "from-standalone-launcher").read_text(), "ok")
            self.assertEqual((destination_root / "direct-launcher").read_text(), "also-ok")
        finally:
            os.chdir(original_cwd)
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    def test_unavailable_is_distinct_from_started_launcher_failure(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        try:
            with mock.patch.object(mount_sync, "mount_sync_available", return_value=False):
                with self.assertRaises(mount_sync.MountSyncUnavailable):
                    mount_sync.sync_copied_mount(source_fd, destination_fd)
            with (
                mock.patch.object(mount_sync, "mount_sync_available", return_value=True),
                mock.patch.object(mount_sync.subprocess, "Popen", return_value=_CompletedProcess(23)),
            ):
                with self.assertRaisesRegex(mount_sync.MountSyncError, "23"):
                    mount_sync.sync_copied_mount(source_fd, destination_fd)
        finally:
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()

    def test_invalid_protected_path_and_timeout_are_rejected_before_spawn(self) -> None:
        directory, source_fd, destination_fd = self._directory_fds()
        try:
            with self.assertRaises(ValueError):
                mount_sync.sync_copied_mount(source_fd, destination_fd, protected_paths=("../outside",))
            with self.assertRaises(ValueError):
                mount_sync.sync_copied_mount(source_fd, destination_fd, timeout_seconds=float("inf"))
        finally:
            os.close(source_fd)
            os.close(destination_fd)
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
