from __future__ import annotations

import errno
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from ragtime.core.mount_health import MountHealthChecker, parse_mountinfo, redact_mount_source


def mountinfo(*lines: str) -> str:
    return "\n".join(lines)


def entry(mount_id: int, mount_point: str, fstype: str, source: str = "server:/share", parent_id: int = 1) -> str:
    return f"{mount_id} {parent_id} 0:1 / {mount_point} rw - {fstype} {source} rw"


class MountHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.release_probe = threading.Event()
        self.threads: list[threading.Thread] = []

    def tearDown(self) -> None:
        self.release_probe.set()
        for thread in self.threads:
            thread.join(1)

    def test_parse_mountinfo_decodes_escapes_and_skips_malformed(self) -> None:
        parsed = parse_mountinfo(mountinfo(entry(1, r"/a\040b\011c\012d\134e", "nfs"), "broken", "x y - nfs source"))
        self.assertEqual(parsed[0].mount_point, "/a b\tc\nd\\e")
        self.assertEqual(len(parsed), 1)

    def test_redacts_mount_credentials(self) -> None:
        self.assertEqual(redact_mount_source("//user:pw@host/share"), "//host/share")
        self.assertEqual(redact_mount_source("user@host:/p"), "host:/p")
        self.assertEqual(redact_mount_source("ssh://user:pw@host/x"), "ssh://host/x")
        self.assertEqual(redact_mount_source("//host/share"), "//host/share")
        self.assertEqual(redact_mount_source("host:/export"), "host:/export")

    def test_stacked_automount_is_active_regardless_of_parent_id(self) -> None:
        checker = MountHealthChecker(
            mountinfo_reader=lambda: mountinfo(entry(1, "/share", "autofs"), entry(2, "/share", "nfs", parent_id=99)), probe=lambda _: None
        )
        result = checker.check()[0]
        self.assertEqual((result.fstype, result.state), ("nfs", "ok"))

    def test_inactive_automount_recovers_when_probe_activates_it(self) -> None:
        reads = iter((mountinfo(entry(1, "/share", "autofs")), mountinfo(entry(1, "/share", "autofs"), entry(2, "/share", "nfs"))))
        checker = MountHealthChecker(mountinfo_reader=lambda: next(reads), probe=lambda _: None)
        result = checker.check()[0]
        self.assertEqual((result.state, result.recovered), ("ok", True))

    def test_inactive_automount_uses_host_automount_as_source(self) -> None:
        checker = MountHealthChecker(mountinfo_reader=lambda: mountinfo(entry(1, "/share", "autofs", "systemd-1")), probe=lambda _: None)
        result = checker.check()[0]
        self.assertEqual(result.source, "host automount")

    def test_probe_errors_and_timeouts(self) -> None:
        failed = MountHealthChecker(
            mountinfo_reader=lambda: mountinfo(entry(1, "/share", "autofs")),
            probe=lambda _: (_ for _ in ()).throw(OSError(errno.ENODEV, "missing")),
        ).check()[0]
        self.assertEqual((failed.state, failed.error), ("failed", "No such device"))

        def hang(_: str) -> None:
            self.threads.append(threading.current_thread())
            self.release_probe.wait()

        timed_out = MountHealthChecker(mountinfo_reader=lambda: mountinfo(entry(1, "/share", "nfs")), probe=hang, probe_timeout_seconds=0.05).check()[0]
        self.assertEqual(timed_out.state, "unresponsive")
        self.assertEqual(timed_out.error, "No response within 0.05s")
        self.assertTrue(self.threads[0].daemon)

    def test_thread_start_failure_does_not_leave_probe_in_flight(self) -> None:
        checker = MountHealthChecker(mountinfo_reader=lambda: mountinfo(entry(1, "/share", "nfs")), probe=lambda _: None)
        with patch.object(threading.Thread, "start", side_effect=RuntimeError):
            failed = checker.check()[0]
        recovered = checker.check()[0]
        self.assertEqual((failed.state, failed.error), ("failed", "Could not start mount probe"))
        self.assertEqual(recovered.state, "ok")

    def test_hung_probe_is_reused_but_changed_mount_id_starts_new_probe(self) -> None:
        current = [mountinfo(entry(1, "/share", "nfs"))]
        started = 0

        def hang(_: str) -> None:
            nonlocal started
            started += 1
            self.threads.append(threading.current_thread())
            self.release_probe.wait()

        checker = MountHealthChecker(mountinfo_reader=lambda: current[0], probe=hang, probe_timeout_seconds=0.05)
        checker.check()
        reused = checker.check()[0]
        self.assertEqual((reused.error, started), ("Previous check still pending", 1))
        current[0] = mountinfo(entry(2, "/share", "nfs"))
        checker.check()
        self.assertEqual(started, 2)

    def test_snapshot_does_not_wait_for_running_check(self) -> None:
        probe_started = threading.Event()

        def hang(_: str) -> None:
            probe_started.set()
            self.release_probe.wait()

        checker = MountHealthChecker(mountinfo_reader=lambda: mountinfo(entry(1, "/share", "nfs")), probe=hang, probe_timeout_seconds=5)
        running = threading.Thread(target=checker.check, daemon=True)
        self.threads.append(running)
        running.start()
        self.assertTrue(probe_started.wait(1))
        snapshot_done = threading.Event()

        def read_snapshot() -> None:
            checker.snapshot()
            checker.last_checked_at()
            snapshot_done.set()

        reader = threading.Thread(target=read_snapshot, daemon=True)
        reader.start()
        self.assertTrue(snapshot_done.wait(0.5))
        self.release_probe.set()

    def test_capacity_hysteresis_and_dropped_state(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        clock = lambda: now
        mounts = [mountinfo(entry(1, "/one", "nfs"), entry(2, "/two", "nfs"))]

        def hang(_: str) -> None:
            self.threads.append(threading.current_thread())
            self.release_probe.wait()

        checker = MountHealthChecker(
            mountinfo_reader=lambda: mounts[0],
            probe=hang,
            max_probe_threads=1,
            probe_timeout_seconds=0.05,
            clock=clock,
        )
        first = {result.mount_point: result for result in checker.check()}
        self.assertEqual(first["/two"].error, "Probe capacity exhausted")
        self.assertFalse(first["/one"].reported)
        now += timedelta(seconds=1)
        second = {result.mount_point: result for result in checker.check()}
        self.assertTrue(second["/one"].reported)
        self.assertEqual(second["/one"].failing_since, first["/one"].failing_since)
        mounts[0] = ""
        checker.check()
        self.assertEqual(checker.snapshot(), [])

    def test_fast_probes_exceeding_capacity_all_succeed(self) -> None:
        release_first = threading.Event()

        def probe(path: str) -> None:
            if path == "/share-0":
                release = threading.Timer(0.02, release_first.set)
                self.threads.append(release)
                release.start()
                release_first.wait()

        checker = MountHealthChecker(
            mountinfo_reader=lambda: mountinfo(*(entry(index, f"/share-{index}", "nfs") for index in range(4))),
            probe=probe,
            max_probe_threads=1,
        )
        results = checker.check()
        self.assertEqual([(result.state, result.reported) for result in results], [("ok", False)] * 4)

    def test_ok_resets_failure_and_exclusions_are_path_bounded(self) -> None:
        result = [
            mountinfo(
                entry(1, "/procfoo", "nfs"),
                entry(2, "/proc/x", "nfs"),
                entry(3, "/custom/x", "nfs"),
                entry(4, f"{tempfile.gettempdir()}/mount", "nfs"),
                entry(5, "/etc/hosts", "ext4"),
            )
        ]
        failures = [True]

        def probe(_: str) -> None:
            if failures[0]:
                raise OSError(errno.ENODEV, "missing")

        checker = MountHealthChecker(mountinfo_reader=lambda: result[0], probe=probe, exclude_prefixes=("/custom",))
        self.assertEqual([item.mount_point for item in checker.check()], ["/procfoo"])
        failures[0] = False
        self.assertEqual(checker.check()[0].consecutive_failures, 0)

    def test_to_dict_is_json_safe(self) -> None:
        checker = MountHealthChecker(mountinfo_reader=lambda: mountinfo(entry(1, "/share", "nfs")), probe=lambda _: None)
        data = checker.check()[0].to_dict()
        self.assertIsInstance(data["checked_at"], str)
        self.assertIsNone(data["failing_since"])


if __name__ == "__main__":
    unittest.main()
