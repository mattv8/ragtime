from __future__ import annotations

import asyncio
import multiprocessing
import os
import select
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Barrier
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace import sqlite_capture_admission as admission
from ragtime.userspace.sqlite_runtime import run_sqlite_blocking

_BLOCKING_CHILD = "import os, sys; os.write(int(sys.argv[1]), b'1'); os.read(int(sys.argv[2]), 1)"


def _run_blocking_child(
    index_data_path: str,
    slot_wait_seconds: float,
    ready: Connection,
    release: Connection,
    result_queue: Queue[str],
) -> None:
    ready_fd = ready.fileno()
    release_fd = release.fileno()
    try:
        with (
            mock.patch.object(admission.settings, "index_data_path", index_data_path),
            mock.patch.object(admission, "CAPTURE_SLOT_WAIT_SECONDS", slot_wait_seconds),
        ):
            admission.run_admitted_subprocess(
                [sys.executable, "-c", _BLOCKING_CHILD, str(ready_fd), str(release_fd)],
                pass_fds=(ready_fd, release_fd),
                check=True,
            )
    except HTTPException as exc:
        result_queue.put(f"http-{exc.status_code}")
    else:
        result_queue.put("completed")
    finally:
        ready.close()
        release.close()


def _initialize_slot_directory(
    index_data_path: str,
    slot_wait_seconds: float,
    start: Barrier,
    result_queue: Queue[str],
) -> None:
    start.wait()
    try:
        with (
            mock.patch.object(admission.settings, "index_data_path", index_data_path),
            mock.patch.object(admission, "CAPTURE_SLOT_WAIT_SECONDS", slot_wait_seconds),
        ):
            admission.run_admitted_subprocess([sys.executable, "-c", ""], check=True)
    except HTTPException as exc:
        result_queue.put(f"http-{exc.status_code}")
    else:
        result_queue.put("completed")


class SqliteCaptureAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.index_root = Path(self.temp.name) / "indexes"
        self.index_root.mkdir()
        self.settings = mock.patch.object(admission.settings, "index_data_path", str(self.index_root))
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.addCleanup(self.temp.cleanup)
        with admission._admission_lock:
            admission._outstanding_requests = 0
            admission._waiting_slot_callers = 0

    def test_slot_files_are_created_only_under_protected_userspace_storage(self) -> None:
        result = admission.run_admitted_subprocess([sys.executable, "-c", ""], check=True)

        self.assertEqual(0, result.returncode)
        slots = self.index_root / "_userspace" / "sqlite_capture_slots"
        self.assertTrue(slots.is_dir())
        self.assertEqual(["capture-0.lock"], sorted(item.name for item in slots.iterdir()))
        self.assertTrue(stat.S_ISREG((slots / "capture-0.lock").lstat().st_mode))

    @unittest.skipUnless(os.name == "posix", "requires POSIX file descriptor semantics")
    def test_concurrent_processes_initialize_missing_slot_directories(self) -> None:
        context = multiprocessing.get_context("spawn")
        start = context.Barrier(4)
        results: Queue[str] = context.Queue()
        processes = [
            context.Process(
                target=_initialize_slot_directory,
                args=(str(self.index_root), admission.CAPTURE_SLOT_WAIT_SECONDS, start, results),
            )
            for _ in range(4)
        ]
        try:
            for process in processes:
                process.start()
            self.assertEqual(["completed"] * 4, sorted(results.get(timeout=5) for _ in processes))
            for process in processes:
                process.join(timeout=5)
                self.assertEqual(0, process.exitcode)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            results.close()
            results.join_thread()

    def test_symlinked_slot_fails_closed(self) -> None:
        slots = self.index_root / "_userspace" / "sqlite_capture_slots"
        slots.mkdir(parents=True)
        (slots / "capture-0.lock").symlink_to(self.index_root / "victim")

        with self.assertRaisesRegex(HTTPException, "admission storage is unavailable"):
            admission.run_admitted_subprocess([sys.executable, "-c", ""], check=True)

    def test_regular_file_in_place_of_slot_directory_fails_closed(self) -> None:
        userspace = self.index_root / "_userspace"
        userspace.mkdir()
        (userspace / "sqlite_capture_slots").write_text("not a directory", encoding="utf-8")

        with self.assertRaisesRegex(HTTPException, "admission storage is unavailable"):
            admission.run_admitted_subprocess([sys.executable, "-c", ""], check=True)

    @unittest.skipUnless(os.name == "posix", "requires POSIX file descriptor semantics")
    def test_two_processes_hold_slots_and_third_is_rejected_until_children_exit(self) -> None:
        context = multiprocessing.get_context("spawn")
        ready_read, ready_write = context.Pipe(duplex=False)
        release_read, release_write = context.Pipe(duplex=False)
        results: Queue[str] = context.Queue()
        slot_wait_seconds = 0.2
        holders: list[BaseProcess] = []
        rejected: BaseProcess | None = None
        try:
            child_args = (str(self.index_root), slot_wait_seconds, ready_write, release_read, results)
            holders = [context.Process(target=_run_blocking_child, args=child_args) for _ in range(2)]
            for holder in holders:
                holder.start()
            for _ in holders:
                self.assertTrue(select.select([ready_read.fileno()], [], [], 5)[0], "child never acquired its slot")
                self.assertEqual(b"1", os.read(ready_read.fileno(), 1))

            rejected_process = context.Process(target=_run_blocking_child, args=child_args)
            rejected_process.start()
            rejected = rejected_process
            ready_write.close()
            release_read.close()
            self.assertEqual("http-503", results.get(timeout=5))

            os.write(release_write.fileno(), b"12")
            self.assertEqual("completed", results.get(timeout=5))
            self.assertEqual("completed", results.get(timeout=5))
            assert rejected is not None
            for process in [*holders, rejected]:
                process.join(timeout=5)
                self.assertEqual(0, process.exitcode)
        finally:
            try:
                os.write(release_write.fileno(), b"12")
            except OSError:
                pass
            for process in [*holders, *([rejected] if rejected is not None else [])]:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            ready_read.close()
            ready_write.close()
            release_read.close()
            release_write.close()
            results.close()
            results.join_thread()

    def test_subprocess_timeout_releases_slot(self) -> None:
        ready_read, ready_write = os.pipe()
        release_read, release_write = os.pipe()
        failure: list[BaseException] = []

        def run_timeout() -> None:
            try:
                admission.run_admitted_subprocess(
                    [sys.executable, "-c", _BLOCKING_CHILD, str(ready_write), str(release_read)],
                    pass_fds=(ready_write, release_read),
                    timeout=0.2,
                )
            except BaseException as exc:  # asserted in the owning test thread
                failure.append(exc)

        thread = threading.Thread(target=run_timeout)
        thread.start()
        try:
            self.assertTrue(select.select([ready_read], [], [], 5)[0], "child did not start")
            self.assertEqual(b"1", os.read(ready_read, 1))
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(failure[0], subprocess.TimeoutExpired)
            self.assertEqual(0, admission.run_admitted_subprocess([sys.executable, "-c", ""], check=True).returncode)
        finally:
            os.close(ready_read)
            os.close(ready_write)
            os.close(release_read)
            os.close(release_write)


class CaptureRequestAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_request_keeps_real_child_slot_until_run_sqlite_blocking_drains(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            index_root = Path(temp) / "indexes"
            index_root.mkdir()
            ready_read, ready_write = os.pipe()
            release_read, release_write = os.pipe()
            ready = threading.Event()

            def wait_for_child() -> None:
                os.read(ready_read, 1)
                ready.set()

            watcher = threading.Thread(target=wait_for_child)
            watcher.start()

            def capture_child() -> None:
                admission.run_admitted_subprocess(
                    [sys.executable, "-c", _BLOCKING_CHILD, str(ready_write), str(release_read)],
                    pass_fds=(ready_write, release_read),
                    check=True,
                )

            async def request() -> None:
                async with admission.capture_request_admission():
                    await run_sqlite_blocking(capture_child)

            def attempt_second_child() -> int:
                try:
                    admission.run_admitted_subprocess([sys.executable, "-c", ""], check=True)
                except HTTPException as exc:
                    return exc.status_code
                return 0

            try:
                with (
                    mock.patch.object(admission.settings, "index_data_path", str(index_root)),
                    mock.patch.object(admission, "MAX_OUTSTANDING_CAPTURE_REQUESTS", 1),
                    mock.patch.object(admission, "MAX_CONCURRENT_CAPTURE_SUBPROCESSES", 1),
                    mock.patch.object(admission, "CAPTURE_SLOT_WAIT_SECONDS", 0.1),
                ):
                    task = asyncio.create_task(request())
                    self.assertTrue(await asyncio.to_thread(ready.wait, 5), "child did not acquire its slot")
                    task.cancel()
                    await asyncio.sleep(0)
                    with self.assertRaises(HTTPException) as error:
                        async with admission.capture_request_admission():
                            pass
                    self.assertEqual(503, error.exception.status_code)
                    self.assertEqual(503, await asyncio.to_thread(attempt_second_child))
                    os.write(release_write, b"1")
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    async with admission.capture_request_admission():
                        pass
            finally:
                os.close(ready_read)
                os.close(ready_write)
                os.close(release_read)
                os.close(release_write)
                watcher.join(timeout=5)
