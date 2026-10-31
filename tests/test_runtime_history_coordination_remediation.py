"""Regression coverage for runtime history coordination lifecycle edges."""

from __future__ import annotations

import asyncio
import concurrent.futures
import http.client
import socket
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from runtime.core.private_file_response import private_file_response
from runtime.worker.sqlite_history.operations import OperationStore
from runtime.worker.sqlite_history.storage import AsyncRepositoryGate


class RepositoryGateRemediationTests(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_writer_is_not_starved_by_later_readers_with_busy_default_executor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order: list[str] = []
            loop = asyncio.get_running_loop()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            loop.set_default_executor(executor)
            occupied = asyncio.create_task(asyncio.to_thread(lambda: __import__("time").sleep(0.05)))
            acquired, release = asyncio.Event(), asyncio.Event()
            try:
                holder = asyncio.create_task(self._hold_reader(root, acquired, release))
                await acquired.wait()
                writer = asyncio.create_task(self._record(root, True, "writer", order))
                await asyncio.sleep(0.02)
                reader = asyncio.create_task(self._record(root, False, "reader", order))
                release.set()
                await asyncio.gather(writer, reader)
                self.assertEqual(order, ["writer", "reader"])
                await holder
            finally:
                await occupied
                executor.shutdown(wait=True)

    async def test_cancelled_waiter_does_not_leave_a_writer_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            acquired, release = asyncio.Event(), asyncio.Event()
            holder = asyncio.create_task(self._hold_reader(root, acquired, release))
            await acquired.wait()
            waiting = asyncio.create_task(self._record(root, True, "writer", []))
            await asyncio.sleep(0.02)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            release.set()
            await holder
            async with AsyncRepositoryGate(root, exclusive=False):
                pass

    @staticmethod
    async def _record(root: Path, exclusive: bool, value: str, order: list[str]) -> None:
        async with AsyncRepositoryGate(root, exclusive=exclusive):
            order.append(value)

    @staticmethod
    async def _hold_reader(root: Path, acquired: asyncio.Event, release: asyncio.Event) -> None:
        async with AsyncRepositoryGate(root, exclusive=False):
            acquired.set()
            await release.wait()


class ReceiptRetentionRemediationTests(unittest.TestCase):
    def test_acknowledged_terminal_receipt_retires_to_a_nonreplayable_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            operation_id = str(uuid4())
            accepted = dict(operation_id=operation_id, workspace_id="workspace", creator_id="user", request_digest="digest", kind="capture")
            store.accept(**accepted)
            store.transition(operation_id, "cancelled")
            store.acknowledge(operation_id, acknowledged_at=(datetime.now(timezone.utc) - timedelta(days=31)).isoformat())
            retired = store.retire_acknowledged_before(datetime.now(timezone.utc) - timedelta(days=30))
            self.assertEqual([operation_id], [item["operation_id"] for item in retired])
            replay = store.accept(**accepted)
            self.assertEqual("cancelled", replay["phase"])
            self.assertIsNotNone(replay["retired_at"])

    def test_malformed_acknowledgement_does_not_block_other_receipt_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            invalid_id, valid_id = str(uuid4()), str(uuid4())
            for operation_id in (invalid_id, valid_id):
                store.accept(
                    operation_id=operation_id,
                    workspace_id="workspace",
                    creator_id="user",
                    request_digest=operation_id,
                    kind="capture",
                )
                store.transition(operation_id, "cancelled")
            store.update_progress(invalid_id, acknowledged_at="not-a-timestamp")
            store.acknowledge(valid_id, acknowledged_at=(datetime.now(timezone.utc) - timedelta(days=31)).isoformat())

            retired = store.retire_acknowledged_before(datetime.now(timezone.utc) - timedelta(days=30))

            self.assertEqual([valid_id], [item["operation_id"] for item in retired])
            self.assertIsNone(store.get(invalid_id)["retired_at"])


class PrivateFileResponseNetworkTests(unittest.TestCase):
    def test_uvicorn_h11_streams_bytes_and_releases_lifetime_after_disconnect(self) -> None:
        import uvicorn
        from fastapi import FastAPI

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "download.sqlite3"
            payload = b"history-bytes" * 262_144
            path.write_bytes(payload)
            cleaned = threading.Event()
            cleanup_count = [0]
            cleanup_lock = threading.Lock()
            pinned = threading.Event()

            class _Lifetime:
                async def __aenter__(self):
                    pinned.set()

                async def __aexit__(self, *_args):
                    pinned.clear()

            async def cleanup() -> None:
                with cleanup_lock:
                    cleanup_count[0] += 1
                cleaned.set()

            app = FastAPI()

            @app.get("/download")
            async def download():
                return private_file_response(path, media_type="application/x-sqlite3", cleanup=cleanup, lifetime=_Lifetime)

            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
            thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
            thread.start()
            try:
                startup_deadline = time.monotonic() + 30
                while not server.started and thread.is_alive() and time.monotonic() < startup_deadline:
                    time.sleep(0.01)
                self.assertTrue(
                    server.started,
                    f"Uvicorn did not start within 30 seconds (thread_alive={thread.is_alive()})",
                )

                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    connection.request("GET", "/download")
                    response = connection.getresponse()
                    self.assertEqual(200, response.status)
                    self.assertEqual(payload, response.read())
                finally:
                    connection.close()
                self.assertTrue(cleaned.wait(30), "response cleanup did not finish within 30 seconds")
                self.assertFalse(pinned.is_set())
                # A real h11 client disconnects after headers/body begin.  The
                # response lifecycle must still close the file and pin.
                with socket.create_connection(("127.0.0.1", port), timeout=5) as disconnect:
                    disconnect.sendall(b"GET /download HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                    disconnect.recv(1024)

                cleanup_deadline = time.monotonic() + 30
                while time.monotonic() < cleanup_deadline:
                    with cleanup_lock:
                        if cleanup_count[0] >= 2:
                            break
                    time.sleep(0.01)
                with cleanup_lock:
                    self.assertEqual(2, cleanup_count[0])
                self.assertFalse(pinned.is_set())
            finally:
                server.should_exit = True
                thread.join(30)
                listener.close()
                self.assertFalse(thread.is_alive(), "Uvicorn did not stop within 30 seconds")
