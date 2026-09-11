import asyncio
import tempfile
import threading
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ragtime.rag.components import RAGComponents


class FaissResourceLoadingTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiter_keeps_load_lease_until_thread_finishes(self) -> None:
        entered = asyncio.Event()
        release_thread = threading.Event()
        releases: list[bool] = []

        class Governor:
            def estimate_request(self, **_kwargs):
                return object()

            @asynccontextmanager
            async def acquire(self, _request):
                try:
                    yield object()
                finally:
                    releases.append(release_thread.is_set())

        def blocking_load(*_args, **_kwargs):
            entered_loop.call_soon_threadsafe(entered.set)
            release_thread.wait(timeout=2)
            return SimpleNamespace(index=SimpleNamespace(d=2))

        rag = RAGComponents()
        rag._app_settings = {"embedding_dimension": 2}
        rag._index_details["example"] = {"status": "pending"}
        entered_loop = asyncio.get_running_loop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (
                patch("ragtime.indexer.resource_governor.resource_governor", Governor()),
                patch("ragtime.rag.components.FAISS.load_local", side_effect=blocking_load),
            ):
                task = asyncio.create_task(
                    rag._load_faiss_local_admitted(
                        index_name="example",
                        index_path=path,
                        embedding_model=object(),
                        metadata={"chunk_count": 1, "embedding_dimension": 2},
                    )
                )
                await entered.wait()
                task.cancel()
                release_thread.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        self.assertEqual(releases, [True])

    async def test_double_cancel_keeps_load_lease_until_thread_finishes(self) -> None:
        entered = asyncio.Event()
        release_thread = threading.Event()
        releases: list[bool] = []

        class Governor:
            def estimate_request(self, **_kwargs):
                return object()

            @asynccontextmanager
            async def acquire(self, _request):
                try:
                    yield object()
                finally:
                    releases.append(release_thread.is_set())

        def blocking_load(*_args, **_kwargs):
            entered_loop.call_soon_threadsafe(entered.set)
            release_thread.wait(timeout=2)
            return SimpleNamespace(index=SimpleNamespace(d=2))

        rag = RAGComponents()
        rag._app_settings = {"embedding_dimension": 2}
        rag._index_details["example"] = {"status": "pending"}
        entered_loop = asyncio.get_running_loop()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("ragtime.indexer.resource_governor.resource_governor", Governor()),
                patch("ragtime.rag.components.FAISS.load_local", side_effect=blocking_load),
            ):
                task = asyncio.create_task(
                    rag._load_faiss_local_admitted(
                        index_name="example", index_path=Path(directory), embedding_model=object(), metadata={"chunk_count": 1, "embedding_dimension": 2}
                    )
                )
                await entered.wait()
                task.cancel()
                task.cancel()
                release_thread.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        self.assertEqual(releases, [True])
