import asyncio
import unittest
from typing import Any, List
from unittest import mock

from langchain_core.documents import Document

from ragtime.indexer import chunking
from ragtime.indexer.chunking import (
    SHARED_CHUNKING_POOL_KEY,
    ChunkingPool,
    ChunkingPoolError,
    pool_manager,
)
from ragtime.indexer.code_extraction import extract_metadata


class _FakeProcess:
    """Stand-in for multiprocessing.Process that records signal calls."""

    def __init__(self, survives_sigterm: bool = False) -> None:
        self.terminated = False
        self.killed = False
        self.survives_sigterm = survives_sigterm
        self.join_history: List[float | None] = []

    def is_alive(self) -> bool:
        # Pretend the worker is alive until kill() proves otherwise. Survives_sigterm
        # only affects whether the kill escalation is required.
        return not self.killed

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def join(self, timeout=None) -> None:
        self.join_history.append(timeout)
        if self.killed:
            self.survives_sigterm = False


class _FakeExecutor:
    """Minimal ProcessPoolExecutor substitute for termination tests."""

    def __init__(self, processes: List[_FakeProcess]) -> None:
        self._processes = {i + 1: p for i, p in enumerate(processes)}
        self.shutdown_args: tuple[bool, bool] | None = None

    def shutdown(self, *, wait=True, cancel_futures=False) -> None:
        self.shutdown_args = (wait, cancel_futures)


class ChunkingResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure every test starts from a clean manager state; the module-level
        # pool_manager is a singleton shared across tests.
        pool_manager.shutdown_all(terminate_workers=False)

    def tearDown(self) -> None:
        pool_manager.shutdown_all(terminate_workers=False)

    # ------------------------------------------------------------------
    # Existing behavioural tests
    # ------------------------------------------------------------------

    def test_tsx_metadata_extraction_uses_supported_parser_input(self) -> None:
        text = """
import React from 'react';

interface Props {
  label: string;
}

function Example(props: Props) {
  return <div>{props.label}</div>;
}
"""

        imports, definitions = extract_metadata(text, ".tsx")

        self.assertIn("import React from 'react';", imports)
        self.assertTrue(any("function Example" in item for item in definitions))

    def test_worker_count_respects_configured_cap(self) -> None:
        original_workers = chunking._configured_max_workers
        original_batch = chunking._configured_max_batch_size
        try:
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=100)
            # 32 cores // 4 = 8, capped at configured 4
            self.assertEqual(chunking._resolve_worker_count(32), 4)
            # 8 cores // 4 = 2 (below cap)
            self.assertEqual(chunking._resolve_worker_count(8), 2)
            # Tightening the cap reduces the resolved worker count.
            chunking.configure_chunking_pool(max_workers=2, max_batch_size=100)
            self.assertEqual(chunking._resolve_worker_count(32), 2)
        finally:
            chunking.configure_chunking_pool(
                max_workers=original_workers,
                max_batch_size=original_batch,
            )

    def test_batch_size_respects_configured_cap(self) -> None:
        original_workers = chunking._configured_max_workers
        original_batch = chunking._configured_max_batch_size
        try:
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=100)
            self.assertEqual(chunking._effective_batch_size(500), 100)
            self.assertEqual(chunking._effective_batch_size(50), 50)
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=25)
            self.assertEqual(chunking._effective_batch_size(500), 25)
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=200)
            self.assertEqual(chunking._effective_batch_size(500), 200)
        finally:
            chunking.configure_chunking_pool(
                max_workers=original_workers,
                max_batch_size=original_batch,
            )

    def test_configure_chunking_pool_clamps_to_safe_range(self) -> None:
        original_workers = chunking._configured_max_workers
        original_batch = chunking._configured_max_batch_size
        try:
            chunking.configure_chunking_pool(max_workers=999, max_batch_size=999_999)
            self.assertEqual(chunking._configured_max_workers, chunking._CHUNKING_WORKERS_HARD_CEILING)
            self.assertEqual(chunking._configured_max_batch_size, chunking._CHUNKING_BATCH_SIZE_HARD_CEILING)
            chunking.configure_chunking_pool(max_workers=0, max_batch_size=0)
            self.assertEqual(chunking._configured_max_workers, 1)
            self.assertEqual(chunking._configured_max_batch_size, 1)
        finally:
            chunking.configure_chunking_pool(
                max_workers=original_workers,
                max_batch_size=original_batch,
            )

    def test_recursive_fallback_marks_chunks(self) -> None:
        chunks, counts = chunking._chunk_document_batch_recursive_sync(
            [("alpha\n\nbeta\n\ngamma", {"source": "notes.txt"})],
            chunk_size=100,
            chunk_overlap=0,
            use_tokens=False,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1]["chunker"], "no_chunk_small")
        self.assertEqual(counts["chonkie_recursive_pool_fallback"], 1)

    def test_recursive_fallback_keeps_document_when_recursive_chunker_fails(self) -> None:
        with mock.patch.object(chunking, "_chunk_with_recursive", side_effect=RuntimeError("boom")):
            chunks, counts = chunking._chunk_document_batch_recursive_sync(
                [("alpha", {"source": "bad.txt"})],
                chunk_size=100,
                chunk_overlap=0,
                use_tokens=False,
            )

        self.assertEqual(chunks[0][0], "alpha")
        self.assertEqual(chunks[0][1]["chunker"], "whole_document_pool_fallback")
        self.assertEqual(counts["whole_document_pool_fallback"], 1)

    def test_rechunk_oversized_content_uses_legacy_recursive_chunker_signature(self) -> None:
        class LegacyRecursiveChunker:
            def __init__(self, tokenizer_or_token_counter, chunk_size, min_characters_per_chunk):
                self.tokenizer_or_token_counter = tokenizer_or_token_counter
                self.chunk_size = chunk_size
                self.min_characters_per_chunk = min_characters_per_chunk

            def chunk(self, content):
                return [mock.Mock(text=content[: self.chunk_size]), mock.Mock(text=content[self.chunk_size :])]

        class LegacyOverlapRefinery:
            def __init__(self, tokenizer_or_token_counter, context_size, mode, method, merge, inplace):
                self.tokenizer_or_token_counter = tokenizer_or_token_counter
                self.context_size = context_size
                self.mode = mode
                self.method = method
                self.merge = merge
                self.inplace = inplace

            def refine(self, chunks):
                return chunks

        with (
            mock.patch.object(chunking, "RecursiveChunker", LegacyRecursiveChunker),
            mock.patch.object(chunking, "OverlapRefinery", LegacyOverlapRefinery),
        ):
            docs = chunking.rechunk_oversized_content(
                "alpha beta gamma delta",
                safe_token_limit=3,
                chunk_overlap=1,
                metadata={"source": "oversized.txt"},
            )

        self.assertEqual([doc.metadata["chunker"] for doc in docs], ["rechunk_oversized", "rechunk_oversized"])
        self.assertEqual([doc.metadata["source"] for doc in docs], ["oversized.txt", "oversized.txt"])

    # ------------------------------------------------------------------
    # New resilience tests (per-job pools, SIGKILL escalation, shutdown detect)
    # ------------------------------------------------------------------

    def test_get_or_create_returns_same_pool_for_same_key(self) -> None:
        pool_a = pool_manager.get_or_create("job-a", max_workers=2)
        pool_b = pool_manager.get_or_create("job-a", max_workers=2)
        self.assertIs(pool_a, pool_b)
        self.assertEqual(pool_a.max_workers, 2)

    def test_get_or_create_isolates_jobs(self) -> None:
        pool_a = pool_manager.get_or_create("job-a", max_workers=2)
        pool_b = pool_manager.get_or_create("job-b", max_workers=2)
        self.assertIsNot(pool_a, pool_b)
        # Releasing job-a's pool must not affect job-b's pool.
        pool_manager.release("job-a")
        self.assertIsNone(pool_manager.get("job-a"))
        self.assertIs(pool_manager.get("job-b"), pool_b)
        self.assertFalse(pool_b.is_closed())

    def test_release_marks_pool_closed(self) -> None:
        pool = pool_manager.get_or_create("job-x", max_workers=1)
        self.assertFalse(pool.is_closed())
        pool_manager.release("job-x")
        self.assertTrue(pool.is_closed())

    def test_terminate_escalates_to_sigkill_when_sigterm_ignored(self) -> None:
        stubborn = _FakeProcess(survives_sigterm=True)
        cooperative = _FakeProcess(survives_sigterm=False)
        executor = _FakeExecutor([stubborn, cooperative])
        pool = ChunkingPool(key="job-x", executor=executor, max_workers=2)

        chunking._terminate_pool(pool, terminate_workers=True)

        self.assertTrue(stubborn.terminated, "SIGTERM should be attempted")
        self.assertTrue(stubborn.killed, "stubborn worker must be SIGKILLed")
        # First join during grace, second join after kill.
        self.assertEqual(len(stubborn.join_history), 2)
        first_join_timeout = stubborn.join_history[0]
        assert first_join_timeout is not None
        self.assertGreaterEqual(first_join_timeout, chunking._WORKER_SIGTERM_GRACE_SECONDS)
        # The executor is told to cancel pending futures and not block.
        self.assertEqual(executor.shutdown_args, (False, True))

    def test_terminate_skips_signal_escalation_when_disabled(self) -> None:
        stubborn = _FakeProcess(survives_sigterm=True)
        executor = _FakeExecutor([stubborn])
        pool = ChunkingPool(key="job-x", executor=executor, max_workers=1)

        chunking._terminate_pool(pool, terminate_workers=False)

        self.assertFalse(stubborn.terminated)
        self.assertEqual(stubborn.join_history, [])
        self.assertEqual(executor.shutdown_args, (False, True))

    def test_configure_chunking_pool_releases_existing_pools_on_worker_resize(self) -> None:
        original_workers = chunking._configured_max_workers
        try:
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=100)
            pool = pool_manager.get_or_create("job-resize", max_workers=2)
            self.assertIs(pool_manager.get("job-resize"), pool)

            chunking.configure_chunking_pool(max_workers=3, max_batch_size=100)
            # The pool was evicted; a new lookup creates a fresh one.
            self.assertIsNone(pool_manager.get("job-resize"))
        finally:
            chunking.configure_chunking_pool(
                max_workers=original_workers,
                max_batch_size=chunking._configured_max_batch_size,
            )

    def test_shutdown_process_pool_releases_every_registered_pool(self) -> None:
        pool_manager.get_or_create("job-a", max_workers=2)
        pool_manager.get_or_create("job-b", max_workers=2)
        chunking.shutdown_process_pool(terminate_workers=True)
        self.assertEqual(pool_manager.active_keys(), [])

    def test_normalize_ext_handles_dot_prefix(self) -> None:
        self.assertEqual(chunking._normalize_ext(".pdf"), (".pdf", "pdf"))
        self.assertEqual(chunking._normalize_ext("pdf"), (".pdf", "pdf"))
        self.assertEqual(chunking._normalize_ext(".py"), (".py", "py"))
        self.assertEqual(chunking._normalize_ext(""), ("", ""))

    def test_plain_text_mapping_finds_both_dot_forms(self) -> None:
        # .pdf is NOT in LANG_MAPPING with a leading dot, but `pdf` is.
        # The chunker must still treat it as plain text so the dispatcher
        # falls back to recursive chunking at paragraph/sentence boundaries.
        self.assertTrue(chunking._is_extension_mapped_to_plain_text(".pdf"))
        self.assertTrue(chunking._is_extension_mapped_to_plain_text("pdf"))
        # Large JSON lockfiles can wedge the AST detector; treat JSON as
        # structured text and split recursively instead.
        self.assertTrue(chunking._is_extension_mapped_to_plain_text(".json"))
        self.assertTrue(chunking._is_extension_mapped_to_plain_text("json"))
        # .txt / .csv have dotted entries; both forms resolve.
        self.assertTrue(chunking._is_extension_mapped_to_plain_text(".txt"))
        self.assertTrue(chunking._is_extension_mapped_to_plain_text("txt"))
        # Known code extensions stay on the code-chunker path.
        self.assertFalse(chunking._is_extension_mapped_to_plain_text(".py"))
        self.assertFalse(chunking._is_extension_mapped_to_plain_text(".md"))
        # Unknown extensions (e.g. .eps) are not explicit plain-text
        # mappings, but the dispatcher still routes them to RecursiveChunker
        # because they are not known code/document extensions.
        self.assertFalse(chunking._is_extension_mapped_to_plain_text(".eps"))

    def test_known_code_extension_is_not_treated_as_unmapped(self) -> None:
        """LANG_MAPPING is sparse; common code extensions must still use CodeChunker."""
        self.assertTrue(chunking._is_known_document_or_code_extension(".py"))
        self.assertTrue(chunking._is_known_document_or_code_extension("py"))
        self.assertTrue(chunking._is_known_document_or_code_extension(".js"))
        self.assertFalse(chunking._is_known_document_or_code_extension(".eps"))

        text = "def alpha():\n    return 1\n\nclass Beta:\n    pass\n" * 60
        chunks, counts = chunking._chunk_document_batch_sync(
            [(text, {"source": "src/sample.py"})],
            chunk_size=1000,
            chunk_overlap=100,
            use_tokens=False,
        )

        self.assertIn("chonkie_code", counts)
        self.assertNotIn("chonkie_recursive", counts)
        self.assertGreater(len(chunks), 0)

    def test_all_parseable_documents_route_to_recursive_chunker(self) -> None:
        """Every extension in PARSEABLE_DOCUMENT_EXTENSIONS must skip tree-sitter.

        PDF/Office text was extracted by the document parser — running
        tree-sitter's auto-detect on it is slow and adds no value. The
        RecursiveChunker splits at paragraph and sentence boundaries which
        is what we want for these document types.
        """
        from ragtime.core.file_constants import PARSEABLE_DOCUMENT_EXTENSIONS

        for ext in PARSEABLE_DOCUMENT_EXTENSIONS:
            with self.subTest(ext=ext):
                self.assertTrue(
                    chunking._is_extension_mapped_to_plain_text(ext),
                    f"{ext} should route to RecursiveChunker, not the slow auto-detect path",
                )

    def test_extract_to_text_then_chunk_uses_recursive_chunker(self) -> None:
        """End-to-end: simulate the dispatch path used by service._create_faiss_index.

        Each parseable document type's extracted text gets dispatched to
        RecursiveChunker (chonkie_recursive splitter), not the slow
        tree-sitter code chunker.
        """
        # Realistic extracted document text (multi-paragraph).
        text = (
            "This is the first paragraph of an extracted document.\n\n"
            "It has multiple sentences that should be kept together. "
            "And here is another sentence.\n\n"
            "Second paragraph follows after a blank line. The recursive "
            "chunker should split between these paragraphs at semantic "
            "boundaries, not at the byte/character level.\n\n"
            "Third paragraph with a code-like identifier ABC-1234. "
            "The recursive chunker does not need to know this is code."
        )

        document_extensions = [
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".odt",
            ".ods",
            ".odp",
            ".rtf",
            ".epub",
            ".eml",
            ".msg",
        ]
        for ext in document_extensions:
            with self.subTest(ext=ext):
                chunks, counts = chunking._chunk_document_batch_sync(
                    [(text, {"source": f"document{ext}/sample{ext}"})],
                    chunk_size=100,
                    chunk_overlap=20,
                    use_tokens=False,
                )
                # Dispatcher should have routed to RecursiveChunker, never
                # to the slow chonkie_code path.
                self.assertNotIn("chonkie_code", counts)
                self.assertIn("chonkie_recursive", counts)
                self.assertGreaterEqual(len(chunks), 1)
                # Chunks must contain actual content, not be empty.
                for content, _meta in chunks:
                    self.assertTrue(content.strip())
                # No exception, no BinaryProcessPool crash.

    def test_looks_binary_rejects_cad_with_text_header(self) -> None:
        """CAD formats like Parasolid .x_t have an ASCII translation table
        for the first KB then binary geometry. The existing
        has_binary_content only sees the header. Verify the byte-based
        ``_looks_binary`` helper is the single source of truth and that
        it flags obvious binary samples while leaving real text alone.
        """
        from ragtime.indexer.file_utils import _looks_binary, has_binary_content

        # Pure ASCII translation table that mimics .x_t / .stl headers.
        ascii_header = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" * 100
        self.assertFalse(_looks_binary(ascii_header))

        # NUL byte anywhere -> binary.
        self.assertTrue(_looks_binary(ascii_header + b"\x00abc"))

        # Heavy non-printable control chars (single bytes are valid UTF-8, so
        # we need bytes that don't form a valid UTF-8 sequence at all).
        self.assertTrue(_looks_binary(b"\x80\x81\x82\x83" * 500))

        # UTF-8 BOM is recognised as text.
        self.assertFalse(_looks_binary(b"\xef\xbb\xbfhello world"))

        # Reuse: the file-level helper delegates to the bytes helper.
        tmp = self._tmp_bytes(ascii_header)
        try:
            self.assertFalse(has_binary_content(tmp))
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _tmp_bytes(data: bytes):
        """Helper: write ``data`` to a temp Path for file-based binary checks."""
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkstemp(suffix=".bin")[1])
        path.write_bytes(data)
        return path

    def test_legacy_get_process_pool_returns_shared_pool(self) -> None:
        executor_a = chunking._get_process_pool()
        executor_b = chunking._get_process_pool()
        self.assertIs(executor_a, executor_b)
        shared_pool = pool_manager.get(SHARED_CHUNKING_POOL_KEY)
        self.assertIsNotNone(shared_pool)
        assert shared_pool is not None
        self.assertIs(shared_pool.executor, executor_a)

    def test_unknown_extension_always_bypasses_tree_sitter(self) -> None:
        """Unmapped extensions (e.g. CAD .stp/.eps, PostScript .eps) have no
        tree-sitter grammar and chonkie's auto-detect picks the wrong
        language. RecursiveChunker is both faster and more accurate.
        """
        # PostScript-style text content; .eps has no tree-sitter grammar.
        text = "%!PS-Adobe-3.0 EPSF-3.0\n" + "x" * 100  # small but > 50KB allowed

        chunks, counts = chunking._chunk_document_batch_sync(
            [(text, {"source": "DATA/Users/matt/ODEV/Projects/sample.eps"})],
            chunk_size=1000,
            chunk_overlap=200,
            use_tokens=False,
        )

        # Dispatcher must have skipped the slow code-chunker path.
        self.assertNotIn("chonkie_code", counts)
        self.assertIn("chonkie_recursive", counts)
        # RecursiveChunker should produce at least one chunk.
        self.assertGreater(len(chunks), 0)


class _HangingFuture:
    """Stand-in for a chunking future whose worker never returns."""

    def __init__(self) -> None:
        self._done = False
        self._cancelled = False
        self._result: Any = None
        self._exception: BaseException | None = None

    def cancel(self) -> bool:
        if self._done:
            return False
        self._cancelled = True
        self._done = True
        return True

    def done(self) -> bool:
        return self._done

    def result(self) -> Any:
        if self._exception is not None:
            raise self._exception
        return self._result

    def add_done_callback(self, fn) -> None:
        if self._done:
            fn(self)

    def remove_done_callback(self, fn) -> None:
        return None

    def __await__(self):
        # Should never be awaited: the detector must surface ChunkingPoolError
        # before asyncio waits on this future.
        raise AssertionError("hanging future was awaited")


class _ReadyFuture:
    """Pre-completed stand-in for a chunking future."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def cancel(self) -> bool:
        return False

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._value

    def add_done_callback(self, fn) -> None:
        fn(self)

    def remove_done_callback(self, fn) -> None:
        return None

    def __await__(self):
        raise AssertionError("ready future was awaited")


class PoolDestructionDetectionTests(unittest.IsolatedAsyncioTestCase):
    """Validate that awaiting futures on a closed pool fails fast."""

    async def test_await_pool_futures_raises_when_pool_closed_mid_flight(self) -> None:
        executor = _FakeExecutor(processes=[])
        pool = ChunkingPool(key="job-detect", executor=executor, max_workers=1)
        futures = [_HangingFuture()]

        async def release_after_a_tick() -> None:
            await asyncio.sleep(0.05)
            pool.closed.set()

        asyncio.create_task(release_after_a_tick())

        with self.assertRaises(ChunkingPoolError):
            await chunking._await_pool_futures_or_closed(futures, pool)

        # Pending futures must be cancelled; closed event prevents indefinite hang.
        self.assertTrue(futures[0]._cancelled)

    async def test_await_pool_futures_returns_results_when_pool_alive(self) -> None:
        executor = _FakeExecutor(processes=[])
        pool = ChunkingPool(key="job-ok", executor=executor, max_workers=1)

        ready_a = _ReadyFuture("alpha")
        ready_b = _ReadyFuture("beta")

        results = await chunking._await_pool_futures_or_closed([ready_a, ready_b], pool)
        self.assertEqual(results, ["alpha", "beta"])


class PerJobChunkingTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end: chunk_documents_parallel with explicit pool_key isolation."""

    def setUp(self) -> None:
        pool_manager.shutdown_all(terminate_workers=False)

    def tearDown(self) -> None:
        pool_manager.shutdown_all(terminate_workers=False)

    @staticmethod
    def _install_inproc_pool(key: str) -> ChunkingPool:
        """Insert a ChunkingPool backed by a ThreadPoolExecutor (in-process).

        ThreadPoolExecutor avoids the spawn/pickle round-trip so tests can
        observe routing and termination without forking real workers.
        """
        from concurrent.futures import ThreadPoolExecutor

        pool = ChunkingPool(key=key, executor=ThreadPoolExecutor(max_workers=1), max_workers=1)
        pool_manager._pools[key] = pool  # type: ignore[attr-defined]
        return pool

    async def test_pool_key_routes_to_per_job_pool(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        pool = self._install_inproc_pool("job-per-job")

        docs = [Document(page_content=f"hello {i}", metadata={"source": f"f{i}.txt"}) for i in range(3)]
        results = await chunking.chunk_documents_parallel(
            documents=docs,
            chunk_size=50,
            chunk_overlap=0,
            use_tokens=False,
            batch_size=2,
            pool_key="job-per-job",
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(sorted(d.page_content for d in results), ["hello 0", "hello 1", "hello 2"])
        # The job's pool was used (not the shared pool).
        self.assertIs(pool_manager.get("job-per-job"), pool)
        pool.executor.shutdown(wait=True)

    async def test_releasing_one_job_pool_does_not_affect_another(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        executor_a = ThreadPoolExecutor(max_workers=1)
        executor_b = ThreadPoolExecutor(max_workers=1)

        pool_a = ChunkingPool(key="job-a", executor=executor_a, max_workers=1)
        pool_b = ChunkingPool(key="job-b", executor=executor_b, max_workers=1)
        pool_manager._pools["job-a"] = pool_a  # type: ignore[attr-defined]
        pool_manager._pools["job-b"] = pool_b  # type: ignore[attr-defined]

        pool_manager.release("job-a", terminate_workers=False)

        self.assertIsNone(pool_manager.get("job-a"))
        survivor = pool_manager.get("job-b")
        self.assertIsNotNone(survivor)
        assert survivor is not None
        self.assertIs(survivor.executor, executor_b)
        self.assertFalse(survivor.is_closed())
        executor_b.shutdown(wait=True)

    async def test_chunking_raises_pool_error_when_pool_closed_before_run(self) -> None:
        pool = self._install_inproc_pool("closed-up-front")
        pool.closed.set()

        docs = [Document(page_content="x", metadata={"source": "a.txt"})]
        with self.assertRaises(ChunkingPoolError):
            await chunking.chunk_documents_parallel(
                documents=docs,
                chunk_size=50,
                chunk_overlap=0,
                use_tokens=False,
                batch_size=10,
                pool_key="closed-up-front",
            )
        pool.executor.shutdown(wait=True)

    async def test_individual_retry_recycles_pool_after_broken_worker(self) -> None:
        loop = asyncio.get_running_loop()
        original_pool = ChunkingPool(key="job-retry", executor=mock.Mock(), max_workers=1)
        recycled_pool = ChunkingPool(key="job-retry", executor=mock.Mock(), max_workers=1)
        executor_calls = []

        def fake_run_in_executor(executor, func, batch, chunk_size, chunk_overlap, use_tokens):
            executor_calls.append(executor)
            future = loop.create_future()
            if len(executor_calls) == 1:
                future.set_exception(chunking.BrokenProcessPool("boom"))
            else:
                content, metadata = batch[0]
                future.set_result(
                    (
                        [(content, {**metadata, "chunker": "worker"})],
                        {"worker": 1},
                    )
                )
            return future

        def fake_release(key, terminate_workers=True):
            original_pool.closed.set()
            return True

        with (
            mock.patch.object(loop, "run_in_executor", side_effect=fake_run_in_executor),
            mock.patch.object(pool_manager, "release", side_effect=fake_release) as release_mock,
            mock.patch.object(pool_manager, "get_or_create", return_value=recycled_pool) as get_or_create_mock,
            mock.patch.object(
                chunking,
                "_chunk_document_batch_recursive_sync",
                return_value=(
                    [("bad", {"source": "bad.txt", "chunker": "fallback"})],
                    {"fallback": 1},
                ),
            ),
        ):
            results = await chunking._retry_chunk_documents_individually(
                [
                    ("bad", {"source": "bad.txt"}),
                    ("good", {"source": "good.txt"}),
                ],
                chunk_size=50,
                chunk_overlap=0,
                use_tokens=False,
                pool=original_pool,
            )

        self.assertEqual([result[0][0][1]["chunker"] for result in results], ["fallback", "worker"])
        self.assertEqual(executor_calls, [original_pool.executor, recycled_pool.executor])
        release_mock.assert_called_once_with("job-retry", terminate_workers=True)
        get_or_create_mock.assert_called_once_with("job-retry", 1)


if __name__ == "__main__":
    unittest.main()
