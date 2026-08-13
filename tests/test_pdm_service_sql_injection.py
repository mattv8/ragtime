"""
Tests verifying that PDM service methods use parameterized queries instead of
f-string interpolation for SQL execution, preventing SQL injection.
"""

import sys
import types
import unittest
from hashlib import sha256
from typing import Any
from unittest import mock

# Stub heavy optional dependencies before importing pdm_service
for _mod in [
    "langchain_core",
    "langchain_core.embeddings",
    "langchain_core.messages",
    "pydantic",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Provide a minimal SecretStr stub so vector_utils imports cleanly
_pydantic = sys.modules.setdefault("pydantic", types.ModuleType("pydantic"))
if not hasattr(_pydantic, "SecretStr"):
    setattr(_pydantic, "SecretStr", str)

# Stub openrouter used by vector_utils
_openrouter_mod = types.ModuleType("ragtime.core.openrouter")
sys.modules.setdefault("ragtime.core.openrouter", _openrouter_mod)

# Stub copilot_auth (imported transitively)
_copilot_auth = types.ModuleType("ragtime.core.copilot_auth")


async def _noop_ensure_copilot(*_a: Any, **_kw: Any) -> None:
    return None


setattr(_copilot_auth, "ensure_copilot_token_fresh", _noop_ensure_copilot)
sys.modules.setdefault("ragtime.core.copilot_auth", _copilot_auth)

from ragtime.indexer.pdm_service import PdmIndexerService, _build_pdm_extension_filter, search_pdm_index  # noqa: E402


class _FakeDb:
    """Minimal DB stub that records calls to execute_raw and query_raw."""

    def __init__(self) -> None:
        self.execute_raw_calls: list[tuple[Any, ...]] = []
        self.query_raw_calls: list[tuple[Any, ...]] = []
        self.pdmindexjob = _FakePdmIndexJobDelegate()

    async def execute_raw(self, query: str, *params: Any) -> int:
        self.execute_raw_calls.append((query, *params))
        return 1

    async def query_raw(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.query_raw_calls.append((query, *params))
        return [{"count": 0}]


class _FakePdmIndexJobDelegate:
    async def delete_many(self, **_kwargs: Any) -> None:
        pass


class _FakeToolConfig:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakePdmJob:
    def __init__(self, index_name: str) -> None:
        self.index_name = index_name
        self.total_chunks = 0
        self.processed_chunks = 0


class _FakePdmDocument:
    document_id = 42
    document_type = "part'; DROP TABLE pdm_embeddings; --"
    filename = "file'; DROP TABLE pdm_document_metadata; --.sldprt"
    part_number = "PN-1"
    folder_path = "/vault/path"
    revision_no = 7
    variables = {"k": "v"}

    def to_embedding_text(self) -> str:
        return "document text"

    def compute_metadata_hash(self) -> str:
        return sha256(b"metadata").hexdigest()


class _HashBatchingFakeDb(_FakeDb):
    def __init__(self, stored_hashes: dict[int, str]) -> None:
        super().__init__()
        self.stored_hashes = stored_hashes

    async def query_raw(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.query_raw_calls.append((query, *params))

        if "metadata_hash" not in query:
            return []

        if len(params) >= 3:
            document_ids = [int(doc_id) for doc_id in params[1:]]
        elif len(params) == 2 and isinstance(params[1], (list, tuple, set)):
            document_ids = [int(doc_id) for doc_id in params[1]]
        elif len(params) == 2:
            document_ids = [int(params[1])]
        else:
            document_ids = []

        return [{"document_id": doc_id, "metadata_hash": self.stored_hashes[doc_id]} for doc_id in document_ids if doc_id in self.stored_hashes]


class _FakeHashBatchDoc:
    def __init__(self, document_id: int, metadata_hash: str) -> None:
        self.document_id = document_id
        self._metadata_hash = metadata_hash

    def compute_metadata_hash(self) -> str:
        return self._metadata_hash


class PdmServiceSqlInjectionTests(unittest.IsolatedAsyncioTestCase):
    """Verify that PDM cleanup methods pass index_name as a bound parameter."""

    def _make_service(self) -> PdmIndexerService:
        return PdmIndexerService()

    async def test_delete_index_uses_parameterized_query(self) -> None:
        """delete_index must not interpolate index_name into the SQL string."""
        service = self._make_service()
        fake_db = _FakeDb()
        malicious_name = "pdm_x' OR '1'='1"
        fake_tool_config = _FakeToolConfig(name="x' OR '1'='1")

        with (
            mock.patch("ragtime.indexer.pdm_service.get_db", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.pdm_service.repository.get_tool_config",
                return_value=fake_tool_config,
            ),
        ):
            success, _msg = await service.delete_index("tool-id-1")

        self.assertTrue(success)

        # Every execute_raw call must use $1 placeholder, not inline the name
        for call in fake_db.execute_raw_calls:
            query: str = call[0]
            self.assertIn("$1", query, f"Query missing $1 placeholder: {query!r}")
            self.assertNotIn(malicious_name, query, f"index_name was interpolated into query: {query!r}")
            # The actual value must be passed as a separate argument
            self.assertGreater(len(call), 1, f"No parameter passed alongside query: {query!r}")

    async def test_clear_embeddings_uses_parameterized_query(self) -> None:
        """_clear_embeddings must pass index_name as a bound parameter."""
        service = self._make_service()
        fake_db = _FakeDb()
        index_name = "pdm_test'; DROP TABLE pdm_embeddings; --"

        with mock.patch("ragtime.indexer.pdm_service.get_db", return_value=fake_db):
            await service._clear_embeddings(index_name)

        self.assertEqual(len(fake_db.execute_raw_calls), 2)
        for call in fake_db.execute_raw_calls:
            query: str = call[0]
            self.assertIn("$1", query, f"Query missing $1 placeholder: {query!r}")
            self.assertNotIn(index_name, query, f"index_name was interpolated into query: {query!r}")
            self.assertEqual(call[1], index_name, "index_name not passed as bound parameter")

    async def test_get_embedding_count_uses_parameterized_query(self) -> None:
        """get_embedding_count must pass index_name as a bound parameter."""
        service = self._make_service()
        fake_db = _FakeDb()

        with mock.patch("ragtime.indexer.pdm_service.get_db", return_value=fake_db):
            count = await service.get_embedding_count("tool-id-2", tool_name="myname")

        self.assertEqual(count, 0)
        self.assertGreater(len(fake_db.query_raw_calls), 0)
        for call in fake_db.query_raw_calls:
            query: str = call[0]
            self.assertIn("$1", query, f"Query missing $1 placeholder: {query!r}")
            self.assertGreater(len(call), 1, "index_name not passed as bound parameter")

    async def test_get_document_count_uses_parameterized_query(self) -> None:
        """get_document_count must pass index_name as a bound parameter."""
        service = self._make_service()
        fake_db = _FakeDb()

        with mock.patch("ragtime.indexer.pdm_service.get_db", return_value=fake_db):
            count = await service.get_document_count("tool-id-3")

        self.assertEqual(count, 0)
        self.assertGreater(len(fake_db.query_raw_calls), 0)
        for call in fake_db.query_raw_calls:
            query: str = call[0]
            self.assertIn("$1", query, f"Query missing $1 placeholder: {query!r}")
            self.assertGreater(len(call), 1, "index_name not passed as bound parameter")

    async def test_get_stored_hash_uses_parameterized_query(self) -> None:
        """_get_stored_hash must pass index_name and document_id as bound parameters."""
        service = self._make_service()
        fake_db = _FakeDb()

        with mock.patch("ragtime.indexer.pdm_service.get_db", return_value=fake_db):
            result = await service._get_stored_hash("pdm_myindex", 42)

        self.assertIsNone(result)
        self.assertEqual(len(fake_db.query_raw_calls), 1)
        call = fake_db.query_raw_calls[0]
        query: str = call[0]
        self.assertIn("$1", query)
        self.assertIn("$2", query)
        self.assertEqual(call[1], "pdm_myindex")
        self.assertEqual(call[2], 42)
        # Neither value should appear literally in the query string
        self.assertNotIn("pdm_myindex", query)
        self.assertNotIn("42", query)

    async def test_process_batch_uses_parameterized_upserts(self) -> None:
        """_process_batch must not interpolate job or document values into SQL."""
        service = self._make_service()
        fake_db = _FakeDb()
        malicious_index_name = "pdm_x'; DROP TABLE pdm_embeddings; --"
        document = _FakePdmDocument()

        with (
            mock.patch("ragtime.indexer.pdm_service.get_db", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.pdm_service.embed_documents_subbatched",
                return_value=[[0.1, 0.2, 0.3]],
            ),
        ):
            await service._process_batch(
                _FakePdmJob(malicious_index_name),  # type: ignore[arg-type]
                [document],  # type: ignore[list-item]
                embeddings=object(),
            )

        self.assertEqual(len(fake_db.execute_raw_calls), 2)
        for call in fake_db.execute_raw_calls:
            query: str = call[0]
            self.assertIn("$1", query)
            self.assertNotIn(malicious_index_name, query)
            self.assertNotIn(document.filename, query)
            self.assertNotIn(document.document_type, query)
            self.assertGreater(len(call), 1)

        embedding_call = fake_db.execute_raw_calls[0]
        self.assertEqual(embedding_call[1], malicious_index_name)
        self.assertEqual(embedding_call[3], document.document_type)
        self.assertEqual(embedding_call[6], document.filename)

        metadata_call = fake_db.execute_raw_calls[1]
        self.assertEqual(metadata_call[1], malicious_index_name)
        self.assertEqual(metadata_call[3], document.filename)

    async def test_extension_filter_rejects_sql_metacharacters(self) -> None:
        """Configured file extensions must be strictly validated before SQL use."""
        with self.assertRaises(ValueError):
            _build_pdm_extension_filter(["SLDPRT' OR 1=1 --"], "Filename")

    async def test_extension_filter_allows_expected_extensions(self) -> None:
        result = _build_pdm_extension_filter(["SLDPRT", ".SLDASM"], "d.Filename")
        self.assertEqual(result, "d.Filename LIKE '%SLDPRT' OR d.Filename LIKE '%.SLDASM'")

    async def test_extension_filter_empty_list_matches_existing_all_behavior(self) -> None:
        self.assertEqual(_build_pdm_extension_filter([], "Filename"), "1=1")

    async def test_search_pdm_index_rejects_malicious_document_type(self) -> None:
        """document_type must be allowlisted before it reaches extra_where SQL."""
        result = await search_pdm_index(
            index_name="pdm_index",
            query="gear",
            document_type="SLDPRT' OR '1'='1",
        )
        self.assertIn("Invalid document_type", result)

    async def test_search_pdm_index_uses_allowlisted_document_type_filter(self) -> None:
        """Valid document_type filters should produce a constrained extra_where."""

        class _FakeEmbeddings:
            def embed_documents(self, _texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3]]

            async def aembed_query(self, _text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

        captured: dict[str, Any] = {}

        async def _fake_search_pgvector_embeddings(**kwargs: Any) -> list[dict[str, Any]]:
            captured.update(kwargs)
            return []

        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", return_value={}),
            mock.patch(
                "ragtime.indexer.pdm_service.get_embeddings_model",
                return_value=_FakeEmbeddings(),
            ),
            mock.patch(
                "ragtime.indexer.pdm_service.search_pgvector_embeddings",
                side_effect=_fake_search_pgvector_embeddings,
            ),
        ):
            await search_pdm_index(
                index_name="pdm_index",
                query="gear",
                document_type="sldprt",
            )

        self.assertEqual(captured.get("extra_where"), "document_type = 'SLDPRT'")

    async def test_process_index_batches_stored_hash_reads_per_extraction_batch(self) -> None:
        """_process_index should fetch stored hashes once per extracted batch."""

        class _FakeEmbeddings:
            def embed_documents(self, _texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3]]

            async def aembed_documents(self, _texts: list[str]) -> list[list[float]]:
                return self.embed_documents(_texts)

        class _FakeSettings:
            embedding_dimension = 3
            embedding_config_hash = "config-hash"

            def get_embedding_config_hash(self) -> str:
                return "config-hash"

        async def _fake_extract_documents_batched(**_kwargs: Any):
            yield [
                _FakeHashBatchDoc(101, "same-hash"),
                _FakeHashBatchDoc(202, "new-hash"),
            ]

        service = self._make_service()
        fake_db = _HashBatchingFakeDb({101: "same-hash", 202: "old-hash"})
        processed_docs: list[list[int]] = []
        job = mock.Mock(
            id="job-1",
            index_name="pdm_batch_test",
            status=None,
            started_at=None,
            current_step="",
            total_documents=0,
            skipped_documents=0,
            processed_documents=0,
            extracted_documents=0,
            total_chunks=0,
            processed_chunks=0,
            completed_at=None,
            error_message=None,
        )

        async def _record_process_batch(_job: Any, documents: list[Any], _embeddings: Any) -> None:
            processed_docs.append([doc.document_id for doc in documents])

        with (
            mock.patch("ragtime.indexer.pdm_service.get_db", return_value=fake_db),
            mock.patch.object(service, "_ensure_pgvector", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={})),
            mock.patch(
                "ragtime.indexer.pdm_service.repository.get_settings",
                mock.AsyncMock(return_value=_FakeSettings()),
            ),
            mock.patch.object(service, "_get_embeddings", mock.AsyncMock(return_value=_FakeEmbeddings())),
            mock.patch.object(service, "_ensure_embedding_column", mock.AsyncMock(return_value=True)),
            mock.patch.object(service, "_count_documents", mock.AsyncMock(return_value=2)),
            mock.patch.object(service, "_get_variable_map", mock.AsyncMock(return_value={})),
            mock.patch.object(service, "extract_documents_batched", side_effect=_fake_extract_documents_batched),
            mock.patch.object(service, "_process_batch", side_effect=_record_process_batch),
            mock.patch.object(service, "_update_job", mock.AsyncMock()),
        ):
            await service._process_index(job, {"host": "pdm.example", "database": "vault", "user": "u", "password": "p"}, False)

        self.assertEqual(processed_docs, [[202]])
        self.assertEqual(job.skipped_documents, 1)
        self.assertEqual(job.processed_documents, 1)
        self.assertEqual(len(fake_db.query_raw_calls), 1)
        query_call = fake_db.query_raw_calls[0]
        self.assertEqual(query_call[1], "pdm_batch_test")
        self.assertGreater(len(query_call), 3)
