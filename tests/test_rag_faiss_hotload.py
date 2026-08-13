import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.documents import Document

from ragtime.indexer.embedding_errors import EmbeddingFailureKind, EmbeddingOperationError
from ragtime.indexer.models import IndexJobPhase, IndexStatus
from ragtime.indexer.service import IndexerService
from ragtime.rag.components import RAGComponents
from ragtime.rag.components import rag as global_rag


class FakeEmbeddings:
    def embed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def aembed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeFaissIndex:
    def __init__(self) -> None:
        self.index = SimpleNamespace(d=3)

    def as_retriever(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    def similarity_search(self, query: str, k: int):
        return [Document(page_content=f"match for {query}", metadata={"source": "src.py"})][:k]

    def max_marginal_relevance_search(self, query: str, k: int, fetch_k: int, lambda_mult: float):
        return self.similarity_search(query, k)

    def similarity_search_by_vector(self, embedding: list[float], k: int):
        return [Document(page_content=f"match for {embedding}", metadata={"source": "src.py"})][:k]

    def max_marginal_relevance_search_by_vector(self, embedding: list[float], k: int, fetch_k: int, lambda_mult: float):
        return self.similarity_search_by_vector(embedding, k)


class RagFaissHotLoadTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_faiss_index_from_metadata_loads_index_before_returning(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "hot-index"
            index_path.mkdir()

            rag = RAGComponents()
            rag._core_ready = True
            rag._app_settings = {
                "aggregate_search": True,
                "search_results_k": 5,
                "search_use_mmr": True,
                "search_mmr_lambda": 0.5,
                "tool_output_mode": "default",
            }
            rag._tool_configs = []
            rag._embedding_model = FakeEmbeddings()
            rag._create_agent = AsyncMock()

            metadata = {
                "name": "hot-index",
                "path": str(index_path),
                "description": "",
                "enabled": True,
                "search_weight": 1.0,
                "document_count": 1,
                "chunk_count": 1,
                "source_type": "git",
                "size_bytes": 1024,
                "embedding_dimension": 3,
            }
            rag._load_index_metadata = AsyncMock(return_value=[metadata])

            with (
                patch(
                    "ragtime.rag.components.get_app_settings",
                    new=AsyncMock(return_value=rag._app_settings),
                ),
                patch(
                    "ragtime.rag.components.get_tool_configs",
                    new=AsyncMock(return_value=[]),
                ),
                patch(
                    "ragtime.rag.components.FAISS.load_local",
                    return_value=FakeFaissIndex(),
                ) as load_local,
                patch(
                    "ragtime.rag.components.repository.update_index_memory_stats",
                    new=AsyncMock(return_value=True),
                ),
            ):
                loaded = await rag.load_faiss_index_from_metadata("hot-index")

            self.assertTrue(loaded)
            self.assertIn("hot-index", rag.faiss_dbs)
            self.assertIn("hot-index", rag.retrievers)
            self.assertEqual(rag._index_details["hot-index"]["status"], "loaded")
            load_local.assert_called_once()
            rag._create_agent.assert_awaited_once()

    async def test_hot_loaded_retriever_invokes_faiss_search_coordinator(self):
        rag = RAGComponents()
        rag._app_settings = {
            "search_results_k": 2,
            "search_use_mmr": False,
            "search_mmr_lambda": 0.5,
        }
        retriever = rag._create_retriever_from_faiss(FakeFaissIndex(), "hot-index")

        async def _run(index_name, operation, *args, **kwargs):
            self.assertEqual(index_name, "hot-index")
            return operation(*args, **kwargs)

        with patch(
            "ragtime.rag.components.faiss_search_coordinator.run",
            new=AsyncMock(side_effect=_run),
        ) as coordinator_run:
            docs = await retriever.ainvoke("needle")

        self.assertEqual(docs[0].metadata["source"], "src.py")
        coordinator_run.assert_awaited_once()

    async def test_hot_loaded_retriever_surfaces_sanitized_typed_failure(self):
        rag = RAGComponents()
        rag._app_settings = {
            "search_results_k": 2,
            "search_use_mmr": False,
            "search_mmr_lambda": 0.5,
        }

        class FailingFaissIndex(FakeFaissIndex):
            def similarity_search(self, query: str, k: int):
                raise EmbeddingOperationError(
                    kind=EmbeddingFailureKind.CONNECTION,
                    provider="ollama",
                    model="nomic-embed-text:latest",
                    operation="query",
                    endpoint="http://private-embedding-host:11434",
                )

        retriever = rag._create_retriever_from_faiss(FailingFaissIndex(), "hot-index")

        async def _run(index_name, operation, *args, **kwargs):
            self.assertEqual(index_name, "hot-index")
            return operation(*args, **kwargs)

        with patch(
            "ragtime.rag.components.faiss_search_coordinator.run",
            new=AsyncMock(side_effect=_run),
        ):
            with self.assertRaises(EmbeddingOperationError) as exc_info:
                await retriever.ainvoke("needle")

        self.assertIn("Could not connect to the Ollama embedding server", str(exc_info.exception))
        self.assertNotIn("private-embedding-host", str(exc_info.exception))

    async def test_get_embedding_model_delegates_to_shared_factory(self):
        rag = RAGComponents()
        rag._app_settings = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "openai_api_key": "secret",
        }
        sentinel = object()

        with patch("ragtime.rag.components.get_embeddings_model", new=AsyncMock(return_value=sentinel)) as get_embeddings_model:
            result = await rag._get_embedding_model()

        self.assertIs(result, sentinel)
        get_embeddings_model.assert_awaited_once_with(rag._app_settings, return_none_on_error=True)

    async def test_hot_load_dimension_probe_logs_sanitized_embedding_failure_and_uses_tracked_dimension(self):
        rag = RAGComponents()
        rag._core_ready = True
        rag._app_settings = {
            "aggregate_search": True,
            "search_results_k": 5,
            "search_use_mmr": True,
            "search_mmr_lambda": 0.5,
            "tool_output_mode": "default",
            "embedding_dimension": 3,
        }
        rag._tool_configs = []
        rag._create_agent = AsyncMock()
        rag._embedding_model = SimpleNamespace(
            aembed_query=AsyncMock(
                side_effect=EmbeddingOperationError(
                    kind=EmbeddingFailureKind.CONNECTION,
                    provider="ollama",
                    model="nomic-embed-text:latest",
                    operation="query",
                    endpoint="http://private-embedding-host:11434",
                )
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "hot-index"
            index_path.mkdir()
            metadata = {
                "name": "hot-index",
                "path": str(index_path),
                "description": "",
                "enabled": True,
                "search_weight": 1.0,
                "document_count": 1,
                "chunk_count": 1,
                "source_type": "git",
                "size_bytes": 1024,
                "embedding_dimension": 3,
            }
            rag._load_index_metadata = AsyncMock(return_value=[metadata])

            with (
                patch("ragtime.rag.components.get_app_settings", new=AsyncMock(return_value=rag._app_settings)),
                patch("ragtime.rag.components.get_tool_configs", new=AsyncMock(return_value=[])),
                patch("ragtime.rag.components.FAISS.load_local", return_value=FakeFaissIndex()),
                patch("ragtime.rag.components.repository.update_index_memory_stats", new=AsyncMock(return_value=True)),
                self.assertLogs("ragtime.rag.components", level="WARNING") as logs,
            ):
                loaded = await rag.load_faiss_index_from_metadata("hot-index")

        self.assertTrue(loaded)
        joined = "\n".join(logs.output)
        self.assertIn("Could not connect to the Ollama embedding server for model 'nomic-embed-text:latest'", joined)
        self.assertNotIn("private-embedding-host", joined)
        self.assertIn("Using tracked dimension: 3", joined)

    async def test_completed_index_job_hot_loads_specific_index(self):
        with tempfile.TemporaryDirectory() as directory:
            service = IndexerService(index_base_path=directory)
            load_index = AsyncMock(return_value=True)
            job = SimpleNamespace(status=IndexStatus.COMPLETED, name="hot-index")

            with (
                patch.object(
                    global_rag,
                    "load_faiss_index_from_metadata",
                    load_index,
                ),
                patch("ragtime.indexer.service.invalidate_settings_cache"),
            ):
                await service._maybe_reinitialize_rag(cast(Any, job))

            load_index.assert_awaited_once_with("hot-index")

    async def test_git_job_hot_loads_before_completed_status_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = IndexerService(index_base_path=directory)
            order: list[str] = []

            job = SimpleNamespace(
                id="job-1",
                name="hot-index",
                status=IndexStatus.PENDING,
                started_at=None,
                completed_at=None,
                error_message=None,
                git_url="https://example.com/repo.git",
                git_branch="main",
                git_token=None,
            )

            async def update_job(current_job):
                order.append(f"update:{current_job.status.value}")
                return current_job

            async def hot_load(_current_job):
                order.append("hot-load")

            with (
                patch(
                    "ragtime.indexer.service.repository.update_job",
                    new=update_job,
                ),
                patch.object(service, "_clone_git_repo", new=AsyncMock()),
                patch.object(service, "_create_faiss_index", new=AsyncMock()),
                patch.object(service, "_maybe_reinitialize_rag", new=hot_load),
                patch("ragtime.indexer.service.pool_manager"),
            ):
                await service._process_git(cast(Any, job))

            self.assertEqual(order[-2:], ["hot-load", "update:completed"])
            self.assertEqual(job.phase, IndexJobPhase.COMPLETED)

    async def test_failed_git_job_sets_completed_at(self):
        with tempfile.TemporaryDirectory() as directory:
            service = IndexerService(index_base_path=directory)

            job = SimpleNamespace(
                id="job-1",
                name="failed-index",
                status=IndexStatus.PENDING,
                started_at=None,
                completed_at=None,
                error_message=None,
                git_url="https://example.com/repo.git",
                git_branch="main",
                git_token=None,
            )

            async def update_job(current_job):
                return current_job

            with (
                patch(
                    "ragtime.indexer.service.repository.update_job",
                    new=update_job,
                ),
                patch.object(service, "_clone_git_repo", new=AsyncMock()),
                patch.object(service, "_create_faiss_index", new=AsyncMock(side_effect=RuntimeError("boom"))),
                patch.object(service, "_cleanup_failed_index_metadata", new=AsyncMock()),
                patch.object(service, "_maybe_reinitialize_rag", new=AsyncMock()),
                patch("ragtime.indexer.service.pool_manager"),
            ):
                await service._process_git(cast(Any, job))

            self.assertEqual(job.status, IndexStatus.FAILED)
            self.assertEqual(job.phase, IndexJobPhase.FAILED)
            self.assertEqual(job.error_message, "boom")
            self.assertIsNotNone(job.completed_at)

    async def test_rate_limit_detection_and_retry_delay_use_raw_cause_while_wrapper_stays_sanitized(self) -> None:
        service = IndexerService()
        request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        response = httpx.Response(429, headers={"retry-after": "7"}, request=request)

        class _RateLimitCause(RuntimeError):
            def __init__(self) -> None:
                super().__init__("rate limit exceeded; try again in 1200ms")
                self.status_code = 429
                self.response = response

        cause = _RateLimitCause()
        wrapped = EmbeddingOperationError(
            kind=EmbeddingFailureKind.PROVIDER,
            provider="openai",
            model="text-embedding-3-small",
            operation="documents",
            endpoint="https://private-endpoint/v1",
            cause=cause,
        )

        self.assertTrue(service._is_rate_limit_error(wrapped))
        self.assertEqual(service._retry_delay_seconds(wrapped, attempt=0), 7.0)
        self.assertNotIn("private-endpoint", str(wrapped))
        self.assertNotIn("rate limit exceeded", str(wrapped))


if __name__ == "__main__":
    unittest.main()
