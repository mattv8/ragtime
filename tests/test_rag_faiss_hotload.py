import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from ragtime.indexer.models import IndexStatus
from ragtime.indexer.service import IndexerService
from ragtime.rag.components import RAGComponents
from ragtime.rag.components import rag as global_rag


class FakeEmbeddings:
    def embed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeFaissIndex:
    def __init__(self) -> None:
        self.index = SimpleNamespace(d=3)

    def as_retriever(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


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

    async def test_git_job_hot_loads_before_completed_status_is_published(self):
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
                patch("ragtime.indexer.service.shutdown_process_pool"),
            ):
                await service._process_git(cast(Any, job))

            self.assertEqual(order[-2:], ["hot-load", "update:completed"])


if __name__ == "__main__":
    unittest.main()
