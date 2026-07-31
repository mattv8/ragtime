import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document

from ragtime.indexer.vector_backends import FaissBackend


class _FakeFaissDb:
    def similarity_search_with_score_by_vector(self, query_embedding, k):
        return [
            (
                Document(
                    page_content="body",
                    metadata={"id": "doc-1", "file_path": "src.py", "chunk_index": 0},
                ),
                0.25,
            )
        ][:k]


class VectorBackendFaissCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_faiss_backend_search_routes_loaded_index_through_coordinator(self):
        backend = FaissBackend()
        backend._loaded_indexes["docs"] = _FakeFaissDb()

        async def _run(index_name, operation, *args, **kwargs):
            self.assertEqual(index_name, "docs")
            return operation(*args, **kwargs)

        with patch(
            "ragtime.indexer.vector_backends.faiss_search_coordinator.run",
            new=AsyncMock(side_effect=_run),
        ) as coordinator_run:
            results = await backend.search([0.1, 0.2], index_name="docs", max_results=1)

        self.assertEqual(results[0]["index_name"], "docs")
        self.assertEqual(results[0]["file_path"], "src.py")
        coordinator_run.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
