import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer.embedding_errors import EmbeddingFailureKind, EmbeddingOperationError
from ragtime.indexer.pdm_service import search_pdm_index


class PdmEmbeddingSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_pdm_index_returns_typed_embedding_failure(self) -> None:
        settings = {
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text:latest",
        }
        embeddings = SimpleNamespace(
            aembed_query=mock.AsyncMock(
                side_effect=EmbeddingOperationError(
                    kind=EmbeddingFailureKind.CONNECTION,
                    provider="ollama",
                    model="nomic-embed-text:latest",
                    operation="query",
                    endpoint="http://private-host:11434",
                    cause=RuntimeError("socket secret detail"),
                )
            )
        )

        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch("ragtime.indexer.pdm_service.get_embeddings_model", new=mock.AsyncMock(return_value=embeddings)),
            mock.patch("ragtime.indexer.pdm_service.search_pgvector_embeddings", new=mock.AsyncMock()) as search_pgvector,
        ):
            result = await search_pdm_index(query="gear", index_name="pdm_index")

        self.assertEqual(
            result,
            "Error: Could not connect to the Ollama embedding server for model 'nomic-embed-text:latest'. Verify the service is running and reachable.",
        )
        self.assertNotIn("private-host", result)
        self.assertNotIn("socket secret detail", result)
        search_pgvector.assert_not_awaited()

    async def test_search_pdm_index_returns_configuration_failure_when_embeddings_missing(self) -> None:
        settings = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
        }

        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch("ragtime.indexer.pdm_service.get_embeddings_model", new=mock.AsyncMock(return_value=None)),
            mock.patch("ragtime.indexer.pdm_service.search_pgvector_embeddings", new=mock.AsyncMock()) as search_pgvector,
        ):
            result = await search_pdm_index(query="gear", index_name="pdm_index")

        self.assertEqual(
            result,
            "Error: The OpenAI embedding configuration for model 'text-embedding-3-large' is invalid or unauthorized. Verify the embedding provider settings and credentials.",
        )
        search_pgvector.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
