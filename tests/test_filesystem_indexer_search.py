import unittest
from unittest import mock

from ragtime.indexer.embedding_errors import EmbeddingFailureKind, EmbeddingOperationError
from ragtime.tools.filesystem_indexer import search_filesystem_index


class FilesystemIndexerSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_filesystem_index_returns_typed_embedding_failure(self) -> None:
        settings = {
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text:latest",
        }
        error = EmbeddingOperationError(
            kind=EmbeddingFailureKind.CONNECTION,
            provider="ollama",
            model="nomic-embed-text:latest",
            operation="query",
            endpoint="http://private-host:11434",
            cause=RuntimeError("socket secret detail"),
        )

        with (
            mock.patch("ragtime.tools.filesystem_indexer.get_app_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch("ragtime.tools.filesystem_indexer._get_query_embedding", new=mock.AsyncMock(side_effect=error)),
            mock.patch("ragtime.tools.filesystem_indexer.get_pgvector_backend") as get_pgvector_backend,
            mock.patch("ragtime.tools.filesystem_indexer.get_faiss_backend") as get_faiss_backend,
        ):
            result = await search_filesystem_index("gear")

        self.assertEqual(
            result,
            "Error: Could not connect to the Ollama embedding server for model 'nomic-embed-text:latest'. Verify the service is running and reachable.",
        )
        self.assertNotIn("private-host", result)
        self.assertNotIn("socket secret detail", result)
        get_pgvector_backend.assert_not_called()
        get_faiss_backend.assert_not_called()

    async def test_search_filesystem_index_returns_configuration_failure_when_embeddings_missing(self) -> None:
        settings = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
        }

        with (
            mock.patch("ragtime.tools.filesystem_indexer.get_app_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch("ragtime.tools.filesystem_indexer.get_embeddings_model", new=mock.AsyncMock(return_value=None)),
            mock.patch("ragtime.tools.filesystem_indexer.get_pgvector_backend") as get_pgvector_backend,
            mock.patch("ragtime.tools.filesystem_indexer.get_faiss_backend") as get_faiss_backend,
        ):
            result = await search_filesystem_index("gear")

        self.assertEqual(
            result,
            "Error: The OpenAI embedding configuration for model 'text-embedding-3-large' is invalid or unauthorized. Verify the embedding provider settings and credentials.",
        )
        get_pgvector_backend.assert_not_called()
        get_faiss_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
