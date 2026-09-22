import unittest
from unittest import mock

from pydantic import ValidationError

from ragtime.indexer.embedding_errors import EmbeddingFailureKind, EmbeddingOperationError
from ragtime.tools.filesystem_indexer import (
    FilesystemSearchInput,
    create_filesystem_search_tool,
    search_filesystem_index,
)


class FilesystemIndexerSearchTests(unittest.IsolatedAsyncioTestCase):
    def test_global_and_configured_search_schemas_preserve_common_contract(self) -> None:
        configured_tool = create_filesystem_search_tool(
            "configured_search",
            "Search only the configured index.",
            index_name="fixture-index",
        )
        configured_input_schema = configured_tool.get_input_schema()
        global_schema = FilesystemSearchInput.model_json_schema()
        configured_schema = configured_input_schema.model_json_schema()

        self.assertEqual(global_schema["title"], "FilesystemSearchInput")
        self.assertEqual(configured_schema["title"], "SearchInput")
        self.assertEqual(global_schema["required"], ["query"])
        self.assertEqual(configured_schema["required"], ["query"])
        self.assertEqual(
            global_schema["properties"]["query"],
            configured_schema["properties"]["query"],
        )
        self.assertEqual(
            global_schema["properties"]["max_results"],
            configured_schema["properties"]["max_results"],
        )
        self.assertEqual(
            global_schema["properties"]["max_chars_per_result"],
            configured_schema["properties"]["max_chars_per_result"],
        )
        self.assertEqual(
            global_schema["properties"]["index_name"],
            {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Optional: specific index name to search (searches all if not specified)",
                "title": "Index Name",
            },
        )
        self.assertNotIn("index_name", configured_schema["properties"])

    def test_global_and_configured_search_inputs_validate_common_boundaries(self) -> None:
        configured_input_schema = create_filesystem_search_tool(
            "configured_search",
            "Search only the configured index.",
            index_name="fixture-index",
        ).get_input_schema()

        self.assertEqual(
            FilesystemSearchInput(query="gear").model_dump(),
            {
                "query": "gear",
                "index_name": None,
                "max_results": 10,
                "max_chars_per_result": 500,
            },
        )
        self.assertEqual(
            configured_input_schema(query="gear").model_dump(),
            {"query": "gear", "max_results": 10, "max_chars_per_result": 500},
        )
        self.assertEqual(
            FilesystemSearchInput(query="gear", max_results=50, max_chars_per_result=0).model_dump(),
            {
                "query": "gear",
                "index_name": None,
                "max_results": 50,
                "max_chars_per_result": 0,
            },
        )
        self.assertEqual(
            configured_input_schema(query="gear", max_results=50, max_chars_per_result=0).model_dump(),
            {"query": "gear", "max_results": 50, "max_chars_per_result": 0},
        )

        for schema in (FilesystemSearchInput, configured_input_schema):
            with self.assertRaises(ValidationError):
                schema(query="gear", max_results=0)
            with self.assertRaises(ValidationError):
                schema(query="gear", max_results=51)
            with self.assertRaises(ValidationError):
                schema(query="gear", max_chars_per_result=-1)
            with self.assertRaises(ValidationError):
                schema(query="gear", max_chars_per_result=10001)

    async def test_configured_search_uses_its_bound_index(self) -> None:
        configured_tool = create_filesystem_search_tool(
            "configured_search",
            "Search only the configured index.",
            index_name="fixture-index",
        )
        pgvector_backend = mock.Mock()
        pgvector_backend.search = mock.AsyncMock(return_value=[])
        faiss_backend = mock.Mock()
        faiss_backend.get_loaded_indexes.return_value = []

        with (
            mock.patch(
                "ragtime.tools.filesystem_indexer.get_app_settings",
                new=mock.AsyncMock(return_value={}),
            ),
            mock.patch(
                "ragtime.tools.filesystem_indexer._get_query_embedding",
                new=mock.AsyncMock(return_value=[0.1]),
            ),
            mock.patch(
                "ragtime.tools.filesystem_indexer.get_pgvector_backend",
                return_value=pgvector_backend,
            ),
            mock.patch(
                "ragtime.tools.filesystem_indexer.get_faiss_backend",
                return_value=faiss_backend,
            ),
        ):
            result = await configured_tool.ainvoke({"query": "gear"})

        self.assertEqual(result, "No relevant documents found in index 'fixture-index' for query: gear")
        pgvector_backend.search.assert_awaited_once_with(
            query_embedding=[0.1],
            index_name="fixture-index",
            max_results=10,
        )

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
