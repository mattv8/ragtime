import json
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.core.faiss_concurrency import FaissSearchBusyError
from ragtime.mcp.tools import MCPToolAdapter


class _FakeDoc:
    def __init__(self, page_content: str, source: str) -> None:
        self.page_content = page_content
        self.metadata = {"source": source}


class _FakeFaissDb:
    def __init__(self, docs: list[_FakeDoc]) -> None:
        self._docs = docs

    def similarity_search(self, query: str, k: int):
        return self._docs[:k]

    def max_marginal_relevance_search(self, query: str, k: int, fetch_k: int, lambda_mult: float):
        return self._docs[:k]


class McpFaissSearchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.adapter = MCPToolAdapter()

    def _set_rag_state(self, *, db) -> tuple[bool, dict, dict, list, dict | None]:
        from ragtime.mcp import tools as mcp_tools

        previous = (
            mcp_tools.rag.is_ready,
            dict(mcp_tools.rag.faiss_dbs),
            dict(mcp_tools.rag.retrievers),
            list(mcp_tools.rag._index_metadata or []),  # pyright: ignore[reportPrivateUsage]
            mcp_tools.rag._app_settings,  # pyright: ignore[reportPrivateUsage]
        )
        mcp_tools.rag.is_ready = True
        mcp_tools.rag.faiss_dbs = {"docs": db}
        mcp_tools.rag.retrievers = {"docs": SimpleNamespace()}
        mcp_tools.rag._index_metadata = [{"name": "docs", "description": "Docs index", "enabled": True}]  # pyright: ignore[reportPrivateUsage]
        mcp_tools.rag._app_settings = {"search_results_k": 5, "search_use_mmr": False, "search_mmr_lambda": 0.5}  # pyright: ignore[reportPrivateUsage]
        return previous

    def _restore_rag_state(self, previous: tuple[bool, dict, dict, list, dict | None]) -> None:
        from ragtime.mcp import tools as mcp_tools

        is_ready, faiss_dbs, retrievers, index_metadata, app_settings = previous
        mcp_tools.rag.is_ready = is_ready
        mcp_tools.rag.faiss_dbs = faiss_dbs
        mcp_tools.rag.retrievers = retrievers
        mcp_tools.rag._index_metadata = index_metadata  # pyright: ignore[reportPrivateUsage]
        mcp_tools.rag._app_settings = app_settings  # pyright: ignore[reportPrivateUsage]

    async def test_aggregate_knowledge_search_uses_faiss_search_coordinator(self) -> None:
        db = _FakeFaissDb([_FakeDoc("first body line", "docs/one.txt")])
        coordinator_run = mock.AsyncMock(return_value=db._docs)
        previous = self._set_rag_state(db=db)

        try:
            with mock.patch(
                "ragtime.mcp.tools.faiss_search_coordinator",
                SimpleNamespace(run=coordinator_run),
                create=True,
            ):
                tool = await self.adapter._create_knowledge_search_tool()  # pyright: ignore[reportPrivateUsage]
                assert tool is not None
                result = await tool.execute_fn(query="hello world", k=1)
        finally:
            self._restore_rag_state(previous)

        payload = json.loads(result)
        self.assertEqual(payload["status"], "completed")
        coordinator_run.assert_awaited_once()
        await_args = coordinator_run.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(await_args.args[:2], ("docs", db.similarity_search))
        self.assertEqual(await_args.kwargs, {"k": 1})

    async def test_per_index_knowledge_search_busy_error_stays_structured(self) -> None:
        db = _FakeFaissDb([_FakeDoc("first body line", "docs/one.txt")])
        coordinator_run = mock.AsyncMock(side_effect=FaissSearchBusyError("Timed out waiting for FAISS search slot for index 'docs'"))
        previous = self._set_rag_state(db=db)

        try:
            with mock.patch(
                "ragtime.mcp.tools.faiss_search_coordinator",
                SimpleNamespace(run=coordinator_run),
                create=True,
            ):
                tools = await self.adapter._create_per_index_search_tools()  # pyright: ignore[reportPrivateUsage]
                self.assertEqual(len(tools), 1)
                result = await tools[0].execute_fn(query="hello world", k=1)
        finally:
            self._restore_rag_state(previous)

        payload = json.loads(result)
        self.assertEqual(payload["status"], "error")
        self.assertFalse(payload["ok"])
        self.assertIn("Search error", payload["message"])
        self.assertIn("FAISS search is busy. Please retry shortly.", payload["message"])
        coordinator_run.assert_awaited_once()
