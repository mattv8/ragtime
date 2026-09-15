from __future__ import annotations

import unittest
from unittest import mock

from ragtime.rag import components as rag_components
from tests.test_tool_skill_shared import make_rag_components


class PdmToolTests(unittest.IsolatedAsyncioTestCase):
    def _make_rag(self):
        return make_rag_components()

    @staticmethod
    def _config(**overrides):
        return {
            "id": "pdm-tool-1",
            "name": "Demo PDM",
            "tool_type": "solidworks_pdm",
            **overrides,
        }

    async def test_search_tool_clamps_configured_max_results(self) -> None:
        rag = self._make_rag()
        search = mock.AsyncMock(return_value="results")

        with (
            mock.patch.object(
                rag_components.pdm_indexer,
                "get_embedding_count",
                new=mock.AsyncMock(return_value=0),
            ),
            mock.patch("ragtime.rag.components.search_pdm_index", new=search),
        ):
            tool = await rag._create_pdm_search_tool(self._config(max_results=100), "demo_pdm", "pdm-tool-1")
            self.assertEqual(tool.name, "search_demo_pdm")
            await tool.ainvoke({"query": "bracket"})

            missing_tool = await rag._create_pdm_search_tool(self._config(), "demo_pdm", "pdm-tool-1")
            await missing_tool.ainvoke({"query": "bracket"})

            none_tool = await rag._create_pdm_search_tool(self._config(max_results=None), "demo_pdm", "pdm-tool-1")
            await none_tool.ainvoke({"query": "bracket"})

            zero_tool = await rag._create_pdm_search_tool(self._config(max_results=0), "demo_pdm", "pdm-tool-1")
            await zero_tool.ainvoke({"query": "bracket"})

            negative_tool = await rag._create_pdm_search_tool(self._config(max_results=-5), "demo_pdm", "pdm-tool-1")
            await negative_tool.ainvoke({"query": "bracket"})

        self.assertEqual([call.kwargs["max_results"] for call in search.await_args_list], [50, 10, 10, 10, 1])

    async def test_lookup_tool_validates_and_propagates_filename(self) -> None:
        rag = self._make_rag()
        lookup = mock.AsyncMock(return_value="document")

        with mock.patch("ragtime.indexer.pdm_service.lookup_pdm_documents", new=lookup, create=True):
            tool = await rag._create_pdm_lookup_tool(self._config(max_results=100), "demo_pdm", "pdm-tool-1")
            self.assertEqual(tool.name, "lookup_demo_pdm")

            self.assertEqual(
                await tool.ainvoke({}),
                "Error: Provide document_id, filename, or part_number.",
            )
            lookup.assert_not_awaited()

            self.assertEqual(await tool.ainvoke({"filename": "bracket"}), "document")

        lookup.assert_awaited_once_with(
            index_name="pdm_demo_pdm",
            document_id=None,
            filename="bracket",
            part_number=None,
            configuration=None,
            max_results=50,
        )

    async def test_runtime_pdm_config_builds_search_then_lookup_tools(self) -> None:
        rag = self._make_rag()

        with mock.patch.object(
            rag_components.pdm_indexer,
            "get_embedding_count",
            new=mock.AsyncMock(return_value=0),
        ):
            tools = await rag.build_tools_from_runtime_config(self._config())

        self.assertEqual([tool.name for tool in tools], ["search_demo_pdm", "lookup_demo_pdm"])

    async def test_search_description_mentions_snapshot(self) -> None:
        rag = self._make_rag()

        with mock.patch.object(
            rag_components.pdm_indexer,
            "get_embedding_count",
            new=mock.AsyncMock(return_value=0),
        ):
            tool = await rag._create_pdm_search_tool(self._config(), "demo_pdm", "pdm-tool-1")

        self.assertIn("snapshot", tool.description.lower())
