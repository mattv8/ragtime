import json
import unittest
from types import SimpleNamespace
from typing import cast

from langchain_core.tools import StructuredTool

from ragtime.rag.components import (
    FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES,
    KNOWLEDGE_SEARCH_TOOL_ID,
    RAGComponents,
    build_knowledge_search_payload,
    serialize_knowledge_search_payload,
    should_truncate_stream_display_output,
    wrap_tool_with_truncation,
)


class FakeDoc:
    def __init__(self, page_content: str, source: str) -> None:
        self.page_content = page_content
        self.metadata = {"source": source}


class FakeFaissDb:
    def __init__(self, docs: list[FakeDoc]) -> None:
        self._docs = docs

    def similarity_search(self, query: str, k: int):
        return self._docs[:k]

    def max_marginal_relevance_search(self, query: str, k: int, fetch_k: int, lambda_mult: float):
        return self._docs[:k]


class KnowledgeSearchToolOutputTests(unittest.IsolatedAsyncioTestCase):
    async def _search_tool(self, faiss_dbs: dict | None, index_metadata: list | None = None):
        rag = RAGComponents()
        rag._app_settings = {
            "search_results_k": 5,
            "search_use_mmr": False,
            "search_mmr_lambda": 0.5,
        }
        rag.faiss_dbs = faiss_dbs or {}
        rag.retrievers = {name: SimpleNamespace() for name in rag.faiss_dbs}
        rag._index_metadata = index_metadata or [{"name": name, "description": f"Test index {name}", "enabled": True} for name in rag.faiss_dbs]
        return rag._create_knowledge_search_tool()

    async def test_search_knowledge_emits_structured_json_payload(self):
        tool = await self._search_tool(
            {
                "docs": FakeFaissDb(
                    [
                        FakeDoc("first body line", "path/one.py"),
                        FakeDoc("second body line", "path/two.py"),
                    ]
                )
            }
        )
        coroutine = tool.coroutine
        assert coroutine is not None
        output = await coroutine(query="hello world", k=2, max_chars_per_result=200)
        payload = json.loads(output)

        self.assertEqual(payload["tool"], "search_knowledge")
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["query"], "hello world")
        self.assertIsNone(payload["index_name"])
        self.assertEqual(payload["k"], 2)
        self.assertEqual(payload["max_chars_per_result"], 200)
        self.assertEqual(payload["total_results"], 2)
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["results"][0]["index_name"], "docs")
        self.assertEqual(payload["results"][0]["source"], "path/one.py")
        self.assertEqual(payload["results"][0]["content"], "first body line")
        self.assertFalse(payload["results"][0]["truncated"])
        self.assertEqual(payload["results"][1]["source"], "path/two.py")
        self.assertEqual(payload["indexes_searched"][0]["name"], "docs")
        self.assertEqual(payload["indexes_searched"][0]["result_count"], 2)
        self.assertTrue(payload["indexes_searched"][0]["ok"])
        self.assertIn("Found 2 relevant documents", payload["message"])
        self.assertIn("duration_ms", payload)

    async def test_search_knowledge_marks_truncated_results(self):
        long_body = "x" * 1000
        tool = await self._search_tool({"docs": FakeFaissDb([FakeDoc(long_body, "big.txt")])})
        coroutine = tool.coroutine
        assert coroutine is not None
        output = await coroutine(query="anything", k=1, max_chars_per_result=50)
        payload = json.loads(output)

        self.assertTrue(payload["results"][0]["truncated"])
        self.assertTrue(payload["results"][0]["content"].endswith("... (truncated)"))
        self.assertLessEqual(len(payload["results"][0]["content"]), 200)

    async def test_search_knowledge_returns_no_results_payload_when_empty(self):
        tool = await self._search_tool({"docs": FakeFaissDb([])})
        coroutine = tool.coroutine
        assert coroutine is not None
        output = await coroutine(query="anything", k=5)
        payload = json.loads(output)

        self.assertEqual(payload["status"], "no_results")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["total_results"], 0)
        self.assertEqual(payload["results"], [])
        self.assertIn("No relevant documentation", payload["message"])

    async def test_search_knowledge_returns_no_indexes_payload_when_unloaded(self):
        tool = await self._search_tool({})
        coroutine = tool.coroutine
        assert coroutine is not None
        output = await coroutine(query="anything", k=5)
        payload = json.loads(output)

        self.assertEqual(payload["status"], "no_indexes")
        self.assertFalse(payload["ok"])
        self.assertIn("No knowledge indexes", payload["message"])

    async def test_search_knowledge_returns_loading_payload_when_indexing(self):
        rag = RAGComponents()
        rag._app_settings = {"search_results_k": 5, "search_use_mmr": False, "search_mmr_lambda": 0.5}
        rag.faiss_dbs = {}
        rag.retrievers = {}
        rag._index_metadata = []
        rag._indexes_loading = True
        rag._indexes_loaded = 1
        rag._indexes_total = 3
        tool = rag._create_knowledge_search_tool()
        coroutine = tool.coroutine
        assert coroutine is not None
        output = await coroutine(query="anything", k=5)
        payload = json.loads(output)

        self.assertEqual(payload["status"], "loading")
        self.assertFalse(payload["ok"])
        self.assertIn("currently loading", payload["message"])

    async def test_search_knowledge_returns_error_payload_on_search_failure(self):
        class FailingDb:
            def similarity_search(self, query, k):
                raise RuntimeError("embedding service offline")

            def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
                raise RuntimeError("embedding service offline")

        tool = await self._search_tool({"docs": FailingDb()})
        coroutine = tool.coroutine
        assert coroutine is not None
        output = await coroutine(query="anything", k=5)
        payload = json.loads(output)

        self.assertEqual(payload["status"], "error")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["total_results"], 0)
        self.assertGreater(len(payload["error_details"]), 0)
        self.assertEqual(payload["error_details"][0]["index_name"], "docs")
        self.assertIn("Search error", payload["error_details"][0]["message"])
        self.assertEqual(payload["indexes_searched"][0]["ok"], False)

    async def test_search_knowledge_per_index_search_emits_json(self):
        rag = RAGComponents()
        rag._app_settings = {"search_results_k": 5, "search_use_mmr": False, "search_mmr_lambda": 0.5}
        rag.faiss_dbs = {"docs": FakeFaissDb([FakeDoc("body", "src.py")])}
        rag.retrievers = {"docs": SimpleNamespace()}
        rag._index_metadata = [{"name": "docs", "description": "Docs index", "enabled": True}]
        tools = rag._create_per_index_search_tools()
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertTrue(tool.name.startswith("search_"))

        coroutine = tool.coroutine
        assert coroutine is not None
        output = await coroutine(query="hi", k=3, max_chars_per_result=200)
        payload = json.loads(output)

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["index_name"], "docs")
        self.assertEqual(payload["k"], 3)
        self.assertEqual(payload["results"][0]["source"], "src.py")
        self.assertEqual(payload["results"][0]["index_name"], "docs")
        self.assertFalse(payload["results"][0]["truncated"])


class KnowledgeSearchFrontendIntegrityTests(unittest.TestCase):
    def test_search_knowledge_in_integrity_tool_set(self):
        self.assertIn(KNOWLEDGE_SEARCH_TOOL_ID, FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES)
        self.assertIn("search_knowledge", FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES)

    def test_search_knowledge_bypasses_stream_display_truncation(self):
        output = json.dumps(
            {"tool": "search_knowledge", "results": [{"content": "x" * 3000}]},
            indent=2,
        )
        self.assertFalse(should_truncate_stream_display_output("search_knowledge", output))

    def test_search_knowledge_bypasses_global_output_truncation(self):
        class _Stub:
            name = "search_knowledge"

        stub = cast(StructuredTool, _Stub())
        wrapped = wrap_tool_with_truncation(
            stub,
            32,
            preserve_output_tool_names=FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES,
        )
        self.assertIs(wrapped, stub)


class KnowledgeSearchPayloadHelpersTests(unittest.TestCase):
    def test_build_payload_populates_index_counts_from_entries(self):
        entries = [
            {"index_name": "a", "source": "x", "content": "x", "truncated": False},
            {"index_name": "a", "source": "y", "content": "y", "truncated": False},
            {"index_name": "b", "source": "z", "content": "z", "truncated": False},
        ]
        payload = build_knowledge_search_payload(
            tool_name="search_knowledge",
            status="completed",
            ok=True,
            query="q",
            index_name=None,
            k=3,
            max_chars_per_result=500,
            entries=entries,
            errors=[],
            message="ok",
        )
        self.assertEqual(payload["total_results"], 3)
        self.assertEqual(payload["indexes_searched"][0]["result_count"], 2)
        self.assertEqual(payload["indexes_searched"][1]["result_count"], 1)

    def test_serialize_payload_emits_valid_json(self):
        payload = build_knowledge_search_payload(
            tool_name="search_knowledge",
            status="completed",
            ok=True,
            query="q",
            index_name=None,
            k=1,
            max_chars_per_result=500,
            entries=[{"index_name": "a", "source": "x.py", "content": "hello", "truncated": False}],
            errors=[],
            message="ok",
        )
        serialized = serialize_knowledge_search_payload(payload)
        parsed = json.loads(serialized)
        self.assertEqual(parsed["tool"], "search_knowledge")
        self.assertEqual(parsed["results"][0]["source"], "x.py")


if __name__ == "__main__":
    unittest.main()
