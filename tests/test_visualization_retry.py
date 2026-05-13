from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer import visualization_retry
from ragtime.indexer.models import Conversation, RetryVisualizationRequest, ToolType
from ragtime.indexer.visualization_retry import (
    VisualizationRetryContext,
    retry_visualization_with_repair,
)


def _context(selected_tool_ids: set[str] | None = None) -> VisualizationRetryContext:
    return VisualizationRetryContext(
        conversation=Conversation(id="conv-1", title="Test", model="test/model"),
        user_id="user-1",
        selected_tool_ids=selected_tool_ids or set(),
    )


class _FakeLLM:
    def __init__(self, content: str):
        self.content = content

    async def ainvoke(self, _messages):
        return SimpleNamespace(content=self.content)


class _FakeRuntimeTool:
    def __init__(self, output: str):
        self.output = output

    async def ainvoke(self, _input):
        return self.output


class _FakeToolConfig:
    enabled = True
    tool_type = ToolType.POSTGRES

    def model_dump(self):
        return {"id": "tool-1", "tool_type": "postgres"}


@contextmanager
def _rag_ready(value: bool = True):
    original = visualization_retry.rag.is_ready
    visualization_retry.rag.is_ready = value
    try:
        yield
    finally:
        visualization_retry.rag.is_ready = original


class VisualizationRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_datatable_retry_uses_deterministic_source_data(self) -> None:
        request = RetryVisualizationRequest(
            tool_type="datatable",
            source_data={"columns": ["Name", "Count"], "rows": [["A", 2], ["B", 3]]},
            title="Counts",
        )

        result = await retry_visualization_with_repair(request, _context())

        self.assertTrue(result.success)
        self.assertFalse(result.repair_used)
        self.assertEqual(result.repair_strategy, "provided_source_data")
        assert result.output is not None
        payload = json.loads(result.output)
        self.assertTrue(payload["__datatable__"])
        self.assertEqual(payload["config"]["data"], [["A", 2], ["B", 3]])

    async def test_chart_retry_can_convert_table_data_without_ai(self) -> None:
        request = RetryVisualizationRequest(
            tool_type="chart",
            source_data={"columns": ["Month", "Revenue"], "rows": [["Jan", 12], ["Feb", 20]]},
            title="Revenue",
        )

        result = await retry_visualization_with_repair(request, _context())

        self.assertTrue(result.success)
        self.assertFalse(result.repair_used)
        self.assertEqual(result.repair_strategy, "provided_source_data_table_to_chart")
        assert result.output is not None
        payload = json.loads(result.output)
        self.assertTrue(payload["__chart__"])
        self.assertEqual(payload["config"]["data"]["labels"], ["Jan", "Feb"])

    async def test_missing_source_data_uses_ai_repair(self) -> None:
        repaired = json.dumps(
            {
                "source_data": {
                    "columns": ["Name", "Count"],
                    "rows": [["A", 2], ["B", 3]],
                }
            }
        )
        request = RetryVisualizationRequest(
            tool_type="datatable",
            failed_tool_input={"title": "Bad table"},
            failed_tool_output="Validation error: data missing",
            allow_ai_repair=True,
        )

        with (
            _rag_ready(),
            mock.patch.object(
                visualization_retry.rag,
                "_get_request_scoped_llm",
                new=mock.AsyncMock(
                    return_value=SimpleNamespace(
                        llm=_FakeLLM(repaired), provider="test", model="test/model"
                    )
                ),
            ),
            mock.patch.object(visualization_retry.settings, "debug_mode", False),
        ):
            result = await retry_visualization_with_repair(request, _context())

        self.assertTrue(result.success)
        self.assertTrue(result.repair_used)
        self.assertEqual(result.repair_strategy, "ai_repair")

    async def test_source_query_rerun_can_supply_missing_tabledata(self) -> None:
        metadata = '<!--TABLEDATA:{"columns":["Name","Count"],"rows":[["A",2],["B",3]]}-->\nName | Count'
        request = RetryVisualizationRequest(
            tool_type="datatable",
            failed_tool_output="Cannot render table",
            allow_source_rerun=True,
            context_events=[
                {
                    "type": "tool",
                    "tool": "postgres_query",
                    "input": {"query": "select name, count from metrics"},
                    "connection": {"tool_config_id": "tool-1", "tool_type": "postgres"},
                }
            ],
        )

        with (
            _rag_ready(),
            mock.patch.object(
                visualization_retry.repository,
                "get_tool_config",
                new=mock.AsyncMock(return_value=_FakeToolConfig()),
            ),
            mock.patch.object(
                visualization_retry.rag,
                "build_primary_runtime_tool_from_config",
                new=mock.AsyncMock(return_value=_FakeRuntimeTool(metadata)),
            ),
        ):
            result = await retry_visualization_with_repair(request, _context({"tool-1"}))

        self.assertTrue(result.success)
        self.assertFalse(result.repair_used)
        self.assertTrue(result.source_rerun_used)
        self.assertEqual(result.repair_strategy, "source_rerun")

    async def test_invalid_ai_repair_is_rejected(self) -> None:
        request = RetryVisualizationRequest(
            tool_type="chart",
            failed_tool_output="Validation error",
            allow_ai_repair=True,
            allow_source_rerun=False,
        )

        with (
            _rag_ready(),
            mock.patch.object(
                visualization_retry.rag,
                "_get_request_scoped_llm",
                new=mock.AsyncMock(
                    return_value=SimpleNamespace(
                        llm=_FakeLLM('{"source_data":{"labels":["A"],"datasets":[{"label":"x","data":["nope"]}]}}'),
                        provider="test",
                        model="test/model",
                    )
                ),
            ),
            mock.patch.object(visualization_retry.settings, "debug_mode", False),
        ):
            result = await retry_visualization_with_repair(request, _context())

        self.assertFalse(result.success)
        self.assertTrue(result.repair_used)
        self.assertEqual(result.repair_strategy, "failed")

    async def test_successful_repair_persists_event_output(self) -> None:
        request = RetryVisualizationRequest(
            tool_type="datatable",
            source_data={"columns": ["Name", "Count"], "rows": [["A", 2]]},
            title="Counts",
            message_id="msg-42",
            event_index=3,
        )
        update_mock = mock.AsyncMock(return_value=True)
        with mock.patch.object(
            visualization_retry.repository,
            "update_message_event_output",
            new=update_mock,
        ):
            result = await retry_visualization_with_repair(request, _context())

        self.assertTrue(result.success)
        update_mock.assert_awaited_once()
        args, kwargs = update_mock.call_args
        self.assertEqual(args[0], "conv-1")
        self.assertEqual(args[1], "msg-42")
        self.assertEqual(args[2], 3)
        self.assertEqual(args[3], result.output)
        self.assertIsNone(kwargs.get("message_index"))
        self.assertEqual(kwargs.get("expected_tool"), "create_datatable")

    async def test_successful_repair_persists_with_message_index_fallback(self) -> None:
        request = RetryVisualizationRequest(
            tool_type="chart",
            source_data={"columns": ["Month", "Revenue"], "rows": [["Jan", 12], ["Feb", 20]]},
            title="Revenue",
            message_index=2,
            event_index=1,
        )
        update_mock = mock.AsyncMock(return_value=True)
        with mock.patch.object(
            visualization_retry.repository,
            "update_message_event_output",
            new=update_mock,
        ):
            result = await retry_visualization_with_repair(request, _context())

        self.assertTrue(result.success)
        update_mock.assert_awaited_once()
        args, kwargs = update_mock.call_args
        self.assertEqual(args[0], "conv-1")
        self.assertIsNone(args[1])
        self.assertEqual(args[2], 1)
        self.assertEqual(args[3], result.output)
        self.assertEqual(kwargs.get("message_index"), 2)
        self.assertEqual(kwargs.get("expected_tool"), "create_chart")

    async def test_persistence_skipped_without_message_identifiers(self) -> None:
        request = RetryVisualizationRequest(
            tool_type="datatable",
            source_data={"columns": ["Name", "Count"], "rows": [["A", 2]]},
        )
        update_mock = mock.AsyncMock(return_value=True)
        with mock.patch.object(
            visualization_retry.repository,
            "update_message_event_output",
            new=update_mock,
        ):
            result = await retry_visualization_with_repair(request, _context())

        self.assertTrue(result.success)
        update_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
