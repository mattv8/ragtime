from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from ragtime.indexer import routes


class _HangingRetryTool:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def ainvoke(self, tool_input: dict[str, object]) -> str:
        self.started.set()
        await asyncio.Event().wait()
        return "unreachable"


class _SuccessfulRetryTool:
    async def ainvoke(self, tool_input: dict[str, object]) -> str:
        return f"ok: {tool_input['query']}"


class RetryTerminalToolTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_tool_returns_output_before_proxy_timeout(self) -> None:
        tool = _HangingRetryTool()

        with mock.patch.object(routes, "get_http_proxy_safe_timeout_seconds", mock.AsyncMock(return_value=0.01)):
            output = await routes._invoke_retry_terminal_tool_with_http_timeout(
                tool,
                {"query": "select * from slow_table limit 10"},
            )

        self.assertIn("Error: Replay query timed out after 1 second", output)
        self.assertIn("before the tool finished", output)
        self.assertTrue(tool.started.is_set())

    async def test_retry_tool_returns_successful_output(self) -> None:
        output = await routes._invoke_retry_terminal_tool_with_http_timeout(
            _SuccessfulRetryTool(),
            {"query": "select 1"},
        )

        self.assertEqual(output, "ok: select 1")


if __name__ == "__main__":
    unittest.main()
