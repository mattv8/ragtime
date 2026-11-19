import asyncio
import copy
import unittest
from contextlib import asynccontextmanager
from unittest import mock

from mcp.server import Server
from mcp.types import ListToolsRequest

from ragtime.mcp.server import _register_handlers, mcp_tool_adapter, run_mcp_server
from ragtime.mcp.tools import MCPToolAdapter, MCPToolDefinition
from ragtime.tools.ssh_transfer import SSHTransferInput


class McpHttpTimeoutLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.adapter = MCPToolAdapter()
        self.executor = mock.AsyncMock(return_value="ok")
        self.definition = MCPToolDefinition(
            name="ssh_test",
            description="test",
            input_schema=self.adapter._build_input_schema({"tool_type": "ssh_shell", "timeout_max_seconds": 300}),
            tool_config={"timeout_max_seconds": 300},
            execute_fn=self.executor,
        )
        self.adapter._tool_definitions["ssh_test"] = self.definition
        self.proxy_timeout_patch = mock.patch("ragtime.mcp.tools.get_http_proxy_safe_timeout_seconds", new=mock.AsyncMock(return_value=90))
        self.proxy_timeout_patch.start()
        self.addCleanup(self.proxy_timeout_patch.stop)

    async def test_http_caps_and_copies_arguments(self) -> None:
        for request, expected in [({}, 80), ({"timeout": 0}, 80), ({"timeout": 300}, 80), ({"timeout": 30}, 30)]:
            with self.subTest(request=request):
                original = request.copy()
                self.assertEqual(await self.adapter.execute_tool("ssh_test", request), "ok")
                self.executor.assert_awaited_with(timeout=expected)
                self.assertEqual(request, original)

    async def test_stdio_preserves_unlimited_and_omitted_behavior(self) -> None:
        self.adapter.http_call_limit_enabled = False
        with mock.patch("ragtime.mcp.tools.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
            await self.adapter.execute_tool("ssh_test", {})
            self.executor.assert_awaited_with()
            self.assertEqual(wait_for.call_args.kwargs["timeout"], 310)
            wait_for.reset_mock()
            await self.adapter.execute_tool("ssh_test", {"timeout": 0})
            self.executor.assert_awaited_with(timeout=0)
            wait_for.assert_not_called()

    async def test_unlimited_configuration_with_stdio_does_not_wait_for(self) -> None:
        self.adapter.http_call_limit_enabled = False
        self.definition.tool_config["timeout_max_seconds"] = 0
        with mock.patch("ragtime.mcp.tools.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
            await self.adapter.execute_tool("ssh_test", {"timeout": 0})
            wait_for.assert_not_called()

    async def test_listed_schema_is_capped_without_mutating_definitions(self) -> None:
        second = MCPToolDefinition(
            name="ssh_short",
            description="test",
            input_schema=self.adapter._build_input_schema({"tool_type": "ssh_shell", "timeout_max_seconds": 45}),
            tool_config={"timeout_max_seconds": 45},
            execute_fn=self.executor,
        )
        original = copy.deepcopy(self.definition.input_schema)
        server = Server("test")
        _register_handlers(server, self.adapter)
        with mock.patch.object(self.adapter, "get_available_tools", new=mock.AsyncMock(return_value=[self.definition, second])):
            result = await server.request_handlers[ListToolsRequest](ListToolsRequest())
        tools = result.root.tools
        for tool, expected in zip(tools, [80, 45], strict=True):
            timeout = tool.inputSchema["properties"]["timeout"]
            self.assertEqual(timeout["default"], expected)
            self.assertEqual(timeout["maximum"], expected)
            self.assertIn(str(expected), timeout["description"])
        self.assertEqual(self.definition.input_schema, original)

    async def test_hung_executor_reports_http_limit(self) -> None:
        async def hang(**_kwargs: object) -> str:
            await asyncio.Event().wait()
            return "unreachable"

        self.definition.execute_fn = hang
        with (
            mock.patch("ragtime.mcp.tools.get_http_proxy_safe_timeout_seconds", new=mock.AsyncMock(return_value=11)),
            mock.patch("ragtime.mcp.tools.MCP_TOOL_TIMEOUT_GRACE_SECONDS", 10),
        ):
            result = await self.adapter.execute_tool("ssh_test", {})
        self.assertIn("timed out after 1 seconds (MCP HTTP per-call limit; split the work into smaller calls)", result)

    async def test_list_preserves_transfer_and_unlimited_and_timeout_free_schemas(self) -> None:
        transfer_schema = SSHTransferInput.model_json_schema()
        unlimited_schema = self.adapter._build_input_schema({"tool_type": "ssh_shell", "timeout_max_seconds": 0})
        no_timeout_schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        schemas = [transfer_schema, unlimited_schema, no_timeout_schema]
        originals = copy.deepcopy(schemas)
        definitions = [
            MCPToolDefinition(name=f"tool_{index}", description="test", input_schema=schema, tool_config={}, execute_fn=self.executor)
            for index, schema in enumerate(schemas)
        ]
        server = Server("test")
        _register_handlers(server, self.adapter)
        with mock.patch.object(self.adapter, "get_available_tools", new=mock.AsyncMock(return_value=definitions)):
            result = await server.request_handlers[ListToolsRequest](ListToolsRequest())
        listed = result.root.tools
        transfer_timeout = listed[0].inputSchema["properties"]["timeout"]
        self.assertEqual(transfer_timeout["minimum"], 1)
        self.assertEqual((transfer_timeout["default"], transfer_timeout["maximum"]), (80, 80))
        self.assertIn(originals[0]["properties"]["timeout"]["description"], transfer_timeout["description"])
        unlimited_timeout = listed[1].inputSchema["properties"]["timeout"]
        self.assertEqual((unlimited_timeout["default"], unlimited_timeout["maximum"]), (80, 80))
        self.assertIn("Command timeout in seconds (default and maximum: 80)", unlimited_timeout["description"])
        self.assertIn("Use 0 or omit to use the maximum.", unlimited_timeout["description"])
        self.assertEqual(listed[2].inputSchema, no_timeout_schema)
        self.assertEqual(schemas, originals)

    async def test_stdio_restores_http_limit_flag(self) -> None:
        original = mcp_tool_adapter.http_call_limit_enabled
        self.addCleanup(setattr, mcp_tool_adapter, "http_call_limit_enabled", original)
        mcp_tool_adapter.http_call_limit_enabled = True

        @asynccontextmanager
        async def stdio():
            yield mock.sentinel.read, mock.sentinel.write

        async def run(*_args: object) -> None:
            self.assertFalse(mcp_tool_adapter.http_call_limit_enabled)

        server = mock.Mock()
        server.run = mock.AsyncMock(side_effect=run)
        with (
            mock.patch("ragtime.mcp.server.connect_db", new=mock.AsyncMock()),
            mock.patch("ragtime.mcp.server.disconnect_db", new=mock.AsyncMock()),
            mock.patch("ragtime.mcp.server.rag.initialize", new=mock.AsyncMock()),
            mock.patch("ragtime.mcp.server.get_mcp_server", new=mock.AsyncMock(return_value=server)),
            mock.patch("mcp.server.stdio.stdio_server", stdio),
        ):
            await run_mcp_server("stdio")
        self.assertTrue(mcp_tool_adapter.http_call_limit_enabled)

    async def test_tool_without_timeout_is_unchanged(self) -> None:
        self.definition.input_schema = {"properties": {"query": {"type": "string"}}}
        await self.adapter.execute_tool("ssh_test", {"query": "hello"})
        self.executor.assert_awaited_with(query="hello")


if __name__ == "__main__":
    unittest.main()
