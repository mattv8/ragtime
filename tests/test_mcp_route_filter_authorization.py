import asyncio
import unittest
from unittest import mock

import ragtime.mcp.tools as mcp_tools
from ragtime.mcp.tools import McpRouteFilter, MCPToolAdapter, MCPToolDefinition


class _HangingMcpTool:
    def __init__(self) -> None:
        self.started = False

    async def __call__(self, **_kwargs: object) -> str:
        self.started = True
        await asyncio.Event().wait()
        return "unreachable"


class McpRouteFilterAuthorizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_database_tool_is_allowed_by_name_without_building_tools(self) -> None:
        adapter = MCPToolAdapter()
        route_filter = McpRouteFilter(
            tool_config_ids=["tool-1"],
            include_knowledge_search=False,
            include_git_history=False,
        )
        tool_configs = [
            {
                "id": "tool-1",
                "name": "Production Infoscan Database",
                "tool_type": "postgres",
                "enabled": True,
            }
        ]

        with mock.patch("ragtime.mcp.tools.get_tool_configs", mock.AsyncMock(return_value=tool_configs)):
            allowed = await adapter.is_tool_allowed_by_route_filter(
                "query_production_infoscan_database",
                route_filter,
            )

        self.assertTrue(allowed)

    async def test_unselected_database_tool_is_rejected(self) -> None:
        adapter = MCPToolAdapter()
        route_filter = McpRouteFilter(
            tool_config_ids=["tool-1"],
            include_knowledge_search=False,
            include_git_history=False,
        )
        tool_configs = [
            {
                "id": "tool-2",
                "name": "Production Infoscan Database",
                "tool_type": "postgres",
                "enabled": True,
            }
        ]

        with mock.patch("ragtime.mcp.tools.get_tool_configs", mock.AsyncMock(return_value=tool_configs)):
            allowed = await adapter.is_tool_allowed_by_route_filter(
                "query_production_infoscan_database",
                route_filter,
            )

        self.assertFalse(allowed)

    def test_ssh_schema_uses_configured_max_as_default(self) -> None:
        adapter = MCPToolAdapter()

        schema = adapter._build_input_schema(  # pyright: ignore[reportPrivateUsage]
            {
                "id": "tool-1",
                "name": "Docker Host",
                "tool_type": "ssh_shell",
                "enabled": True,
                "timeout_max_seconds": 45,
            }
        )

        self.assertIsNotNone(schema)
        assert schema is not None
        timeout_schema = schema["properties"]["timeout"]
        self.assertEqual(timeout_schema["default"], 45)
        self.assertEqual(timeout_schema["maximum"], 45)

    def test_mcp_call_timeout_uses_explicit_requested_timeout(self) -> None:
        adapter = MCPToolAdapter()

        effective_timeout = adapter._resolve_mcp_call_timeout(  # pyright: ignore[reportPrivateUsage]
            {"command": "sleep 999", "timeout": 60},
            timeout_max_seconds=300,
            input_schema={"properties": {"timeout": {"default": 300}}},
        )

        self.assertEqual(effective_timeout, 60)

    async def test_execute_tool_returns_timeout_when_executor_hangs(self) -> None:
        adapter = MCPToolAdapter()
        hanging_tool = _HangingMcpTool()
        adapter._tool_definitions["ssh_docker_1"] = MCPToolDefinition(  # pyright: ignore[reportPrivateUsage]
            name="ssh_docker_1",
            description="Execute shell commands via SSH.",
            input_schema={"properties": {"timeout": {"default": 1}}},
            tool_config={"timeout_max_seconds": 1},
            execute_fn=hanging_tool,
        )

        with mock.patch.object(mcp_tools, "MCP_TOOL_TIMEOUT_GRACE_SECONDS", 0.01):
            result = await adapter.execute_tool(
                "ssh_docker_1",
                {"command": "sleep 999", "timeout": 1},
            )

        self.assertTrue(hanging_tool.started)
        self.assertIn("timed out after 1 seconds", result)

    def test_invalidate_cache_drops_built_tool_definitions(self) -> None:
        adapter = MCPToolAdapter()

        async def stale_executor(**_kwargs: object) -> str:
            return "stale"

        adapter._tool_definitions["ssh_docker_1"] = MCPToolDefinition(  # pyright: ignore[reportPrivateUsage]
            name="ssh_docker_1",
            description="Execute shell commands via SSH.",
            input_schema={},
            tool_config={"id": "tool-1", "tool_type": "ssh_shell", "allow_write": False},
            execute_fn=stale_executor,
        )
        adapter._tool_executors["ssh_docker_1"] = stale_executor  # pyright: ignore[reportPrivateUsage]

        adapter.invalidate_cache()

        self.assertNotIn("ssh_docker_1", adapter._tool_definitions)  # pyright: ignore[reportPrivateUsage]
        self.assertNotIn("ssh_docker_1", adapter._tool_executors)  # pyright: ignore[reportPrivateUsage]

    async def test_execute_tool_rebuilds_from_config_after_invalidation(self) -> None:
        adapter = MCPToolAdapter()
        calls: list[str] = []

        async def stale_executor(**_kwargs: object) -> str:
            calls.append("stale")
            return "stale"

        async def fresh_executor(**_kwargs: object) -> str:
            calls.append("fresh")
            return "fresh"

        # Simulate a previously-built definition with the old (read-only) closure.
        adapter._tool_definitions["ssh_docker_1"] = MCPToolDefinition(  # pyright: ignore[reportPrivateUsage]
            name="ssh_docker_1",
            description="Execute shell commands via SSH.",
            input_schema={},
            tool_config={"id": "tool-1", "tool_type": "ssh_shell", "allow_write": False},
            execute_fn=stale_executor,
        )

        fresh_config = {
            "id": "tool-1",
            "name": "Docker 1",
            "tool_type": "ssh_shell",
            "enabled": True,
            "allow_write": True,
        }
        fresh_definition = MCPToolDefinition(
            name="ssh_docker_1",
            description="Execute shell commands via SSH.",
            input_schema={},
            tool_config=fresh_config,
            execute_fn=fresh_executor,
        )

        # A tool change drops the stale cache; the next call rebuilds from config.
        adapter.invalidate_cache()

        with (
            mock.patch("ragtime.mcp.tools.get_tool_configs", mock.AsyncMock(return_value=[fresh_config])) as get_tool_configs,
            mock.patch.object(adapter, "_create_tool_definition", mock.AsyncMock(return_value=fresh_definition)) as create_tool_definition,
        ):
            result = await adapter.execute_tool("ssh_docker_1", {"command": "touch /tmp/file"})

        self.assertEqual(result, "fresh")
        self.assertEqual(calls, ["fresh"])
        get_tool_configs.assert_awaited_once_with()
        create_tool_definition.assert_awaited_once_with(fresh_config)


if __name__ == "__main__":
    unittest.main()
