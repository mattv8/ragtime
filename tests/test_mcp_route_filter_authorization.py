import unittest
from unittest import mock

from ragtime.mcp.tools import McpRouteFilter, MCPToolAdapter


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


if __name__ == "__main__":
    unittest.main()
