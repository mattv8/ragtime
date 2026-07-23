import unittest
from unittest import mock

from ragtime.http_api.models import HttpApiAuthMode, HttpApiConnectionConfig, OpenApiCatalog
from ragtime.mcp.tools import MCPToolAdapter, MCPToolDefinition


class HttpApiMcpToolTests(unittest.IsolatedAsyncioTestCase):
    def _config(self, *, with_catalog: bool = True, allow_write: bool = False) -> dict:
        connection = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            openapi_catalog=(
                OpenApiCatalog.model_validate(
                    {
                        "title": "Customer API",
                        "version": "1.0",
                        "operations": [
                            {
                                "operation_id": "listCustomers",
                                "method": "GET",
                                "path": "/customers",
                                "summary": "List customers",
                                "description": "Fetch customer records",
                                "tags": ["customers"],
                            }
                        ],
                    }
                )
                if with_catalog
                else None
            ),
        )
        return {
            "id": "tool-http-1",
            "name": "Customer API",
            "tool_type": "http_api",
            "description": "CRM endpoints",
            "enabled": True,
            "allow_write": allow_write,
            "timeout_max_seconds": 45,
            "connection_config": connection.model_dump(),
        }

    def test_http_api_tool_name_and_schema(self) -> None:
        adapter = MCPToolAdapter()

        name = adapter._build_tool_name(self._config())  # pyright: ignore[reportPrivateUsage]
        schema = adapter._build_input_schema(self._config())  # pyright: ignore[reportPrivateUsage]

        self.assertEqual(name, "request_customer_api")
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertEqual(schema["required"], ["method", "path"])
        self.assertIn("response_selector", schema["properties"])

    async def test_http_api_search_definition_created_when_catalog_present(self) -> None:
        adapter = MCPToolAdapter()

        tool_def = await adapter._create_http_api_catalog_search_tool_definition(self._config())  # pyright: ignore[reportPrivateUsage]

        self.assertIsNotNone(tool_def)
        assert tool_def is not None
        self.assertEqual(tool_def.name, "search_customer_api_api")
        self.assertIn("OpenAPI", tool_def.description)

    async def test_http_api_executor_reuses_runtime_builder(self) -> None:
        adapter = MCPToolAdapter()
        fake_tool = mock.Mock()
        fake_tool.ainvoke = mock.AsyncMock(return_value="ok")

        with mock.patch("ragtime.mcp.tools.RAGComponents.build_primary_runtime_tool_from_config", mock.AsyncMock(return_value=fake_tool)) as build_tool:
            executor = await adapter._create_executor(self._config(allow_write=True), "http_api")  # pyright: ignore[reportPrivateUsage]
            result = await executor(method="GET", path="/customers")

        self.assertEqual(result, "ok")
        build_tool.assert_awaited_once()

    async def test_execute_tool_cold_path_builds_and_caches_http_api_catalog_search_tool(self) -> None:
        adapter = MCPToolAdapter()
        config = self._config()
        request_def = MCPToolDefinition(
            name="request_customer_api",
            description="Request API",
            input_schema={"properties": {}},
            tool_config=config,
            execute_fn=mock.AsyncMock(return_value="request-ok"),
        )
        catalog_def = MCPToolDefinition(
            name="search_customer_api_api",
            description="Search OpenAPI catalog",
            input_schema={"properties": {}},
            tool_config=config,
            execute_fn=mock.AsyncMock(return_value="catalog-ok"),
        )

        with (
            mock.patch("ragtime.mcp.tools.get_tool_configs", mock.AsyncMock(return_value=[config])) as get_tool_configs,
            mock.patch.object(adapter, "_check_heartbeats", mock.AsyncMock(return_value={"tool-http-1": mock.Mock(alive=True)})),
            mock.patch.object(adapter, "_create_tool_definition", mock.AsyncMock(return_value=request_def)),
            mock.patch.object(adapter, "_create_http_api_catalog_search_tool_definition", mock.AsyncMock(return_value=catalog_def)) as create_catalog_def,
            mock.patch("ragtime.mcp.tools.get_app_settings", mock.AsyncMock(return_value={"aggregate_search": False})),
            mock.patch.object(adapter, "_create_per_index_search_tools", mock.AsyncMock(return_value=[])),
            mock.patch.object(adapter, "_create_git_history_tools", mock.AsyncMock(return_value=[])),
        ):
            listed = await adapter.get_available_tools()
            adapter.invalidate_cache()
            result = await adapter.execute_tool("search_customer_api_api", {"query": "customers"})

        self.assertIn("search_customer_api_api", [tool.name for tool in listed])
        self.assertEqual(result, "catalog-ok")
        get_tool_configs.assert_awaited()
        create_catalog_def.assert_awaited_with(config)

    def test_http_api_description_guidance_describes_automatic_auth_and_omitted_headers(self) -> None:
        adapter = MCPToolAdapter()
        config = self._config()
        config["connection_config"] = HttpApiConnectionConfig(
            **{**config["connection_config"], "auth_mode": HttpApiAuthMode.TOKEN_EXCHANGE},
        ).model_dump()

        description = adapter._build_tool_description(config)  # pyright: ignore[reportPrivateUsage]
        schema = adapter._build_input_schema(config)  # pyright: ignore[reportPrivateUsage]

        assert schema is not None
        self.assertIn("Authentication is applied automatically", description)
        self.assertIn("do not call login or token endpoints", description)
        self.assertIn("No per-request headers are approved; omit headers", description)
        self.assertIn(
            "No per-request headers are approved; omit headers",
            schema["properties"]["headers"]["description"],
        )

    def test_http_api_description_guidance_lists_exact_approved_headers(self) -> None:
        adapter = MCPToolAdapter()
        config = self._config()
        config["connection_config"] = HttpApiConnectionConfig(
            **{**config["connection_config"], "approved_request_headers": ["X-Trace-Id", "Accept"]},
        ).model_dump()

        description = adapter._build_tool_description(config)  # pyright: ignore[reportPrivateUsage]
        schema = adapter._build_input_schema(config)  # pyright: ignore[reportPrivateUsage]

        assert schema is not None
        self.assertIn("X-Trace-Id", description)
        self.assertIn("Accept", description)
        self.assertNotIn("Authorization", description)
        headers_description = schema["properties"]["headers"]["description"]
        self.assertIn("X-Trace-Id", headers_description)
        self.assertIn("Accept", headers_description)
        self.assertNotIn("Authorization", headers_description)
