import json
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.http_api.models import HttpApiConnectionConfig, HttpApiExecutionResult, HttpApiMethod, OpenApiCatalog, OpenApiCatalogOperation
from ragtime.rag.components import RAGComponents


class HttpApiRuntimeToolTests(unittest.IsolatedAsyncioTestCase):
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
            "allow_write": allow_write,
            "timeout_max_seconds": 45,
            "connection_config": connection.model_dump(),
        }

    async def test_http_api_runtime_builds_request_and_search_tools(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)

        tools = await rag.build_tools_from_runtime_config(self._config())

        self.assertEqual([tool.name for tool in tools], ["request_customer_api", "search_customer_api_api"])

    async def test_http_api_primary_runtime_tool_is_request_tool(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)

        tool = await rag.build_primary_runtime_tool_from_config(self._config())

        self.assertIsNotNone(tool)
        assert tool is not None
        self.assertEqual(tool.name, "request_customer_api")

    async def test_request_tool_uses_broker_and_marks_output_untrusted(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        result = HttpApiExecutionResult(
            status=200,
            output={"token": "secret", "access_token": "abc123", "items": [{"id": 1, "name": "Ada"}]},
            rows=[{"id": 1, "name": "Ada"}],
            columns=["id", "name"],
            row_count=1,
            error=None,
        )

        with mock.patch("ragtime.rag.components.http_api_broker.execute", mock.AsyncMock(return_value=result)) as execute:
            tool = await rag.build_primary_runtime_tool_from_config(self._config(allow_write=True))
            assert tool is not None and tool.coroutine is not None
            output = await tool.coroutine(method="GET", path="/customers", query={"limit": 1})

        payload = json.loads(output)
        self.assertEqual(payload["tool"], "request_customer_api")
        self.assertTrue(payload["untrusted_output"])
        self.assertEqual(payload["status"], 200)
        self.assertEqual(payload["rows"], [{"id": 1, "name": "Ada"}])
        self.assertEqual(payload["columns"], ["id", "name"])
        self.assertEqual(payload["row_count"], 1)
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["output"]["token"], "secret")
        self.assertEqual(payload["output"]["access_token"], "abc123")
        execute.assert_awaited_once()
        execute_call = execute.await_args
        self.assertIsNotNone(execute_call)
        assert execute_call is not None
        self.assertTrue(execute_call.kwargs["allow_write"])

    async def test_search_tool_uses_openapi_catalog_search(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)

        operation = OpenApiCatalogOperation(
            operation_id="listCustomers",
            method=HttpApiMethod.GET,
            path="/customers",
            summary="List customers",
            description="Fetch customer records",
            tags=["customers"],
        )
        with mock.patch("ragtime.rag.components.search_openapi_catalog", return_value=[operation]) as search:
            tools = await rag.build_tools_from_runtime_config(self._config())
            tool = tools[1]
            assert tool.coroutine is not None
            output = await tool.coroutine(query="list customers", limit=5)

        payload = json.loads(output)
        self.assertEqual(payload["tool"], "search_customer_api_api")
        self.assertEqual(payload["results"][0]["method"], "GET")
        self.assertEqual(payload["results"][0]["path"], "/customers")
        search.assert_called_once()


class HttpApiRuntimeMetadataTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "id": "tool-http-1",
            "name": "Customer API",
            "tool_type": "http_api",
            "timeout_max_seconds": 45,
            "connection_config": {
                "base_url": "https://api.example.com",
                "openapi_catalog": {
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
                },
            },
        }

    def test_derive_names_and_scope_prompt_include_http_api_tools(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        config = self._config()
        rag._tool_configs = [config]

        names = rag._derive_config_tool_names(config)
        prompt = rag._build_request_tool_scope_prompt(
            [
                SimpleNamespace(name="request_customer_api", description="HTTP API request tool. Write access is disabled for this request."),
                SimpleNamespace(name="search_customer_api_api", description="HTTP API catalog search tool. Write access is disabled for this request."),
            ],
            mode="chat",
        )

        self.assertEqual(names, {"request_customer_api", "search_customer_api_api"})
        self.assertIn("request_customer_api", prompt)
        self.assertIn("search_customer_api_api", prompt)
        self.assertIn("id=tool-http-1", prompt)

    def test_catalog_search_tool_does_not_get_live_wiring_suffix(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        request_tool = SimpleNamespace(name="request_customer_api", description="Base request description")
        search_tool = SimpleNamespace(name="search_customer_api_api", description="Base search description")

        overridden = rag._apply_mode_specific_tool_description_overrides([request_tool, search_tool], "userspace")

        descriptions = {tool.name: tool.description for tool in overridden}
        self.assertIn("live-wired dashboard components", descriptions["request_customer_api"])
        self.assertNotIn("live-wired dashboard components", descriptions["search_customer_api_api"])
