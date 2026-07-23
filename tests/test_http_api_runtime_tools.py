import json
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.http_api.models import (
    HttpApiAuthMode,
    HttpApiConnectionConfig,
    HttpApiExecutionResult,
    HttpApiMethod,
    HttpApiOAuthFlow,
    OpenApiCatalog,
    OpenApiCatalogOperation,
)
from ragtime.rag.components import RAGComponents


class HttpApiRuntimeToolTests(unittest.IsolatedAsyncioTestCase):
    def _config(self, *, with_catalog: bool = True, allow_write: bool = False, documentation_url: str = "") -> dict:
        connection = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            documentation_url=documentation_url,
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

    async def test_request_tool_schema_documents_configured_field_precedence(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        tool = await rag.build_primary_runtime_tool_from_config(self._config())

        assert tool is not None and tool.args_schema is not None
        descriptions = {name: field.description or "" for name, field in tool.args_schema.model_fields.items() if name in {"json_body", "form_body"}}
        self.assertIn("configured server-side fields merge in and win duplicate keys", descriptions["json_body"])
        self.assertIn("configured server-side fields merge in and win duplicate keys", descriptions["form_body"])

    async def test_request_tool_description_includes_documentation_url_and_web_browse_guidance(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        documentation_url = "https://docs.example.com/api"

        tool = await rag.build_primary_runtime_tool_from_config(self._config(documentation_url=documentation_url))

        assert tool is not None
        self.assertIn(documentation_url, tool.description)
        self.assertIn("web_browse", tool.description)
        self.assertIn("endpoint paths, parameters, or response details", tool.description)

    async def test_request_tool_guidance_describes_automatic_auth_and_omitted_headers(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        config = self._config()
        config["connection_config"] = HttpApiConnectionConfig(
            **{**config["connection_config"], "auth_mode": HttpApiAuthMode.TOKEN_EXCHANGE},
        ).model_dump()

        tool = await rag.build_primary_runtime_tool_from_config(config)

        assert tool is not None and tool.args_schema is not None
        self.assertIn("Authentication is applied automatically", tool.description)
        self.assertIn("do not call login or token endpoints", tool.description)
        self.assertIn("No per-request headers are approved; omit headers", tool.description)
        self.assertIn(
            "No per-request headers are approved; omit headers",
            tool.args_schema.model_fields["headers"].description or "",
        )

    async def test_request_tool_guidance_lists_exact_approved_headers(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        config = self._config()
        config["connection_config"] = HttpApiConnectionConfig(
            **{**config["connection_config"], "approved_request_headers": ["X-Trace-Id", "Accept"]},
        ).model_dump()

        tool = await rag.build_primary_runtime_tool_from_config(config)

        assert tool is not None and tool.args_schema is not None
        self.assertIn("X-Trace-Id", tool.description)
        self.assertIn("Accept", tool.description)
        self.assertNotIn("Authorization", tool.description)
        headers_description = tool.args_schema.model_fields["headers"].description or ""
        self.assertIn("X-Trace-Id", headers_description)
        self.assertIn("Accept", headers_description)
        self.assertNotIn("Authorization", headers_description)

    def _oauth_runtime_config(self, *, access_token: str = "old-access", refresh_token: str = "old-refresh") -> dict:
        connection = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.OAUTH2,
            oauth_flow=HttpApiOAuthFlow.DEVICE_CODE,
            oauth_client_id="client",
            oauth_device_authorization_url="https://api.example.com/device",
            oauth_token_url="https://api.example.com/token",
            oauth_access_token=access_token,
            oauth_refresh_token=refresh_token,
            oauth_token_expires_at="2999-01-01T00:00:00Z",
        )
        return {
            "id": "tool-http-oauth",
            "name": "OAuth API",
            "tool_type": "http_api",
            "allow_write": False,
            "timeout_max_seconds": 45,
            "connection_config": connection.model_dump(),
        }

    async def test_oauth_runtime_persists_only_rotated_credentials_and_rebinds_config(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        runtime_config = self._oauth_runtime_config()
        current = SimpleNamespace(connection_config=dict(runtime_config["connection_config"], token_prefix="Token"))
        persisted: list[dict] = []
        seen_credentials: list[tuple[str, str]] = []
        result = HttpApiExecutionResult(status=200, output={"ok": True})

        async def execute(*args, **kwargs):
            config = args[1]
            seen_credentials.append((config.oauth_access_token, config.oauth_refresh_token))
            if len(seen_credentials) == 1:
                await kwargs["oauth_credential_updater"](
                    {
                        "oauth_access_token": "new-access",
                        "oauth_refresh_token": "new-refresh",
                        "oauth_token_type": "Bearer",
                        "oauth_token_expires_at": "2999-01-02T00:00:00Z",
                        "oauth_refresh_token_used": "old-refresh",
                        "unexpected": "must-not-persist",
                    }
                )
            return result

        async def update_tool(_tool_id, updates):
            persisted.append(updates)

        with (
            mock.patch("ragtime.rag.components.http_api_broker.execute", side_effect=execute),
            mock.patch("ragtime.rag.components.repository.get_tool_config", new=mock.AsyncMock(return_value=current)),
            mock.patch("ragtime.rag.components.repository.update_tool_config", side_effect=update_tool),
        ):
            tool = await rag.build_primary_runtime_tool_from_config(runtime_config)
            assert tool is not None and tool.coroutine is not None
            await tool.coroutine(method="GET", path="/items")
            await tool.coroutine(method="GET", path="/items")

        self.assertEqual(seen_credentials, [("old-access", "old-refresh"), ("new-access", "new-refresh")])
        self.assertEqual(len(persisted), 1)
        saved_config = persisted[0]["connection_config"]
        self.assertEqual(saved_config["oauth_access_token"], "new-access")
        self.assertEqual(saved_config["oauth_refresh_token"], "new-refresh")
        self.assertNotIn("oauth_refresh_token_used", saved_config)
        self.assertNotIn("unexpected", saved_config)
        self.assertEqual(saved_config["token_prefix"], "Token")

    async def test_oauth_runtime_skips_stale_rotation_after_reconnect_and_rebinds_config(self) -> None:
        rag = RAGComponents.__new__(RAGComponents)
        runtime_config = self._oauth_runtime_config()
        reconnected = dict(runtime_config["connection_config"], oauth_access_token="replacement-access", oauth_refresh_token="replacement-refresh")
        current = SimpleNamespace(connection_config=reconnected)
        seen_credentials: list[tuple[str, str]] = []
        result = HttpApiExecutionResult(status=200, output={"ok": True})

        async def execute(*args, **kwargs):
            config = args[1]
            seen_credentials.append((config.oauth_access_token, config.oauth_refresh_token))
            if len(seen_credentials) == 1:
                await kwargs["oauth_credential_updater"](
                    {
                        "oauth_access_token": "stale-access",
                        "oauth_refresh_token": "stale-refresh",
                        "oauth_refresh_token_used": "old-refresh",
                    }
                )
            return result

        update_tool = mock.AsyncMock()
        with (
            mock.patch("ragtime.rag.components.http_api_broker.execute", side_effect=execute),
            mock.patch("ragtime.rag.components.repository.get_tool_config", new=mock.AsyncMock(return_value=current)),
            mock.patch("ragtime.rag.components.repository.update_tool_config", update_tool),
        ):
            tool = await rag.build_primary_runtime_tool_from_config(runtime_config)
            assert tool is not None and tool.coroutine is not None
            await tool.coroutine(method="GET", path="/items")
            await tool.coroutine(method="GET", path="/items")

        self.assertEqual(seen_credentials, [("old-access", "old-refresh"), ("replacement-access", "replacement-refresh")])
        update_tool.assert_not_awaited()

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
