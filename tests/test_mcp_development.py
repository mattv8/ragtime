import asyncio
import json
import unittest
from unittest import mock

from fastapi import HTTPException
from mcp.server import Server
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, ListToolsRequest, ListToolsResult, TextContent, Tool

from ragtime.content_protection.models import ContentProtectionError
from ragtime.mcp import routes as mcp_routes
from ragtime.mcp import server as mcp_server
from ragtime.mcp.tools import MCPToolAdapter, MCPToolDefinition
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_service import development_service


class McpDevelopmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_are_scoped_to_request_principal(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [
            {"name": "context", "description": "Context", "scope": "read", "input_schema": {}},
            {"name": "file_write", "description": "Write", "scope": "write", "input_schema": {}},
        ]

        with mock.patch.object(development_service, "list_operations", return_value=operations):
            with mcp_server.development_principal_context(principal):
                tools = await mcp_server._development_tools_for_principal(principal)  # pyright: ignore[reportPrivateUsage]
                self.assertIs(mcp_server.get_request_development_principal(), principal)

        self.assertIsNone(mcp_server.get_request_development_principal())
        self.assertEqual(tools[0].inputSchema["properties"]["operation"]["enum"], ["context"])
        readers = {tool.name: tool for tool in tools[1:]}
        self.assertIn("offset", readers["workspace_development_contract"].inputSchema["properties"])
        self.assertIn("offset", readers["workspace_development_resource_contract"].inputSchema["properties"])

    async def test_execution_rechecks_operation_scope_and_returns_mcp_error(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "context", "description": "Context", "scope": "read", "input_schema": {}}]

        with mock.patch.object(development_service, "list_operations", return_value=operations):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development",
                    {"workspace_id": "workspace-1", "operation": "file_write", "arguments": {}},
                )

        self.assertTrue(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text)["error"]["code"], "operation_not_allowed")

    async def test_execution_uses_shared_dispatcher_with_request_principal(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "context", "description": "Context", "scope": "read", "input_schema": {}}]

        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value={"context_revision": "abc"})) as execute,
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development_context", {"workspace_id": "workspace-1"}
                )

        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text)["context_revision"], "abc")
        execute.assert_awaited_once_with(principal, "workspace-1", "context", {})

    async def test_both_context_paths_use_bounded_compact_delivery(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "context", "description": "Context", "scope": "read", "input_schema": {}}]
        context = {"context_revision": "abc", "capabilities": {"authorized_tools": [], "authorized_indexes": []}}
        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value=context)),
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.development_bootstrap._guidance_documents", return_value={"workspace": "small"}),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool("workspace_development_context", {"workspace_id": "workspace-1"})  # pyright: ignore[reportPrivateUsage]
        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertIn("guidance_documents", json.loads(content.text))

    async def test_generic_resources_operation_returns_the_bounded_reader_page(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "resources", "description": "Resources", "scope": "read", "input_schema": {}}]
        context = {"capabilities": {"authorized_tools": [{"component_id": "tool-1", "description": "x" * 70000}], "authorized_indexes": []}}

        async def execute(_principal, _workspace_id, operation, _arguments):
            self.assertEqual(operation, "context")
            return context

        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=execute),
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(
                    "workspace_development", {"workspace_id": "workspace-1", "operation": "resources", "arguments": {}}
                )  # pyright: ignore[reportPrivateUsage]
        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        page = json.loads(content.text)
        self.assertIsNone(page["items"][0]["description"])
        self.assertEqual(page["items"][0]["description_reference"]["reader"], "workspace_development_resource_description")
        self.assertLessEqual(len(content.text.encode()), 24 * 1024)

    async def test_error_response_with_huge_code_and_body_is_explicitly_truncated(self) -> None:
        result = mcp_server._development_error("x" * 100000, "\x00" * 100000)  # pyright: ignore[reportPrivateUsage]
        content = result.content[0]
        assert isinstance(content, TextContent)
        payload = json.loads(content.text)
        self.assertTrue(payload["error"]["truncated"])
        self.assertLessEqual(len(content.text.encode("utf-8")), 24 * 1024)

    async def test_large_ordinary_operation_result_preserves_existing_output_semantics(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "file_read", "description": "Read", "scope": "read", "input_schema": {}}]
        result_body = {"content": "\\n\\t" * 20000}
        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value=result_body)),
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(
                    "workspace_development", {"workspace_id": "workspace-1", "operation": "file_read", "arguments": {}}
                )  # pyright: ignore[reportPrivateUsage]
        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text), result_body)

    async def test_content_protection_denial_blocks_development_result_release(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "file_read", "description": "Read", "scope": "read", "input_schema": {}}]
        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value={"content": "restricted"})) as execute,
            mock.patch(
                "ragtime.mcp.server.authorize_external_content",
                new=mock.AsyncMock(
                    side_effect=ContentProtectionError(
                        "content_denied",
                        "request-1",
                        reason="This request conflicts with the access policy.",
                        reason_code="restricted_content",
                    )
                ),
            ),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(
                    "workspace_development", {"workspace_id": "workspace-1", "operation": "file_read", "arguments": {}}
                )  # pyright: ignore[reportPrivateUsage]

        self.assertTrue(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        detail = json.loads(content.text)["error"]
        self.assertEqual(detail["reason"], "This request conflicts with the access policy.")
        self.assertIn("next_step", detail)
        execute.assert_awaited_once_with(principal, "workspace-1", "file_read", {})

    async def test_wrapped_content_protection_detail_is_flattened_for_mcp(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        detail = ContentProtectionError(
            "content_denied",
            "request-2",
            reason="This request conflicts with the access policy.",
            reason_code="restricted_content",
        ).public_detail()
        with (
            mock.patch.object(development_service, "list_operations", return_value=[{"name": "file_read", "scope": "read"}]),
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(side_effect=HTTPException(status_code=403, detail=detail))),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(
                    "workspace_development", {"workspace_id": "workspace-1", "operation": "file_read", "arguments": {}}
                )  # pyright: ignore[reportPrivateUsage]

        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text)["error"], detail)

    async def test_oauth_user_becomes_request_local_development_principal(self) -> None:
        scope = {"type": "http", "headers": [], "_mcp_oauth_user": mock.Mock(id="user-1", role="admin"), "_mcp_user_id": "user-1"}

        principal = await mcp_routes._resolve_default_development_principal(  # pyright: ignore[reportPrivateUsage]
            scope, auth_method="oauth2", resolved_auth_method="oauth2"
        )

        self.assertEqual(principal, DevelopmentPrincipal(user_id="user-1", is_admin=True, general_tools_allowed=True))

    async def test_development_only_oauth_principal_cannot_use_general_tools(self) -> None:
        scope = {"type": "http", "headers": [], "_mcp_oauth_user": mock.Mock(id="user-1", role="user"), "_mcp_user_id": "user-1"}

        principal = await mcp_routes._resolve_default_development_principal(  # pyright: ignore[reportPrivateUsage]
            scope, auth_method="oauth2", resolved_auth_method="oauth2_development"
        )

        assert principal is not None
        self.assertFalse(principal.general_tools_allowed)
        self.assertFalse(mcp_server._general_tools_allowed(principal))  # pyright: ignore[reportPrivateUsage]

    def test_open_route_without_a_bound_principal_retains_general_tools(self) -> None:
        self.assertTrue(mcp_server._general_tools_allowed(None))  # pyright: ignore[reportPrivateUsage]

    async def test_development_only_principal_is_denied_from_registered_general_tool_handlers(self) -> None:
        server = Server("development-only-denial")
        adapter = MCPToolAdapter()
        general_tool = MCPToolDefinition(
            name="ssh_transfer",
            description="general synthetic tool",
            input_schema={},
            tool_config={},
            execute_fn=mock.AsyncMock(return_value="executed"),
        )
        mcp_server._register_handlers(server, adapter)  # pyright: ignore[reportPrivateUsage]
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False)

        with (
            mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=[general_tool])),
            mock.patch.object(adapter, "resolve_canonical_tool_id", new=mock.AsyncMock(return_value="ssh-transfer")),
            mock.patch.object(adapter, "execute_tool", new=mock.AsyncMock(return_value="executed")) as execute_tool,
            mock.patch(
                "ragtime.mcp.server._development_tools_for_principal",
                new=mock.AsyncMock(return_value=[Tool(name="workspace_development", description="dev", inputSchema={})]),
            ),
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with mcp_server.mcp_request_context(principal):
                listed = await server.request_handlers[ListToolsRequest](ListToolsRequest())
                self.assertIsInstance(listed.root, ListToolsResult)
                assert isinstance(listed.root, ListToolsResult)
                self.assertEqual([tool.name for tool in listed.root.tools], ["workspace_development"])
                for name in ("guessed_general_tool", "ssh_transfer"):
                    result = await server.request_handlers[CallToolRequest](CallToolRequest(params=CallToolRequestParams(name=name, arguments={})))
                    self.assertIsInstance(result.root, CallToolResult)
                    assert isinstance(result.root, CallToolResult)
                    self.assertTrue(result.root.isError)

        execute_tool.assert_not_awaited()

    async def test_client_credentials_never_resolve_as_development_principal(self) -> None:
        scope = {"type": "http", "headers": []}

        principal = await mcp_routes._resolve_default_development_principal(  # pyright: ignore[reportPrivateUsage]
            scope, auth_method="client_credentials", resolved_auth_method="client_credentials"
        )

        self.assertIsNone(principal)

    async def test_stateful_session_callback_recovers_verified_principal_and_route(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        session = mock.MagicMock()
        server = mock.MagicMock()
        server.request_context.session = session

        with mcp_server.mcp_request_context(principal, "default"):
            mcp_server._bind_session_request_context(server)  # pyright: ignore[reportPrivateUsage]

        self.assertIsNone(mcp_server.get_request_development_principal())
        mcp_server._bind_session_request_context(server)  # pyright: ignore[reportPrivateUsage]
        self.assertIs(mcp_server.get_request_development_principal(), principal)
        mcp_server._development_principal.set(None)  # pyright: ignore[reportPrivateUsage]
        mcp_server._mcp_route_id.set("default")  # pyright: ignore[reportPrivateUsage]

    async def test_stateful_session_explicit_none_context_replaces_cached_principal(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        session = mock.MagicMock()
        server = mock.MagicMock()
        server.request_context.session = session

        with mcp_server.mcp_request_context(principal, "default"):
            mcp_server._bind_session_request_context(server)  # pyright: ignore[reportPrivateUsage]
        with mcp_server.mcp_request_context(None, "default"):
            mcp_server._bind_session_request_context(server)  # pyright: ignore[reportPrivateUsage]

        mcp_server._bind_session_request_context(server)  # pyright: ignore[reportPrivateUsage]
        self.assertIsNone(mcp_server.get_request_development_principal())
        mcp_server._development_principal.set(None)  # pyright: ignore[reportPrivateUsage]
        mcp_server._mcp_route_id.set("default")  # pyright: ignore[reportPrivateUsage]

    async def test_concurrent_session_callbacks_keep_each_principal_isolated(self) -> None:
        first = DevelopmentPrincipal(user_id="first", is_admin=False, scopes=frozenset({"read"}))
        second = DevelopmentPrincipal(user_id="second", is_admin=False, scopes=frozenset({"read"}))

        async def bind_principal(principal: DevelopmentPrincipal) -> str:
            server = mock.MagicMock()
            server.request_context.session = mock.MagicMock()
            with mcp_server.mcp_request_context(principal, "default"):
                await asyncio.sleep(0)
                mcp_server._bind_session_request_context(server)  # pyright: ignore[reportPrivateUsage]
                bound_principal = mcp_server.get_request_development_principal()
                assert bound_principal is not None
                return str(bound_principal.user_id)

        first_user, second_user = await asyncio.gather(bind_principal(first), bind_principal(second))

        self.assertEqual((first_user, second_user), ("first", "second"))


if __name__ == "__main__":
    unittest.main()
