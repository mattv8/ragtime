import json
import unittest
from contextlib import AbstractContextManager, ExitStack
from types import SimpleNamespace
from typing import Protocol
from unittest import mock

from fastapi import HTTPException
from mcp.server import Server
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, ListToolsRequest, ListToolsResult, ServerResult, TextContent, Tool
from starlette.types import Message, Receive, Scope, Send

from ragtime.mcp import routes, server
from ragtime.mcp.tools import MCPToolAdapter, MCPToolDefinition
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_service import development_service


def _scope(path: str = "/mcp", headers: dict[str, str] | None = None) -> Scope:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(name.encode(), value.encode()) for name, value in (headers or {}).items()],
    }


async def _receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


class _AsgiEndpoint(Protocol):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...


async def _endpoint_messages(endpoint: _AsgiEndpoint, scope: Scope) -> list[Message]:
    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    await endpoint(scope, _receive, send)
    return messages


class _RegisteredHandlerManager:
    def __init__(self, registered_server: Server, request: ListToolsRequest | CallToolRequest) -> None:
        self.server = registered_server
        self.request = request
        self.result: ServerResult | None = None
        self.calls = 0

    async def handle_request(self, _scope: Scope, _receive: Receive, _send: Send) -> None:
        self.calls += 1
        self.result = await self.server.request_handlers[type(self.request)](self.request)


class _RegisteredHandlerDispatcher:
    def __init__(self, request: ListToolsRequest | CallToolRequest) -> None:
        self.request = request
        self.result: ServerResult | None = None

    async def __call__(self, registered_server: Server, _scope: Scope, _receive: Receive, _send: Send) -> None:
        self.result = await registered_server.request_handlers[type(self.request)](self.request)


def _listed_tool_names(result: ServerResult | None) -> list[str]:
    assert result is not None
    assert isinstance(result.root, ListToolsResult)
    return [tool.name for tool in result.root.tools]


def _is_error_result(result: ServerResult | None) -> bool:
    assert result is not None
    assert isinstance(result.root, CallToolResult)
    return bool(result.root.isError)


class McpByohEndpointTests(unittest.IsolatedAsyncioTestCase):
    def _server_with_general_tool(self) -> tuple[Server, MCPToolAdapter, mock.AsyncMock]:
        registered_server = Server("endpoint-test")
        adapter = MCPToolAdapter()
        execute_tool = mock.AsyncMock(return_value="executed")
        server._register_handlers(registered_server, adapter)
        return registered_server, adapter, execute_tool

    def _endpoint_patches(
        self,
        manager: _RegisteredHandlerManager,
        *,
        user: object | None,
        group_matches: object,
        password: str | None = None,
    ) -> tuple[AbstractContextManager[object], ...]:
        token_data = SimpleNamespace(user_id=getattr(user, "id", "user-1"))
        return (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch(
                "ragtime.mcp.routes._check_default_route_auth",
                mock.AsyncMock(return_value=(True, "oauth2", password, "cn=members", None)),
            ),
            mock.patch("ragtime.mcp.routes.validate_session_and_fetch_user", mock.AsyncMock(return_value=(token_data, user))),
            mock.patch("ragtime.mcp.routes.user_matches_group_identifier", mock.AsyncMock(return_value=group_matches)),
            mock.patch("ragtime.mcp.routes.get_session_manager", mock.AsyncMock(return_value=manager)),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
            mock.patch("ragtime.mcp.server.authorize_external_content", mock.AsyncMock()),
        )

    async def test_nonmember_oauth_and_legacy_session_only_reach_development_handlers(self) -> None:
        registered_server, adapter, execute_tool = self._server_with_general_tool()
        general_tool = MCPToolDefinition("execute", "general", {}, {}, execute_tool)
        user = SimpleNamespace(id="user-1", role="user", username="member")
        list_request = ListToolsRequest()
        manager = _RegisteredHandlerManager(registered_server, list_request)

        with ExitStack() as patches:
            patches.enter_context(mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=[general_tool])))
            patches.enter_context(mock.patch.object(adapter, "resolve_canonical_tool_id", new=mock.AsyncMock(return_value="execute")))
            patches.enter_context(
                mock.patch(
                    "ragtime.mcp.server._development_tools_for_principal",
                    new=mock.AsyncMock(return_value=[Tool(name="workspace_development", description="dev", inputSchema={})]),
                )
            )
            for patch in self._endpoint_patches(manager, user=user, group_matches=False):
                patches.enter_context(patch)
            messages = await _endpoint_messages(routes.MCPTransportEndpoint(), _scope(headers={"authorization": "Bearer legacy-session"}))
            self.assertIsNotNone(manager.result, (manager.calls, messages))
            self.assertEqual(_listed_tool_names(manager.result), ["workspace_development"])

            for guessed_name in ("execute", "ssh_transfer"):
                manager.request = CallToolRequest(params=CallToolRequestParams(name=guessed_name, arguments={}))
                await _endpoint_messages(routes.MCPTransportEndpoint(), _scope(headers={"authorization": "Bearer legacy-session"}))
                self.assertTrue(_is_error_result(manager.result))

        self.assertEqual(manager.calls, 3)
        execute_tool.assert_not_awaited()

    async def test_group_member_uses_filtered_handler_and_policy_is_rechecked_per_request(self) -> None:
        registered_server, adapter, execute_tool = self._server_with_general_tool()
        general_tool = MCPToolDefinition("execute", "general", {}, {}, execute_tool)
        user = SimpleNamespace(id="user-1", role="user", username="member")
        dispatcher = _RegisteredHandlerDispatcher(ListToolsRequest())
        manager = _RegisteredHandlerManager(registered_server, ListToolsRequest())
        membership = [True, False]

        async def matches(*_args: object) -> bool:
            return membership.pop(0)

        with (
            mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=[general_tool])),
            mock.patch("ragtime.mcp.routes.get_filtered_server", mock.AsyncMock(return_value=registered_server)),
            mock.patch("ragtime.mcp.routes.get_session_manager", mock.AsyncMock(return_value=manager)),
            mock.patch("ragtime.mcp.routes._get_user_matching_filter", mock.AsyncMock(return_value="filter-1")),
            mock.patch("ragtime.mcp.routes.handle_filtered_request", dispatcher),
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.mcp.routes._check_default_route_auth", mock.AsyncMock(return_value=(True, "oauth2", None, "cn=members", None))),
            mock.patch("ragtime.mcp.routes.validate_session_and_fetch_user", mock.AsyncMock(return_value=(SimpleNamespace(user_id="user-1"), user))),
            mock.patch("ragtime.mcp.routes.user_matches_group_identifier", mock.AsyncMock(side_effect=matches)),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
            mock.patch("ragtime.mcp.server.authorize_external_content", mock.AsyncMock()),
            mock.patch(
                "ragtime.mcp.server._development_tools_for_principal",
                new=mock.AsyncMock(return_value=[Tool(name="workspace_development", description="dev", inputSchema={})]),
            ),
        ):
            await _endpoint_messages(routes.MCPTransportEndpoint(), _scope(headers={"authorization": "Bearer token"}))
            self.assertEqual(_listed_tool_names(dispatcher.result), ["execute", "workspace_development"])
            await _endpoint_messages(routes.MCPTransportEndpoint(), _scope(headers={"authorization": "Bearer token"}))
            self.assertEqual(_listed_tool_names(manager.result), ["workspace_development"])

    async def test_custom_oauth_member_keeps_general_tools_and_nonmember_is_unauthorized(self) -> None:
        registered_server, adapter, _execute_tool = self._server_with_general_tool()
        general_tool = MCPToolDefinition("execute", "general", {}, {}, mock.AsyncMock(return_value="executed"))
        user = SimpleNamespace(id="user-1", role="user", username="member")
        dispatcher = _RegisteredHandlerDispatcher(ListToolsRequest())
        route_result = (registered_server, True, None, "oauth2", "cn=members", None)

        with (
            mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=[general_tool])),
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.mcp.routes.get_custom_route_server_cached", mock.AsyncMock(return_value=route_result)),
            mock.patch("ragtime.mcp.routes.validate_session_and_fetch_user", mock.AsyncMock(return_value=(SimpleNamespace(user_id="user-1"), user))),
            mock.patch("ragtime.mcp.routes.user_matches_group_identifier", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.mcp.routes.handle_filtered_request", dispatcher),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
            mock.patch("ragtime.mcp.server.authorize_external_content", mock.AsyncMock()),
        ):
            await _endpoint_messages(routes.MCPCustomRouteEndpoint(), _scope("/mcp/team", {"authorization": "Bearer token"}))
            self.assertEqual(_listed_tool_names(dispatcher.result), ["execute"])

        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.mcp.routes.get_custom_route_server_cached", mock.AsyncMock(return_value=route_result)),
            mock.patch("ragtime.mcp.routes.validate_session_and_fetch_user", mock.AsyncMock(return_value=(SimpleNamespace(user_id="user-1"), user))),
            mock.patch("ragtime.mcp.routes.user_matches_group_identifier", mock.AsyncMock(return_value=False)),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
        ):
            messages = await _endpoint_messages(routes.MCPCustomRouteEndpoint(), _scope("/mcp/team", {"authorization": "Bearer token"}))
        self.assertEqual(messages[0]["status"], 401)

    async def test_password_open_and_unstored_password_legacy_routes_keep_general_catalog(self) -> None:
        registered_server, adapter, _execute_tool = self._server_with_general_tool()
        general_tool = MCPToolDefinition("execute", "general", {}, {}, mock.AsyncMock(return_value="executed"))
        manager = _RegisteredHandlerManager(registered_server, ListToolsRequest())
        user = SimpleNamespace(id="user-1", role="user", username="member")
        with ExitStack() as patches:
            patches.enter_context(mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=[general_tool])))
            patches.enter_context(mock.patch("ragtime.mcp.routes._validate_route_password", mock.AsyncMock(return_value=True)))
            for patch in self._endpoint_patches(manager, user=user, group_matches=False, password="encrypted"):
                patches.enter_context(patch)
            await _endpoint_messages(routes.MCPTransportEndpoint(), _scope(headers={"authorization": "Bearer token", "mcp-password": "valid"}))
            self.assertEqual(_listed_tool_names(manager.result), ["execute"])

        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.mcp.routes._check_default_route_auth", mock.AsyncMock(return_value=(False, "oauth2", None, None, None))),
            mock.patch("ragtime.mcp.routes.get_session_manager", mock.AsyncMock(return_value=manager)),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
            mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=[general_tool])),
        ):
            await _endpoint_messages(routes.MCPTransportEndpoint(), _scope())
            self.assertEqual(_listed_tool_names(manager.result), ["execute"])

        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch(
                "ragtime.mcp.routes._check_default_route_auth",
                mock.AsyncMock(return_value=(True, "password", None, None, None)),
            ),
            mock.patch(
                "ragtime.mcp.routes.validate_session",
                mock.AsyncMock(return_value=SimpleNamespace(user_id="user-1")),
            ),
            mock.patch("ragtime.mcp.routes.get_session_manager", mock.AsyncMock(return_value=manager)),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
            mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=[general_tool])),
        ):
            await _endpoint_messages(routes.MCPTransportEndpoint(), _scope(headers={"authorization": "Bearer legacy-session"}))
            self.assertEqual(_listed_tool_names(manager.result), ["execute"])

    async def test_invalid_revoked_and_mismatched_development_identities_fail_closed(self) -> None:
        endpoint = routes.MCPTransportEndpoint()
        for validator_result in ((False, None, "OAuth2 token is invalid, expired, or unauthorized."), (True, "oauth2_development", None)):
            with (
                mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
                mock.patch("ragtime.mcp.routes._check_default_route_auth", mock.AsyncMock(return_value=(True, "oauth2", None, "cn=members", None))),
                mock.patch("ragtime.mcp.routes._validate_oauth2_or_password_fallback", mock.AsyncMock(return_value=validator_result)),
                mock.patch("ragtime.mcp.routes._resolve_default_development_principal", mock.AsyncMock(return_value=None)),
                mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
            ):
                messages = await _endpoint_messages(endpoint, _scope(headers={"authorization": "Bearer revoked-or-mismatched"}))
            self.assertEqual(messages[0]["status"], 401)

        manager = mock.Mock(handle_request=mock.AsyncMock())
        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.mcp.routes._check_default_route_auth", mock.AsyncMock(return_value=(True, "oauth2", None, "cn=members", None))),
            mock.patch("ragtime.mcp.routes.is_mcp_access_token_candidate", return_value=True),
            mock.patch("ragtime.mcp.routes.validate_mcp_token_and_fetch_user", mock.AsyncMock(return_value=(None, None))),
            mock.patch("ragtime.mcp.routes.get_session_manager", mock.AsyncMock(return_value=manager)),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
        ):
            messages = await _endpoint_messages(endpoint, _scope(headers={"authorization": "Bearer revoked-grant"}))
        self.assertEqual(messages[0]["status"], 401)
        manager.handle_request.assert_not_awaited()

        async def validated_nonmember(scope: Scope, **_kwargs: object) -> tuple[bool, str, None]:
            scope["_mcp_user_id"] = "verified-user"
            scope["_mcp_oauth_user"] = SimpleNamespace(id="different-user", role="user")
            return True, "oauth2_development", None

        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.mcp.routes._check_default_route_auth", mock.AsyncMock(return_value=(True, "oauth2", None, "cn=members", None))),
            mock.patch("ragtime.mcp.routes._validate_oauth2_or_password_fallback", validated_nonmember),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()),
        ):
            messages = await _endpoint_messages(endpoint, _scope(headers={"authorization": "Bearer mismatched"}))
        self.assertEqual(messages[0]["status"], 401)

    async def test_viewer_role_pipeline_allows_read_and_blocks_write_exec_and_other_workspace(self) -> None:
        principal = DevelopmentPrincipal(user_id="viewer", is_admin=False, scopes=frozenset({"read", "write", "exec"}))
        role_rank = {"viewer": 1, "editor": 2}

        async def enforce_viewer_role(workspace_id: str, user_id: str, minimum_role: str, *, is_admin: bool) -> SimpleNamespace:
            if workspace_id != "workspace-1" or user_id != "viewer" or is_admin or role_rank[minimum_role] > role_rank["viewer"]:
                raise HTTPException(403, "workspace role denied")
            return SimpleNamespace(id=workspace_id)

        role_checks = mock.AsyncMock(side_effect=enforce_viewer_role)
        audit = mock.AsyncMock()
        with (
            mock.patch("ragtime.userspace.development_service.userspace_service.enforce_workspace_role", role_checks),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", audit),
            mock.patch("ragtime.userspace.development_service.authorize_external_content", mock.AsyncMock()),
            mock.patch.object(development_service, "_context", mock.AsyncMock(return_value={"context_revision": "ok"})),
        ):
            self.assertEqual(await development_service.execute(principal, "workspace-1", "context", {}), {"context_revision": "ok"})
            for operation, arguments in (
                ("file_write", {"path": "a", "content": "x", "expected_hash": None}),
                ("exec_start", {"command": "id"}),
            ):
                with self.assertRaises(HTTPException):
                    await development_service.execute(principal, "workspace-1", operation, arguments)
            with self.assertRaises(HTTPException):
                await development_service.execute(principal, "workspace-2", "context", {})
        audit.assert_awaited_once()
        self.assertEqual(
            role_checks.await_args_list,
            [
                mock.call("workspace-1", "viewer", "viewer", is_admin=False),
                mock.call("workspace-1", "viewer", "editor", is_admin=False),
                mock.call("workspace-1", "viewer", "editor", is_admin=False),
                mock.call("workspace-2", "viewer", "viewer", is_admin=False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
