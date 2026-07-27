import json
import unittest
from collections.abc import Awaitable
from typing import Protocol
from unittest import mock

from starlette.types import Message, Receive, Scope, Send

import ragtime.mcp.routes as mcp_routes


def _build_scope(path: str, headers: dict[str, str] | None = None) -> dict:
    raw_headers = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode(), value.encode()))
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": raw_headers,
    }


async def _receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


class _AsgiEndpoint(Protocol):
    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]: ...


async def _invoke_endpoint(endpoint: _AsgiEndpoint, scope: Scope) -> list[Message]:
    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    await endpoint(scope, _receive, send)
    return messages


def _response_body(messages: list[Message]) -> dict:
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return json.loads(body.decode())


class McpOauthPasswordFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_validator_rejects_correct_fallback_password_sent_only_via_bearer(self) -> None:
        scope = _build_scope("/mcp", {"authorization": "Bearer fallback-secret"})

        with (
            mock.patch("ragtime.mcp.routes._validate_oauth2_token", mock.AsyncMock(return_value=False)) as validate_oauth2,
            mock.patch("ragtime.mcp.routes.decrypt_secret", return_value="fallback-secret"),
        ):
            is_valid, resolved_auth_method, failure_detail = await mcp_routes._validate_oauth2_or_password_fallback(
                scope,
                allowed_group_dn=None,
                encrypted_password="enc::pw",
            )

        self.assertFalse(is_valid)
        self.assertIsNone(resolved_auth_method)
        self.assertEqual(failure_detail, "OAuth2 token is invalid, expired, or unauthorized.")
        validate_oauth2.assert_awaited_once_with(scope, None)

    async def test_validator_accepts_correct_fallback_password_only_via_mcp_password_header(self) -> None:
        scope = _build_scope("/mcp", {"mcp-password": "fallback-secret"})

        with (
            mock.patch("ragtime.mcp.routes._validate_oauth2_token", mock.AsyncMock()) as validate_oauth2,
            mock.patch("ragtime.mcp.routes.decrypt_secret", return_value="fallback-secret"),
        ):
            is_valid, resolved_auth_method, failure_detail = await mcp_routes._validate_oauth2_or_password_fallback(
                scope,
                allowed_group_dn=None,
                encrypted_password="enc::pw",
            )

        self.assertTrue(is_valid)
        self.assertEqual(resolved_auth_method, "password")
        self.assertIsNone(failure_detail)
        validate_oauth2.assert_not_awaited()

    async def test_password_only_validator_accepts_legacy_bearer_password(self) -> None:
        scope = _build_scope("/mcp", {"authorization": "Bearer legacy-secret"})

        with mock.patch("ragtime.mcp.routes.decrypt_secret", return_value="legacy-secret"):
            is_valid = await mcp_routes._validate_route_password(scope, "enc::pw")

        self.assertTrue(is_valid)

    async def test_validator_accepts_oauth2_and_reports_lane(self) -> None:
        scope = _build_scope("/mcp", {"authorization": "Bearer oauth-token"})

        with (
            mock.patch("ragtime.mcp.routes._validate_oauth2_token", mock.AsyncMock(return_value=True)) as validate_oauth2,
            mock.patch("ragtime.mcp.routes._validate_route_password", mock.AsyncMock(return_value=False)) as validate_password,
        ):
            is_valid, resolved_auth_method, failure_detail = await mcp_routes._validate_oauth2_or_password_fallback(
                scope,
                allowed_group_dn="cn=allowed,dc=example,dc=com",
                encrypted_password="enc::pw",
            )

        self.assertTrue(is_valid)
        self.assertEqual(resolved_auth_method, "oauth2")
        self.assertIsNone(failure_detail)
        validate_oauth2.assert_awaited_once_with(scope, "cn=allowed,dc=example,dc=com")
        validate_password.assert_not_awaited()

    async def test_validator_accepts_password_fallback_after_oauth_group_denial(self) -> None:
        scope = _build_scope(
            "/mcp",
            {
                "authorization": "Bearer oauth-token",
                "mcp-password": "fallback-secret",
            },
        )

        with (
            mock.patch("ragtime.mcp.routes._validate_oauth2_token", mock.AsyncMock(return_value=False)) as validate_oauth2,
            mock.patch("ragtime.mcp.routes._validate_route_password", mock.AsyncMock(return_value=True)) as validate_password,
        ):
            is_valid, resolved_auth_method, failure_detail = await mcp_routes._validate_oauth2_or_password_fallback(
                scope,
                allowed_group_dn="cn=restricted,dc=example,dc=com",
                encrypted_password="enc::pw",
            )

        self.assertTrue(is_valid)
        self.assertEqual(resolved_auth_method, "password")
        self.assertIsNone(failure_detail)
        validate_oauth2.assert_awaited_once_with(scope, "cn=restricted,dc=example,dc=com")
        validate_password.assert_awaited_once_with(scope, "enc::pw", allow_bearer=False)

    async def test_validator_rejects_missing_credentials(self) -> None:
        scope = _build_scope("/mcp")

        is_valid, resolved_auth_method, failure_detail = await mcp_routes._validate_oauth2_or_password_fallback(
            scope,
            allowed_group_dn=None,
            encrypted_password="enc::pw",
        )

        self.assertFalse(is_valid)
        self.assertIsNone(resolved_auth_method)
        self.assertEqual(
            failure_detail,
            "Authentication required. Use OAuth2 sign-in or the MCP-Password header.",
        )

    async def test_validator_rejects_invalid_oauth2_without_trying_bearer_password_fallback(self) -> None:
        scope = _build_scope("/mcp", {"authorization": "Bearer wrong-secret"})

        with (
            mock.patch("ragtime.mcp.routes._validate_oauth2_token", mock.AsyncMock(return_value=False)) as validate_oauth2,
            mock.patch("ragtime.mcp.routes._validate_route_password", mock.AsyncMock(return_value=False)) as validate_password,
        ):
            is_valid, resolved_auth_method, failure_detail = await mcp_routes._validate_oauth2_or_password_fallback(
                scope,
                allowed_group_dn=None,
                encrypted_password="enc::pw",
            )

        self.assertFalse(is_valid)
        self.assertIsNone(resolved_auth_method)
        self.assertEqual(failure_detail, "OAuth2 token is invalid, expired, or unauthorized.")
        validate_oauth2.assert_awaited_once_with(scope, None)
        validate_password.assert_not_awaited()

    async def test_validator_rejects_invalid_password_only(self) -> None:
        scope = _build_scope("/mcp", {"mcp-password": "wrong-secret"})

        with mock.patch("ragtime.mcp.routes._validate_route_password", mock.AsyncMock(return_value=False)) as validate_password:
            is_valid, resolved_auth_method, failure_detail = await mcp_routes._validate_oauth2_or_password_fallback(
                scope,
                allowed_group_dn=None,
                encrypted_password="enc::pw",
            )

        self.assertFalse(is_valid)
        self.assertIsNone(resolved_auth_method)
        self.assertEqual(failure_detail, "Invalid MCP-Password header.")
        validate_password.assert_awaited_once_with(scope, "enc::pw", allow_bearer=False)

    async def test_validator_rejects_both_invalid_credentials(self) -> None:
        scope = _build_scope(
            "/mcp",
            {
                "authorization": "Bearer invalid-token",
                "mcp-password": "wrong-secret",
            },
        )

        with (
            mock.patch("ragtime.mcp.routes._validate_oauth2_token", mock.AsyncMock(return_value=False)) as validate_oauth2,
            mock.patch("ragtime.mcp.routes._validate_route_password", mock.AsyncMock(return_value=False)) as validate_password,
        ):
            is_valid, resolved_auth_method, failure_detail = await mcp_routes._validate_oauth2_or_password_fallback(
                scope,
                allowed_group_dn=None,
                encrypted_password="enc::pw",
            )

        self.assertFalse(is_valid)
        self.assertIsNone(resolved_auth_method)
        self.assertEqual(failure_detail, "OAuth2 token and MCP-Password were rejected.")
        validate_oauth2.assert_awaited_once_with(scope, None)
        validate_password.assert_awaited_once_with(scope, "enc::pw", allow_bearer=False)

    async def test_default_endpoint_skips_group_filter_for_password_fallback_and_logs_password_lane(self) -> None:
        scope = _build_scope("/mcp", {"mcp-password": "fallback-secret"})
        session_manager = mock.Mock(handle_request=mock.AsyncMock())

        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch(
                "ragtime.mcp.routes._check_default_route_auth",
                mock.AsyncMock(return_value=(True, "oauth2", "enc::pw", "cn=restricted,dc=example,dc=com", None)),
            ),
            mock.patch(
                "ragtime.mcp.routes._validate_oauth2_or_password_fallback",
                mock.AsyncMock(return_value=(True, "password", None)),
            ),
            mock.patch("ragtime.mcp.routes._get_user_matching_filter", mock.AsyncMock()) as get_matching_filter,
            mock.patch("ragtime.mcp.routes.get_session_manager", mock.AsyncMock(return_value=session_manager)),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()) as log_mcp,
        ):
            messages = await _invoke_endpoint(mcp_routes.MCPTransportEndpoint(), scope)

        self.assertEqual(messages, [])
        get_matching_filter.assert_not_awaited()
        session_manager.handle_request.assert_awaited_once()
        log_mcp.assert_awaited_once_with(scope, auth_method="password", status_code=200)

    async def test_default_endpoint_uses_group_filtered_server_and_logs_oauth2_lane(self) -> None:
        scope = _build_scope("/mcp", {"authorization": "Bearer oauth-token"})
        filtered_server = mock.sentinel.filtered_server
        session_manager = mock.Mock(handle_request=mock.AsyncMock())

        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch(
                "ragtime.mcp.routes._check_default_route_auth",
                mock.AsyncMock(return_value=(True, "oauth2", "enc::pw", "cn=restricted,dc=example,dc=com", None)),
            ),
            mock.patch(
                "ragtime.mcp.routes._validate_oauth2_or_password_fallback",
                mock.AsyncMock(return_value=(True, "oauth2", None)),
            ),
            mock.patch("ragtime.mcp.routes._get_user_matching_filter", mock.AsyncMock(return_value="filter-123")) as get_matching_filter,
            mock.patch("ragtime.mcp.routes.get_filtered_server", mock.AsyncMock(return_value=filtered_server)) as get_filtered_server,
            mock.patch("ragtime.mcp.routes.handle_filtered_request", mock.AsyncMock()) as handle_filtered_request,
            mock.patch("ragtime.mcp.routes.get_session_manager", mock.AsyncMock(return_value=session_manager)),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()) as log_mcp,
        ):
            messages = await _invoke_endpoint(mcp_routes.MCPTransportEndpoint(), scope)

        self.assertEqual(messages, [])
        get_matching_filter.assert_awaited_once_with(scope)
        get_filtered_server.assert_awaited_once_with("filter-123")
        handle_filtered_request.assert_awaited_once()
        session_manager.handle_request.assert_not_awaited()
        log_mcp.assert_awaited_once_with(scope, auth_method="oauth2")

    async def test_check_default_route_auth_uses_shared_default_auth_method_constant(self) -> None:
        with mock.patch("ragtime.mcp.routes.get_app_settings", mock.AsyncMock(return_value={})):
            require_auth, auth_method, encrypted_password, allowed_group, client_id = await mcp_routes._check_default_route_auth()

        self.assertFalse(require_auth)
        self.assertEqual(auth_method, "oauth2")
        self.assertIsNone(encrypted_password)
        self.assertIsNone(allowed_group)
        self.assertIsNone(client_id)

    async def test_default_endpoint_returns_dual_auth_challenge_and_failure_detail(self) -> None:
        scope = _build_scope("/mcp", {"authorization": "Bearer wrong-token"})

        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch(
                "ragtime.mcp.routes._check_default_route_auth",
                mock.AsyncMock(return_value=(True, "oauth2", "enc::pw", None, None)),
            ),
            mock.patch(
                "ragtime.mcp.routes._validate_oauth2_or_password_fallback",
                mock.AsyncMock(return_value=(False, None, "OAuth2 token is invalid, expired, or unauthorized.")),
            ),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()) as log_mcp,
        ):
            messages = await _invoke_endpoint(mcp_routes.MCPTransportEndpoint(), scope)

        start = messages[0]
        self.assertEqual(start["status"], 401)
        self.assertIn((b"www-authenticate", b'Bearer realm="mcp", MCP-Password'), start["headers"])
        self.assertEqual(
            _response_body(messages),
            {
                "error": "Unauthorized",
                "detail": "OAuth2 token is invalid, expired, or unauthorized.",
            },
        )
        log_mcp.assert_awaited_once_with(scope, auth_method="none", status_code=401)

    async def test_custom_endpoint_returns_dual_auth_challenge_and_failure_detail(self) -> None:
        scope = _build_scope("/mcp/team-tools", {"mcp-password": "wrong-secret"})

        with (
            mock.patch("ragtime.mcp.routes._check_mcp_enabled", mock.AsyncMock(return_value=True)),
            mock.patch(
                "ragtime.mcp.routes.get_custom_route_server_cached",
                mock.AsyncMock(return_value=(mock.sentinel.server, True, "enc::pw", "oauth2", None, None)),
            ),
            mock.patch(
                "ragtime.mcp.routes._validate_oauth2_or_password_fallback",
                mock.AsyncMock(return_value=(False, None, "Invalid MCP-Password header.")),
            ),
            mock.patch("ragtime.mcp.routes._log_mcp", mock.AsyncMock()) as log_mcp,
        ):
            messages = await _invoke_endpoint(mcp_routes.MCPCustomRouteEndpoint(), scope)

        start = messages[0]
        self.assertEqual(start["status"], 401)
        self.assertIn((b"www-authenticate", b'Bearer realm="mcp", MCP-Password'), start["headers"])
        self.assertEqual(
            _response_body(messages),
            {
                "error": "Unauthorized",
                "detail": "Invalid MCP-Password header.",
            },
        )
        log_mcp.assert_awaited_once_with(
            scope,
            route_name="team-tools",
            auth_method="none",
            status_code=401,
        )


if __name__ == "__main__":
    unittest.main()
