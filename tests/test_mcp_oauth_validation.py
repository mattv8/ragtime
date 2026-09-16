import base64
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from ragtime.core.auth import encode_jwt_payload
from ragtime.mcp import routes, user_oauth


def _candidate_token(payload: object) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _scope(token: str, *, password: str | None = None) -> dict:
    headers = [(b"authorization", f"Bearer {token}".encode()), (b"host", b"ragtime.example")]
    if password is not None:
        headers.append((b"mcp-password", password.encode()))
    return {"type": "http", "method": "POST", "scheme": "https", "path": "/mcp", "headers": headers}


class McpOAuthValidationTests(unittest.IsolatedAsyncioTestCase):
    def test_classifier_rejects_non_object_jwt_payloads_without_raising(self) -> None:
        self.assertFalse(user_oauth.is_mcp_access_token_candidate(_candidate_token([])))
        self.assertFalse(user_oauth.is_mcp_access_token_candidate(_candidate_token(None)))

    async def test_signed_new_shape_cannot_fall_back_when_validation_fails(self) -> None:
        scope = _scope(_candidate_token({"token_use": "mcp_access"}), password="fallback")
        with (
            mock.patch("ragtime.mcp.routes.validate_mcp_token_and_fetch_user", mock.AsyncMock(return_value=(None, None))),
            mock.patch("ragtime.mcp.routes._validate_route_password", mock.AsyncMock(return_value=True)) as password,
        ):
            valid, method, _detail = await routes._validate_oauth2_or_password_fallback(scope, allowed_group_dn=None, encrypted_password="enc::password")

        self.assertFalse(valid)
        self.assertIsNone(method)
        password.assert_not_awaited()

    async def test_signed_malformed_and_expired_mcp_tokens_cannot_fall_back(self) -> None:
        tokens = [
            encode_jwt_payload({"token_use": "mcp_access", "exp": datetime.now(timezone.utc) + timedelta(minutes=1)}),
            encode_jwt_payload({"token_use": "mcp_access", "exp": datetime.now(timezone.utc) - timedelta(minutes=1)}),
        ]
        for token in tokens:
            scope = _scope(token, password="fallback")
            with mock.patch("ragtime.mcp.routes._validate_route_password", mock.AsyncMock(return_value=True)) as password:
                valid, method, _detail = await routes._validate_oauth2_or_password_fallback(scope, allowed_group_dn=None, encrypted_password="enc::password")
            self.assertFalse(valid)
            self.assertIsNone(method)
            password.assert_not_awaited()

    async def test_normalizer_converts_malformed_urls_to_protocol_errors(self) -> None:
        for resource in ("https://[::1", "https://example.com:bad/mcp", "https://example.com\\mcp", "https://example.com/%2e%2e/mcp"):
            with self.assertRaises(user_oauth.McpOAuthError) as raised:
                await user_oauth.normalize_mcp_resource(resource, base_url="https://ragtime.example")
            self.assertEqual(raised.exception.error, "invalid_target")

    async def test_normalizer_preserves_bracketed_ipv6_resource(self) -> None:
        settings = {"mcp_enabled": True, "mcp_default_route_auth": True, "mcp_default_route_auth_method": "oauth2"}
        with mock.patch("ragtime.mcp.user_oauth.get_app_settings", mock.AsyncMock(return_value=settings)):
            resource = await user_oauth.normalize_mcp_resource("https://[2001:db8::1]/mcp/", base_url="https://[2001:db8::1]")
        self.assertEqual(resource, "https://[2001:db8::1]/mcp")
