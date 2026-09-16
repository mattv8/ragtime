import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import Request

from ragtime.core import auth
from ragtime.core.auth_policy import resolve_web_session_hours


def _request(*, headers: list[tuple[bytes, bytes]], scheme: str = "https") -> Request:
    return Request({"type": "http", "method": "GET", "scheme": scheme, "path": "/", "headers": headers})


class WebSessionPolicyTests(unittest.TestCase):
    def test_resolve_web_session_hours_prefers_valid_override(self) -> None:
        self.assertEqual(resolve_web_session_hours(SimpleNamespace(webSessionHours=72), 24), 72)

    def test_resolve_web_session_hours_falls_back_for_null_or_invalid_override(self) -> None:
        self.assertEqual(resolve_web_session_hours(SimpleNamespace(web_session_hours=None), 24), 24)
        self.assertEqual(resolve_web_session_hours(SimpleNamespace(web_session_hours=721), 24), 24)

    def test_canonical_oauth_origin_ignores_browser_origin_and_normalizes(self) -> None:
        request = _request(
            headers=[
                (b"host", b"Ragtime.Example:443"),
                (b"origin", b"https://attacker.example"),
                (b"x-forwarded-host", b"RAGTIME.EXAMPLE:443"),
                (b"x-forwarded-proto", b"HTTPS"),
            ]
        )
        with mock.patch.object(auth.settings, "external_base_url", ""):
            self.assertEqual(auth.canonical_oauth_origin(request), "https://ragtime.example")

    def test_decode_access_token_rejects_mcp_shaped_tokens(self) -> None:
        token = auth.encode_jwt_payload(
            {
                "sub": "user-1",
                "username": "alice",
                "role": "user",
                "token_use": "mcp_access",
                "grant_id": "grant-1",
                "aud": "https://ragtime.example/mcp",
                "exp": 4_102_444_800,
            }
        )
        self.assertIsNone(auth.decode_access_token(token))
