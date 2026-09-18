"""HTTP transport coverage for interactive MCP OAuth token aliases."""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import httpx

from ragtime import main
from ragtime.api import auth as api_auth
from ragtime.core.auth import AuthResult, encode_jwt_payload
from ragtime.core.rate_limit import limiter


class McpOAuthHttpTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the production root and alias routes without application lifespan."""

    def setUp(self) -> None:
        limiter._storage.reset()

    def tearDown(self) -> None:
        limiter._storage.reset()

    async def test_password_token_aliases_omit_null_refresh_and_disable_caching(self) -> None:
        user = SimpleNamespace(id="transport-user", securityGeneration=0)
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        token_expires_at = datetime.now(timezone.utc) + timedelta(hours=48)
        web_session_token = encode_jwt_payload(
            {
                "sub": user.id,
                "username": "transport-user",
                "role": "user",
                "exp": token_expires_at,
                "mfa": False,
                "amr": ["password"],
                "security_generation": 0,
            }
        )
        auth_result = AuthResult(
            success=True,
            user_id=user.id,
            username="transport-user",
            role="user",
        )
        with (
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)),
            mock.patch.object(api_auth, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(api_auth, "_issue_login_session", new=mock.AsyncMock(return_value=web_session_token)),
        ):
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
                for path in ("/token", "/auth/oauth2/token"):
                    response = await client.post(
                        path,
                        data={
                            "grant_type": "password",
                            "username": "transport-user",
                            "password": "correct-password",
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn("refresh_token", response.json())
                    self.assertGreater(response.json()["expires_in"], 47 * 3600)
                    self.assertEqual(response.headers["cache-control"], "no-store")
                    self.assertEqual(response.headers["pragma"], "no-cache")

    async def test_mcp_access_token_is_rejected_by_web_me_route(self) -> None:
        mcp_access = encode_jwt_payload(
            {
                "token_use": "mcp_access",
                "grant_id": "transport-grant",
                "sub": "transport-user",
                "client_id": "transport-client",
                "aud": "http://ragtime.test/mcp",
                "iss": "http://ragtime.test",
                "scope": "",
                "iat": 1,
                "exp": 4_102_444_800,
                "security_generation": 0,
            }
        )
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
            response = await client.get("/auth/me", headers={"Authorization": f"Bearer {mcp_access}"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")
