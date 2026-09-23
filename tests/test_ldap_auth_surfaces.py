"""Consumer contracts for classified shared-directory authentication failures."""

import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from ragtime import main
from ragtime.api import auth as api_auth
from ragtime.core.auth import AuthResult
from ragtime.core.ldap_errors import AuthFailureCode, auth_failure_message
from ragtime.userspace import preview_host


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "headers": [(b"host", b"ragtime.test")],
            "scheme": "https",
            "server": ("ragtime.test", 443),
            "client": ("127.0.0.1", 12345),
        }
    )


class LdapAuthenticationSurfaceTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.message = auth_failure_message(AuthFailureCode.PASSWORD_EXPIRED)
        self.failure = AuthResult(success=False, error=self.message, failure_code=AuthFailureCode.PASSWORD_EXPIRED)

    async def test_web_login_preserves_detail_envelope_and_stops_before_mfa_or_session(self) -> None:
        issue_session = mock.AsyncMock()
        mfa_needed = mock.AsyncMock()
        with (
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=self.failure)) as authenticate,
            mock.patch.object(api_auth, "_issue_login_session", new=issue_session),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mfa_needed),
        ):
            with self.assertRaises(HTTPException) as caught:
                await api_auth.login(
                    _request("/auth/login"),
                    Response(),
                    api_auth.LoginRequest(
                        username="ldap-user",
                        password="password",
                        mfa_challenge_token=None,
                        totp_code=None,
                        remember_device=False,
                    ),
                )

        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.detail, self.message)
        authenticate.assert_awaited_once_with("ldap-user", "password")
        mfa_needed.assert_not_awaited()
        issue_session.assert_not_awaited()

    async def test_authorize_form_preserves_error_envelope_and_stops_before_mfa(self) -> None:
        mfa_needed = mock.AsyncMock()
        with (
            mock.patch.object(main, "authenticate", new=mock.AsyncMock(return_value=self.failure)) as authenticate,
            mock.patch.object(main, "mfa_needed_for_user", new=mfa_needed),
            mock.patch.object(main, "validate_redirect_uri", return_value=SimpleNamespace(is_valid=True)),
            mock.patch.object(main, "_normalize_mcp_resource", new=mock.AsyncMock(return_value=None)),
        ):
            response = await main.authorize_post(
                _request("/authorize"),
                client_id="client",
                redirect_uri="https://client.test/callback",
                response_type="code",
                code_challenge="challenge",
                code_challenge_method="S256",
                state="state",
                scope=None,
                resource=None,
                username="ldap-user",
                password="password",
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(json.loads(bytes(response.body)), {"error": self.message})
        authenticate.assert_awaited_once_with("ldap-user", "password")
        mfa_needed.assert_not_awaited()

    async def test_password_grant_preserves_invalid_grant_and_stops_before_mfa_or_session(self) -> None:
        issue_session = mock.AsyncMock()
        mfa_needed = mock.AsyncMock()
        with (
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=self.failure)) as authenticate,
            mock.patch.object(api_auth, "_issue_login_session", new=issue_session),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mfa_needed),
        ):
            response = await api_auth.oauth2_token(
                request=_request("/auth/oauth2/token"),
                response=Response(),
                grant_type="password",
                username="ldap-user",
                password="password",
                totp_code=None,
                remember_device=False,
                code=None,
                code_verifier=None,
                redirect_uri=None,
                client_id=None,
                scope=None,
                resource=None,
                refresh_token=None,
            )

        self.assertIsInstance(response, Response)
        assert isinstance(response, Response)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(json.loads(bytes(response.body)), {"error": "invalid_grant", "error_description": self.message})
        authenticate.assert_awaited_once_with("ldap-user", "password")
        mfa_needed.assert_not_awaited()
        issue_session.assert_not_awaited()

    async def test_workspace_preview_preserves_detail_envelope_and_does_not_issue_session(self) -> None:
        issue_session = mock.AsyncMock()
        mfa_needed = mock.AsyncMock()
        with (
            mock.patch.object(preview_host, "authenticate", new=mock.AsyncMock(return_value=self.failure)) as authenticate,
            mock.patch.object(preview_host, "issue_authenticated_session", new=issue_session),
            mock.patch.object(preview_host, "mfa_needed_for_user", new=mfa_needed),
        ):
            with self.assertRaises(HTTPException) as caught:
                await preview_host._authenticate_preview_workspace_user(
                    _request("/auth/login"),
                    Response(),
                    "workspace-1",
                    username="ldap-user",
                    password="password",
                )

        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.detail, self.message)
        authenticate.assert_awaited_once_with("ldap-user", "password")
        mfa_needed.assert_not_awaited()
        issue_session.assert_not_awaited()
