from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

import ragtime.userspace.preview_host as _PREVIEW_HOST
import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES
from ragtime.core.user_identity import add_workspace_user_fingerprint, build_user_fingerprint_subject, build_workspace_user_fingerprint
from ragtime.userspace.models import UserSpaceBrowserAuthRequest
from ragtime.userspace.service import UserSpaceService


def _build_request(
    path: str = "/__ragtime/browser-auth",
    headers: list[tuple[bytes, bytes]] | None = None,
    *,
    method: str = "POST",
    query_string: str | bytes = b"",
    body: bytes | None = None,
) -> Request:
    request_headers = headers or [(b"host", b"workspace.ragtime.test")]
    raw_query = query_string.encode("utf-8") if isinstance(query_string, str) else query_string
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": raw_query,
        "headers": request_headers,
        "scheme": "https",
        "server": ("workspace.ragtime.test", 443),
        "client": ("127.0.0.1", 12345),
    }
    if body is None:
        return Request(scope)

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


class _FakeRuntimeService:
    def __init__(self) -> None:
        self.issued: list[tuple[str, str, list[str], str | None]] = []

    async def issue_capability_token(
        self,
        workspace_id: str,
        user_id: str,
        capabilities: list[str],
        session_id: str | None = None,
    ) -> SimpleNamespace:
        self.issued.append((workspace_id, user_id, capabilities, session_id))
        return SimpleNamespace(
            token="token-value",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            workspace_id=workspace_id,
            session_id=session_id,
            capabilities=capabilities,
        )


class _FakePreviewRuntimeService(_FakeRuntimeService):
    def __init__(self) -> None:
        super().__init__()
        self.verified_capability_token: tuple[str, str, str] | None = None

    def build_preview_session_token(self, claims: dict[str, Any]) -> tuple[str, datetime]:
        self.preview_session_claims = claims
        return "preview-session-token", datetime.now(timezone.utc) + timedelta(minutes=15)

    def verify_capability_token(self, token: str, workspace_id: str, capability: str) -> dict[str, Any]:
        self.verified_capability_token = (token, workspace_id, capability)
        return {"workspace_id": workspace_id, "sub": "user-1", "capabilities": [capability]}


class _FakeProxyUpstreamResponse:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self._body = body
        self.headers = {"content-type": content_type}
        self.status_code = 200
        self.closed = False

    async def aread(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        self.closed = True


class _FakeProxyClient:
    def __init__(self, response: _FakeProxyUpstreamResponse) -> None:
        self.response = response
        self.closed = False

    def build_request(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    async def send(self, request: Any, stream: bool = False) -> _FakeProxyUpstreamResponse:
        return self.response

    async def aclose(self) -> None:
        self.closed = True


class UserspaceAuthPrimitiveTests(unittest.IsolatedAsyncioTestCase):
    def test_workspace_user_fingerprint_is_stable_and_workspace_scoped(self) -> None:
        first = build_workspace_user_fingerprint(user_id="user-1", workspace_id="workspace-a", workspace_fingerprint_namespace="portable-a")
        second = build_workspace_user_fingerprint(user_id="user-1", workspace_id="workspace-a", workspace_fingerprint_namespace="portable-a")
        other_workspace = build_workspace_user_fingerprint(user_id="user-1", workspace_id="workspace-b", workspace_fingerprint_namespace="portable-b")
        other_user = build_workspace_user_fingerprint(user_id="user-2", workspace_id="workspace-a", workspace_fingerprint_namespace="portable-a")

        self.assertTrue(first.startswith("v1:"))
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_workspace)
        self.assertNotEqual(first, other_user)

    def test_workspace_user_fingerprint_portable_across_imported_workspace_ids(self) -> None:
        subject = build_user_fingerprint_subject(
            user_id="local-db-id-a",
            username="jane@example.com",
            auth_provider="ldap",
        )

        first = build_workspace_user_fingerprint(
            user_id="local-db-id-a",
            workspace_id="source-workspace-id",
            workspace_fingerprint_namespace="portable-namespace",
            user_identity_subject=subject,
        )
        imported = build_workspace_user_fingerprint(
            user_id="local-db-id-b",
            workspace_id="imported-workspace-id",
            workspace_fingerprint_namespace="portable-namespace",
            user_identity_subject=subject,
        )

        self.assertEqual(first, imported)

    def test_user_fingerprint_subject_prefers_ldap_dn_before_username(self) -> None:
        subject = build_user_fingerprint_subject(
            user_id="local-db-id-a",
            username="jane@example.com",
            auth_provider="ldap",
            ldap_dn="CN=Jane Example,OU=People,DC=example,DC=com",
        )

        self.assertEqual(subject, "ldap:ldap_dn:cn=jane example,ou=people,dc=example,dc=com")

    def test_add_workspace_user_fingerprint_preserves_existing_payload(self) -> None:
        payload = add_workspace_user_fingerprint(
            {"action": "save"},
            user_id="user-1",
            workspace_id="workspace-a",
            workspace_fingerprint_namespace="portable-a",
        )

        self.assertEqual(payload["action"], "save")
        self.assertEqual(payload["user_fingerprint_scope"], "workspace")
        self.assertTrue(str(payload["user_fingerprint"]).startswith("v1:"))

    async def test_primitive_session_payload_exposes_auth_methods_and_fingerprint(self) -> None:
        async def fake_capabilities(workspace_id: str, user_id: str | None, *, preview_mode: str) -> dict[str, Any]:
            return {"workspace_id": workspace_id, "mode": preview_mode, "can_read_files": True}

        fake_service = SimpleNamespace(get_workspace_audit_fingerprint_namespace=mock.AsyncMock(return_value="portable-a"))
        with (
            mock.patch.object(_RUNTIME_ROUTES, "_primitive_capabilities", new=fake_capabilities),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=fake_service),
            mock.patch.object(
                _RUNTIME_ROUTES,
                "build_auth_method_statuses",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "key": "local_managed",
                            "label": "Internal Users",
                            "configured": True,
                            "available": True,
                            "status": "available",
                            "detail": "Ready",
                        }
                    ]
                ),
            ),
        ):
            payload = await _RUNTIME_ROUTES._primitive_session_payload(
                "workspace-a",
                "user-1",
                mode="workspace",
            )

        self.assertEqual(payload["workspace_id"], "workspace-a")
        self.assertEqual(payload["mode"], "workspace")
        self.assertTrue(str(payload["user_fingerprint"]).startswith("v1:"))
        self.assertEqual(payload["user_fingerprint_scope"], "workspace")
        self.assertTrue(payload["auth"]["authenticated"])
        self.assertEqual(payload["auth"]["binding_strategy"], "browser_capability_token")
        self.assertEqual(payload["auth"]["interactive_auth_endpoint"], "/__ragtime/browser-auth/start")
        self.assertEqual(payload["auth"]["login_endpoint"], "/auth/login")
        self.assertEqual(payload["auth"]["current_user_endpoint"], "/auth/me")
        self.assertEqual(payload["auth"]["methods"][0]["key"], "local_managed")
        self.assertEqual(payload["auth"]["methods"][0]["binding_surfaces"], ["collab", "runtime_pty"])
        self.assertEqual(payload["auth"]["browser_logout_endpoint"], "/indexes/userspace/runtime/workspaces/workspace-a/browser-auth/logout")

    async def test_same_origin_primitive_session_payload_uses_preview_auth_endpoints(self) -> None:
        async def fake_capabilities(workspace_id: str, user_id: str | None, *, preview_mode: str) -> dict[str, Any]:
            return {"workspace_id": workspace_id, "mode": preview_mode, "can_read_files": True}

        fake_service = SimpleNamespace(get_workspace_audit_fingerprint_namespace=mock.AsyncMock(return_value="portable-a"))
        with (
            mock.patch.object(_RUNTIME_ROUTES, "_primitive_capabilities", new=fake_capabilities),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=fake_service),
            mock.patch.object(_RUNTIME_ROUTES, "build_auth_method_statuses", new=mock.AsyncMock(return_value=[])),
        ):
            payload = await _RUNTIME_ROUTES._primitive_session_payload(
                "workspace-a",
                "user-1",
                mode="workspace",
                same_origin_auth_endpoints=True,
            )

        self.assertEqual(payload["auth"]["browser_auth_endpoint"], "/__ragtime/browser-auth")
        self.assertEqual(payload["auth"]["browser_logout_endpoint"], "/__ragtime/browser-auth/logout")

    async def test_primitive_session_payload_exposes_auth_user_display_identity(self) -> None:
        async def fake_capabilities(workspace_id: str, user_id: str | None, *, preview_mode: str) -> dict[str, Any]:
            return {"workspace_id": workspace_id, "mode": preview_mode, "can_read_files": True}

        user = SimpleNamespace(
            username="local:jane",
            displayName="Jane Example",
            email="jane@example.com",
            authProvider="local_managed",
        )
        fake_service = SimpleNamespace(get_workspace_audit_fingerprint_namespace=mock.AsyncMock(return_value="portable-a"))
        with (
            mock.patch.object(_RUNTIME_ROUTES, "_primitive_capabilities", new=fake_capabilities),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=fake_service),
            mock.patch.object(_RUNTIME_ROUTES, "build_auth_method_statuses", new=mock.AsyncMock(return_value=[])),
        ):
            payload = await _RUNTIME_ROUTES._primitive_session_payload(
                "workspace-a",
                "user-1",
                mode="workspace",
                user=user,
            )

        auth_user = payload["auth"]["user"]
        self.assertEqual(auth_user["display_name"], "Jane Example")
        self.assertEqual(auth_user["username"], "jane")
        self.assertEqual(auth_user["email"], "jane@example.com")
        self.assertEqual(auth_user["auth_provider"], "local_managed")
        self.assertEqual(auth_user["user_fingerprint"], payload["user_fingerprint"])

    async def test_primitive_session_payload_uses_username_as_display_fallback(self) -> None:
        async def fake_capabilities(workspace_id: str, user_id: str | None, *, preview_mode: str) -> dict[str, Any]:
            return {"workspace_id": workspace_id, "mode": preview_mode, "can_read_files": True}

        user = SimpleNamespace(username="jane@example.com", displayName="", email=None, authProvider="ldap")
        fake_service = SimpleNamespace(get_workspace_audit_fingerprint_namespace=mock.AsyncMock(return_value="portable-a"))
        with (
            mock.patch.object(_RUNTIME_ROUTES, "_primitive_capabilities", new=fake_capabilities),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=fake_service),
            mock.patch.object(_RUNTIME_ROUTES, "build_auth_method_statuses", new=mock.AsyncMock(return_value=[])),
        ):
            payload = await _RUNTIME_ROUTES._primitive_session_payload(
                "workspace-a",
                "user-1",
                mode="workspace",
                user=user,
            )

        self.assertEqual(payload["auth"]["user"]["display_name"], "jane@example.com")
        self.assertEqual(payload["auth"]["user"]["username"], "jane@example.com")

    async def test_anonymous_preview_session_payload_has_same_origin_auth_shape(self) -> None:
        async def fake_capabilities(workspace_id: str, user_id: str | None, *, preview_mode: str) -> dict[str, Any]:
            return {"workspace_id": workspace_id, "mode": preview_mode, "can_read_files": False}

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_primitive_capabilities", new=fake_capabilities),
            mock.patch.object(
                _RUNTIME_ROUTES,
                "build_auth_method_statuses",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "key": "ldap",
                            "label": "LDAP",
                            "configured": True,
                            "available": True,
                            "status": "available",
                            "detail": "Reachable",
                        }
                    ]
                ),
            ),
        ):
            payload = await _RUNTIME_ROUTES._primitive_session_payload(
                "workspace-a",
                None,
                mode="shared_public_host",
                share_access_mode="token",
            )

        self.assertFalse(payload["auth"]["authenticated"])
        self.assertIsNone(payload["auth"]["user"])
        self.assertIsNone(payload["user_id"])
        self.assertIsNone(payload["user_fingerprint"])
        self.assertEqual(payload["auth"]["login_endpoint"], "/auth/login")
        self.assertEqual(payload["auth"]["interactive_auth_endpoint"], "/__ragtime/browser-auth/start")
        self.assertEqual(payload["auth"]["current_user_endpoint"], "/auth/me")
        self.assertEqual(payload["auth"]["browser_auth_endpoint"], "/__ragtime/browser-auth")
        self.assertEqual(payload["auth"]["browser_logout_endpoint"], "/__ragtime/browser-auth/logout")
        self.assertEqual(payload["auth"]["methods"][0]["binding_surfaces"], ["collab", "runtime_pty"])

    async def test_browser_auth_rejects_unavailable_method(self) -> None:
        with mock.patch.object(
            _RUNTIME_ROUTES,
            "build_auth_method_statuses",
            new=mock.AsyncMock(
                return_value=[
                    {
                        "key": "ldap",
                        "label": "LDAP",
                        "configured": True,
                        "available": False,
                        "status": "unavailable",
                        "detail": "Connection failed",
                    }
                ]
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await _RUNTIME_ROUTES.authorize_browser_surfaces(
                    "workspace-a",
                    UserSpaceBrowserAuthRequest(auth_method_key="ldap", surfaces=["collab"]),
                    _build_request(),
                    Response(),
                    SimpleNamespace(id="user-1"),
                )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("not available", str(raised.exception.detail))

    async def test_browser_auth_issues_scoped_cookie_for_available_method(self) -> None:
        fake_runtime = _FakeRuntimeService()
        response = Response()

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(
                _RUNTIME_ROUTES,
                "build_auth_method_statuses",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "key": "local_managed",
                            "label": "Internal Users",
                            "configured": True,
                            "available": True,
                            "status": "available",
                            "detail": "Ready",
                        }
                    ]
                ),
            ),
        ):
            result = await _RUNTIME_ROUTES.authorize_browser_surfaces(
                "workspace-a",
                UserSpaceBrowserAuthRequest(auth_method_key="local_managed", surfaces=["collab"]),
                _build_request(),
                response,
                SimpleNamespace(id="user-1"),
            )

        self.assertEqual(result.auth_method_key, "local_managed")
        self.assertEqual(result.authorizations[0].auth_method_key, "local_managed")
        self.assertEqual(fake_runtime.issued[0][2], ["userspace.collab_connect"])
        self.assertIn("userspace_collab_capability=token-value", response.headers.get("set-cookie", ""))

    async def test_preview_session_reuses_primitive_session_payload(self) -> None:
        request = _build_request("/__ragtime/session")

        with (
            mock.patch.object(
                _PREVIEW_HOST,
                "_verify_preview_session_cookie",
                new=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "sub": "user-1",
                        "preview_mode": "workspace",
                    }
                ),
            ),
            mock.patch.object(
                _PREVIEW_HOST,
                "_primitive_session_payload",
                new=mock.AsyncMock(return_value={"workspace_id": "workspace-a", "user_fingerprint": "v1:abc"}),
            ) as primitive_session,
            mock.patch.object(_PREVIEW_HOST, "_preview_user_from_id", new=mock.AsyncMock(return_value="preview-user")),
        ):
            payload = await _PREVIEW_HOST.preview_session(request)

        self.assertEqual(payload["user_fingerprint"], "v1:abc")
        primitive_session.assert_awaited_once_with(
            "workspace-a",
            "user-1",
            mode="workspace",
            share_access_mode=None,
            user="preview-user",
            same_origin_auth_endpoints=True,
        )

    async def test_preview_browser_auth_upgrades_anonymous_subdomain_with_session_token(self) -> None:
        fake_runtime = _FakePreviewRuntimeService()
        response = Response()
        request = _build_request(headers=[(b"host", b"workspace-a.ragtime.test"), (b"authorization", b"Bearer root-session")])
        user = SimpleNamespace(id="user-1")
        fake_service = SimpleNamespace(enforce_workspace_role=mock.AsyncMock())

        with (
            mock.patch.object(
                _PREVIEW_HOST,
                "_verify_preview_session_cookie",
                new=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "preview_mode": "shared_public_host",
                    }
                ),
            ),
            mock.patch.object(_PREVIEW_HOST, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(_PREVIEW_HOST, "_userspace_service", return_value=fake_service),
            mock.patch.object(_PREVIEW_HOST, "validate_session_and_fetch_user", new=mock.AsyncMock(return_value=(object(), user))),
            mock.patch.object(_PREVIEW_HOST, "_register_preview_session", new=mock.AsyncMock()),
            mock.patch.object(
                _RUNTIME_ROUTES,
                "build_auth_method_statuses",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "key": "local_managed",
                            "label": "Internal Users",
                            "configured": True,
                            "available": True,
                            "status": "available",
                            "detail": "Ready",
                        }
                    ]
                ),
            ),
        ):
            result = await _PREVIEW_HOST.preview_browser_auth(
                request,
                response,
                UserSpaceBrowserAuthRequest(auth_method_key="local_managed", surfaces=["collab"]),
            )

        self.assertEqual(result.workspace_id, "workspace-a")
        self.assertEqual(fake_runtime.preview_session_claims["preview_mode"], "workspace")
        self.assertEqual(fake_runtime.issued[0][0:3], ("workspace-a", "user-1", ["userspace.collab_connect"]))
        set_cookie = "\n".join(value.decode("latin-1") for key, value in response.raw_headers if key == b"set-cookie")
        self.assertIn("userspace_preview_session=preview-session-token", set_cookie)
        self.assertIn("userspace_collab_capability=token-value", set_cookie)
        fake_service.enforce_workspace_role.assert_awaited_once_with("workspace-a", "user-1", "viewer")

    async def test_preview_browser_auth_accepts_direct_preview_capability_token(self) -> None:
        fake_runtime = _FakePreviewRuntimeService()
        response = Response()
        request = _build_request(headers=[(b"host", b"workspace-a.ragtime.test"), (b"x-userspace-preview-capability-token", b"preview-cap")])
        fake_service = SimpleNamespace(enforce_workspace_role=mock.AsyncMock())

        with (
            mock.patch.object(
                _PREVIEW_HOST,
                "_verify_preview_session_cookie",
                new=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "preview_mode": "shared_public_host",
                    }
                ),
            ),
            mock.patch.object(_PREVIEW_HOST, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(_PREVIEW_HOST, "_userspace_service", return_value=fake_service),
            mock.patch.object(_PREVIEW_HOST, "validate_session_and_fetch_user", new=mock.AsyncMock(return_value=(None, None))),
            mock.patch.object(_PREVIEW_HOST, "_preview_user_from_id", new=mock.AsyncMock(return_value=SimpleNamespace(id="user-1"))),
            mock.patch.object(_PREVIEW_HOST, "_register_preview_session", new=mock.AsyncMock()),
            mock.patch.object(
                _RUNTIME_ROUTES,
                "build_auth_method_statuses",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "key": "local_managed",
                            "label": "Internal Users",
                            "configured": True,
                            "available": True,
                            "status": "available",
                            "detail": "Ready",
                        }
                    ]
                ),
            ),
        ):
            result = await _PREVIEW_HOST.preview_browser_auth(
                request,
                response,
                UserSpaceBrowserAuthRequest(auth_method_key="local_managed", surfaces=["collab"]),
            )

        self.assertEqual(result.workspace_id, "workspace-a")
        self.assertEqual(fake_runtime.verified_capability_token, ("preview-cap", "workspace-a", "userspace.preview_session"))
        self.assertEqual(fake_runtime.issued[0][1], "user-1")

    async def test_preview_browser_auth_accepts_credentials_for_anonymous_subdomain(self) -> None:
        fake_runtime = _FakePreviewRuntimeService()
        response = Response()
        request = _build_request(headers=[(b"host", b"workspace-a.ragtime.test")])
        user = SimpleNamespace(
            id="user-1",
            username="jane@example.com",
            displayName="Jane Example",
            email="jane@example.com",
            role="user",
            authProvider="ldap",
        )
        fake_service = SimpleNamespace(enforce_workspace_role=mock.AsyncMock())

        with (
            mock.patch.object(
                _PREVIEW_HOST,
                "_verify_preview_session_cookie",
                new=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "preview_mode": "shared_public_host",
                    }
                ),
            ),
            mock.patch.object(_PREVIEW_HOST, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(_PREVIEW_HOST, "_userspace_service", return_value=fake_service),
            mock.patch.object(
                _PREVIEW_HOST,
                "authenticate",
                new=mock.AsyncMock(return_value=SimpleNamespace(success=True, user_id="user-1", username="jane@example.com", role="user")),
            ),
            mock.patch.object(_PREVIEW_HOST, "create_access_token", return_value="root-session-token"),
            mock.patch.object(_PREVIEW_HOST, "create_session", new=mock.AsyncMock()),
            mock.patch.object(_PREVIEW_HOST, "_preview_user_from_id", new=mock.AsyncMock(return_value=user)),
            mock.patch.object(_PREVIEW_HOST, "_register_preview_session", new=mock.AsyncMock()),
            mock.patch.object(
                _RUNTIME_ROUTES,
                "build_auth_method_statuses",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "key": "ldap",
                            "label": "LDAP",
                            "configured": True,
                            "available": True,
                            "status": "available",
                            "detail": "Reachable",
                        }
                    ]
                ),
            ),
        ):
            result = await _PREVIEW_HOST.preview_browser_auth(
                request,
                response,
                UserSpaceBrowserAuthRequest(auth_method_key="ldap", username="jane@example.com", password="secret", surfaces=["collab"]),
            )

        self.assertEqual(result.workspace_id, "workspace-a")
        self.assertEqual(result.auth_method_key, "ldap")
        self.assertEqual(fake_runtime.preview_session_claims["preview_mode"], "workspace")
        self.assertEqual(fake_runtime.preview_session_claims["sub"], "user-1")
        set_cookie = "\n".join(value.decode("latin-1") for key, value in response.raw_headers if key == b"set-cookie")
        self.assertIn("ragtime_session=root-session-token", set_cookie)
        self.assertIn("userspace_preview_session=preview-session-token", set_cookie)
        self.assertIn("userspace_collab_capability=token-value", set_cookie)

    async def test_preview_browser_auth_without_user_returns_interactive_url(self) -> None:
        request = _build_request(headers=[(b"host", b"workspace-a.ragtime.test")])

        with mock.patch.object(
            _PREVIEW_HOST,
            "_verify_preview_session_cookie",
            new=mock.AsyncMock(
                return_value={
                    "workspace_id": "workspace-a",
                    "preview_mode": "shared_public_host",
                }
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await _PREVIEW_HOST.preview_browser_auth(
                    request,
                    Response(),
                    UserSpaceBrowserAuthRequest(auth_method_key="local", surfaces=["collab", "runtime_pty"]),
                )

        self.assertEqual(raised.exception.status_code, 401)
        detail = raised.exception.detail
        self.assertIsInstance(detail, dict)
        detail_payload = cast(dict[str, Any], detail)
        self.assertEqual(detail_payload["code"], "authentication_required")
        self.assertIn("/__ragtime/browser-auth/start?", detail_payload["interactive_auth_url"])
        self.assertIn("auth_method_key=local", detail_payload["interactive_auth_url"])
        self.assertIn("surfaces=collab", detail_payload["interactive_auth_url"])

    async def test_preview_browser_auth_start_renders_platform_login_when_anonymous(self) -> None:
        request = _build_request(
            "/__ragtime/browser-auth/start",
            headers=[(b"host", b"workspace-a.ragtime.test")],
            method="GET",
            query_string="surfaces=collab&auth_method_key=ldap&return_to=/dashboard",
        )

        with (
            mock.patch.object(
                _PREVIEW_HOST,
                "_verify_preview_session_cookie",
                new=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "preview_mode": "shared_public_host",
                    }
                ),
            ),
            mock.patch.object(
                _PREVIEW_HOST,
                "build_auth_method_statuses",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "key": "ldap",
                            "label": "LDAP",
                            "configured": True,
                            "available": True,
                            "status": "available",
                            "detail": "Reachable",
                        }
                    ]
                ),
            ),
        ):
            response = await _PREVIEW_HOST.preview_browser_auth_start(request)

        self.assertEqual(response.status_code, 200)
        body = bytes(response.body).decode("utf-8")
        self.assertIn("Sign in to Ragtime", body)
        self.assertIn("/__ragtime/browser-auth/start", body)
        self.assertIn('name="return_to" value="/dashboard"', body)

    async def test_preview_browser_auth_start_submit_mints_cookies_and_redirects(self) -> None:
        fake_runtime = _FakePreviewRuntimeService()
        body = b"surfaces=collab&auth_method_key=local&return_to=%2Fdashboard&username=admin&password=secret"
        request = _build_request(
            "/__ragtime/browser-auth/start",
            headers=[
                (b"host", b"workspace-a.ragtime.test"),
                (b"content-type", b"application/x-www-form-urlencoded"),
            ],
            method="POST",
            body=body,
        )
        user = SimpleNamespace(id="user-1", username="local:admin", role="admin")
        fake_service = SimpleNamespace(enforce_workspace_role=mock.AsyncMock())

        with (
            mock.patch.object(
                _PREVIEW_HOST,
                "_verify_preview_session_cookie",
                new=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "preview_mode": "shared_public_host",
                    }
                ),
            ),
            mock.patch.object(_PREVIEW_HOST, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=fake_runtime),
            mock.patch.object(_PREVIEW_HOST, "_userspace_service", return_value=fake_service),
            mock.patch.object(
                _PREVIEW_HOST,
                "authenticate",
                new=mock.AsyncMock(return_value=SimpleNamespace(success=True, user_id="user-1", username="local:admin", role="admin")),
            ),
            mock.patch.object(_PREVIEW_HOST, "create_access_token", return_value="root-session-token"),
            mock.patch.object(_PREVIEW_HOST, "create_session", new=mock.AsyncMock()),
            mock.patch.object(_PREVIEW_HOST, "_preview_user_from_id", new=mock.AsyncMock(return_value=user)),
            mock.patch.object(_PREVIEW_HOST, "_register_preview_session", new=mock.AsyncMock()),
            mock.patch.object(
                _RUNTIME_ROUTES,
                "build_auth_method_statuses",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "key": "local",
                            "label": "Local Admin",
                            "configured": True,
                            "available": True,
                            "status": "available",
                            "detail": "Ready",
                        }
                    ]
                ),
            ),
        ):
            response = await _PREVIEW_HOST.preview_browser_auth_start_submit(request)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/dashboard")
        set_cookie = "\n".join(value.decode("latin-1") for key, value in response.raw_headers if key == b"set-cookie")
        self.assertIn("ragtime_session=root-session-token", set_cookie)
        self.assertIn("userspace_preview_session=preview-session-token", set_cookie)
        self.assertIn("userspace_collab_capability=token-value", set_cookie)

    async def test_auth_required_html_redirects_before_app_content_is_returned(self) -> None:
        html = b"""
        <html><head>
          <meta name="ragtime-auth" content="required; surfaces=collab,runtime_pty; auth_method_key=local">
          <script src="/main.js"></script>
        </head><body>Protected dashboard contents</body></html>
        """
        upstream = _FakeProxyUpstreamResponse(html)
        request = _build_request(
            "/dashboard",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"accept", b"text/html")],
            method="GET",
            body=b"",
        )

        with (
            mock.patch.object(_RUNTIME_ROUTES.httpx, "AsyncClient", return_value=_FakeProxyClient(upstream)),
            mock.patch.object(_RUNTIME_ROUTES, "get_app_settings", new=mock.AsyncMock(return_value={"userspace_preview_sandbox_flags": []})),
        ):
            response = await _RUNTIME_ROUTES._proxy_http_request(
                request,
                "http://runtime/dashboard",
                primitive_session_factory=mock.AsyncMock(return_value={"auth": {"authenticated": False}}),
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/__ragtime/browser-auth/start?", response.headers["location"])
        self.assertIn("auth_method_key=local", response.headers["location"])
        self.assertIn("return_to=%2Fdashboard", response.headers["location"])
        self.assertNotIn("Protected dashboard contents", bytes(response.body).decode("utf-8", errors="ignore"))

    async def test_authenticated_auth_required_html_injects_session_before_app_scripts(self) -> None:
        html = b"""
        <html><head>
          <meta name="ragtime-auth" content="required; surfaces=collab">
          <script src="/main.js"></script>
        </head><body>Protected dashboard contents</body></html>
        """
        upstream = _FakeProxyUpstreamResponse(html)
        request = _build_request(
            "/dashboard",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"accept", b"text/html")],
            method="GET",
            body=b"",
        )

        with (
            mock.patch.object(_RUNTIME_ROUTES.httpx, "AsyncClient", return_value=_FakeProxyClient(upstream)),
            mock.patch.object(_RUNTIME_ROUTES, "get_app_settings", new=mock.AsyncMock(return_value={"userspace_preview_sandbox_flags": []})),
        ):
            response = await _RUNTIME_ROUTES._proxy_http_request(
                request,
                "http://runtime/dashboard",
                bridge_workspace_id="workspace-a",
                primitive_session_factory=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "user_id": "user-1",
                        "auth": {"authenticated": True},
                    }
                ),
            )

        self.assertEqual(response.status_code, 200)
        body = bytes(response.body).decode("utf-8")
        self.assertIn("window.__ragtime_session=", body)
        self.assertLess(body.index("window.__ragtime_session="), body.index("/main.js"))
        self.assertLess(body.index("bridge.js"), body.index("/main.js"))

    async def test_preview_handoff_cleanup_injects_before_app_scripts(self) -> None:
        html = b"""
        <html><head><script src="/main.js"></script></head><body>Dashboard</body></html>
        """
        upstream = _FakeProxyUpstreamResponse(html)
        request = _build_request(
            "/dashboard",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"accept", b"text/html")],
            method="GET",
            query_string="__ragtime_preview_handoff=nonce-1",
            body=b"",
        )

        with (
            mock.patch.object(_RUNTIME_ROUTES.httpx, "AsyncClient", return_value=_FakeProxyClient(upstream)),
            mock.patch.object(_RUNTIME_ROUTES, "get_app_settings", new=mock.AsyncMock(return_value={"userspace_preview_sandbox_flags": []})),
        ):
            response = await _RUNTIME_ROUTES._proxy_http_request(request, "http://runtime/dashboard")

        self.assertEqual(response.status_code, 200)
        body = bytes(response.body).decode("utf-8")
        self.assertIn("__ragtime_cleanup_preview_handoff", body)
        self.assertLess(body.index("__ragtime_cleanup_preview_handoff"), body.index("/main.js"))

    async def test_public_direct_preview_document_still_proxies_for_token_links(self) -> None:
        request = _build_request(
            "/",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"accept", b"text/html")],
            method="GET",
        )
        proxied = Response("proxied", media_type="text/html")

        with (
            mock.patch.object(
                _PREVIEW_HOST,
                "_verify_preview_session_cookie",
                new=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "preview_mode": "shared_public_host",
                    }
                ),
            ),
            mock.patch.object(_PREVIEW_HOST, "_build_upstream_url", new=mock.AsyncMock(return_value="http://runtime/")),
            mock.patch.object(_PREVIEW_HOST, "_proxy_http_request", new=mock.AsyncMock(return_value=proxied)) as proxy,
        ):
            response = await _PREVIEW_HOST.preview_proxy(request)

        self.assertIs(response, proxied)
        proxy.assert_awaited_once()

    def test_preview_bootstrap_handoff_query_is_internal(self) -> None:
        target = _PREVIEW_HOST._add_preview_handoff_to_target_path("/dashboard?tab=main", "nonce-1")

        self.assertEqual(target, "/dashboard?tab=main&__ragtime_preview_handoff=nonce-1")
        self.assertEqual(
            _RUNTIME_ROUTES._sanitize_preview_query("tab=main&__ragtime_preview_handoff=nonce-1&cap_token=old"),
            "tab=main",
        )

    async def test_top_level_preview_document_ignores_host_registry_without_cookie(self) -> None:
        request = _build_request(
            "/",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"accept", b"text/html"), (b"sec-fetch-dest", b"document")],
            method="GET",
        )

        with (
            mock.patch.object(
                _PREVIEW_HOST,
                "_lookup_preview_session",
                new=mock.AsyncMock(
                    return_value={
                        "workspace_id": "workspace-a",
                        "sub": "user-1",
                        "preview_mode": "workspace",
                    }
                ),
            ) as lookup,
            mock.patch.object(_PREVIEW_HOST, "_resolve_public_preview_session", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()) as invalidate,
        ):
            with self.assertRaises(HTTPException) as raised:
                await _PREVIEW_HOST._verify_preview_session_cookie(request)

        self.assertEqual(raised.exception.status_code, 401)
        lookup.assert_not_awaited()
        invalidate.assert_awaited_once_with("workspace-a.ragtime.test")

    async def test_top_level_public_document_clears_stale_registry_before_primitive_calls(self) -> None:
        request = _build_request(
            "/",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"accept", b"text/html"), (b"sec-fetch-dest", b"document")],
            method="GET",
        )
        public_claims = {
            "workspace_id": "workspace-a",
            "preview_mode": "shared_public_host",
            "share_access_mode": "token",
            "preview_host": "workspace-a.ragtime.test",
        }

        with (
            mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()) as invalidate,
            mock.patch.object(_PREVIEW_HOST, "_resolve_public_preview_session", new=mock.AsyncMock(return_value=public_claims)),
        ):
            claims = await _PREVIEW_HOST._verify_preview_session_cookie(request)

        self.assertEqual(claims, public_claims)
        invalidate.assert_awaited_once_with("workspace-a.ragtime.test")

    async def test_bootstrap_handoff_document_can_use_workspace_session_without_fetch_headers(self) -> None:
        request = _build_request(
            "/",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"accept", b"text/html")],
            method="GET",
            query_string="__ragtime_preview_handoff=nonce-1",
        )
        registry_claims = {
            "workspace_id": "workspace-a",
            "sub": "user-1",
            "preview_mode": "workspace",
            "preview_host": "workspace-a.ragtime.test",
        }

        with (
            mock.patch.object(_PREVIEW_HOST, "_consume_preview_handoff", new=mock.AsyncMock(return_value=registry_claims)) as consume,
            mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()) as invalidate,
            mock.patch.object(_PREVIEW_HOST, "_enforce_shared_subdomain_allowed", new=mock.AsyncMock()) as enforce,
        ):
            claims = await _PREVIEW_HOST._verify_preview_session_cookie(request)

        self.assertEqual(claims, registry_claims)
        consume.assert_awaited_once_with("workspace-a.ragtime.test", "nonce-1")
        enforce.assert_awaited_once_with(registry_claims)
        invalidate.assert_not_awaited()

    async def test_same_site_workspace_iframe_document_can_use_host_registry(self) -> None:
        request = _build_request(
            "/",
            headers=[
                (b"host", b"workspace-a.ragtime.test"),
                (b"accept", b"text/html"),
                (b"sec-fetch-dest", b"document"),
                (b"sec-fetch-site", b"same-site"),
            ],
            method="GET",
        )
        registry_claims = {
            "workspace_id": "workspace-a",
            "sub": "user-1",
            "preview_mode": "workspace",
            "preview_host": "workspace-a.ragtime.test",
        }

        with (
            mock.patch.object(_PREVIEW_HOST, "_lookup_preview_session", new=mock.AsyncMock(return_value=registry_claims)) as lookup,
            mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()) as invalidate,
            mock.patch.object(_PREVIEW_HOST, "_enforce_shared_subdomain_allowed", new=mock.AsyncMock()),
        ):
            claims = await _PREVIEW_HOST._verify_preview_session_cookie(request)

        self.assertEqual(claims, registry_claims)
        lookup.assert_awaited_once()
        invalidate.assert_not_awaited()

    async def test_headerless_preview_asset_request_can_use_host_registry(self) -> None:
        request = _build_request(
            "/dist/main.js",
            headers=[(b"host", b"workspace-a.ragtime.test")],
            method="GET",
        )
        registry_claims = {
            "workspace_id": "workspace-a",
            "sub": "user-1",
            "preview_mode": "workspace",
            "preview_host": "workspace-a.ragtime.test",
        }

        with (
            mock.patch.object(_PREVIEW_HOST, "_lookup_preview_session", new=mock.AsyncMock(return_value=registry_claims)) as lookup,
            mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()) as invalidate,
            mock.patch.object(_PREVIEW_HOST, "_enforce_shared_subdomain_allowed", new=mock.AsyncMock()),
        ):
            claims = await _PREVIEW_HOST._verify_preview_session_cookie(request)

        self.assertEqual(claims, registry_claims)
        lookup.assert_awaited_once()
        invalidate.assert_not_awaited()

    async def test_preview_primitive_request_can_use_host_registry_fallback(self) -> None:
        request = _build_request(
            "/__ragtime/session",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"accept", b"application/json")],
            method="GET",
        )
        registry_claims = {
            "workspace_id": "workspace-a",
            "sub": "user-1",
            "preview_mode": "workspace",
            "preview_host": "workspace-a.ragtime.test",
        }

        with (
            mock.patch.object(_PREVIEW_HOST, "_lookup_preview_session", new=mock.AsyncMock(return_value=registry_claims)) as lookup,
            mock.patch.object(_PREVIEW_HOST, "_enforce_shared_subdomain_allowed", new=mock.AsyncMock()),
        ):
            claims = await _PREVIEW_HOST._verify_preview_session_cookie(request)

        self.assertEqual(claims, registry_claims)
        lookup.assert_awaited_once()

    async def test_preview_browser_auth_logout_clears_preview_session_and_registry(self) -> None:
        response = Response()
        request = _build_request(
            "/__ragtime/browser-auth/logout",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"cookie", b"userspace_preview_session=session-token")],
        )

        with mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()) as invalidate:
            result = await _PREVIEW_HOST.preview_browser_auth_logout(request, response)

        self.assertEqual(result, {"success": True})
        set_cookie = "\n".join(value.decode("latin-1") for key, value in response.raw_headers if key == b"set-cookie")
        self.assertIn("userspace_preview_session=", set_cookie)
        self.assertIn("ragtime_session=", set_cookie)
        self.assertIn("Max-Age=0", set_cookie)
        self.assertIn("userspace_collab_capability=", set_cookie)
        invalidate.assert_awaited_once_with("workspace-a.ragtime.test")

    async def test_preview_workspace_browser_auth_logout_alias_clears_cookies(self) -> None:
        response = Response()
        request = _build_request(
            "/indexes/userspace/runtime/workspaces/workspace-a/browser-auth/logout",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"cookie", b"userspace_preview_session=session-token")],
        )

        with mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()) as invalidate:
            result = await _PREVIEW_HOST.preview_workspace_browser_auth_logout_alias("workspace-a", request, response)

        self.assertEqual(result, {"success": True})
        set_cookie = "\n".join(value.decode("latin-1") for key, value in response.raw_headers if key == b"set-cookie")
        self.assertIn("userspace_preview_session=", set_cookie)
        self.assertIn("ragtime_session=", set_cookie)
        self.assertIn("userspace_collab_capability=", set_cookie)
        invalidate.assert_awaited_once_with("workspace-a.ragtime.test")

    async def test_preview_browser_auth_logout_get_redirects_and_clears_cookies(self) -> None:
        request = _build_request(
            "/__ragtime/browser-auth/logout",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"cookie", b"userspace_preview_session=session-token")],
            method="GET",
            query_string="return_to=%2Fdashboard",
        )

        with mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()) as invalidate:
            response = await _PREVIEW_HOST.preview_browser_auth_logout_nav(request)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/dashboard")
        set_cookie = "\n".join(value.decode("latin-1") for key, value in response.raw_headers if key == b"set-cookie")
        self.assertIn("userspace_preview_session=", set_cookie)
        self.assertIn("ragtime_session=", set_cookie)
        self.assertIn("userspace_collab_capability=", set_cookie)
        invalidate.assert_awaited_once_with("workspace-a.ragtime.test")

    async def test_preview_workspace_browser_auth_logout_get_alias_redirects(self) -> None:
        request = _build_request(
            "/indexes/userspace/runtime/workspaces/workspace-a/browser-auth/logout",
            headers=[(b"host", b"workspace-a.ragtime.test"), (b"cookie", b"userspace_preview_session=session-token")],
            method="GET",
        )

        with mock.patch.object(_PREVIEW_HOST, "_invalidate_preview_session_for_host", new=mock.AsyncMock()):
            response = await _PREVIEW_HOST.preview_workspace_browser_auth_logout_nav_alias("workspace-a", request)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/")

    async def test_workspace_share_token_authorization_uses_share_access_mode(self) -> None:
        service = UserSpaceService()
        share_record = SimpleNamespace(
            id="share-1",
            workspaceId="workspace-a",
            shareToken="token-value",
            shareAccessMode="authenticated_users",
            ownerUserId="owner-1",
            shareSelectedUserIds=[],
            shareSelectedLdapGroups=[],
        )

        with mock.patch.object(service, "_find_workspace_share_by_token", new=mock.AsyncMock(return_value=share_record)):
            with self.assertRaises(HTTPException) as raised:
                await service.authorize_shared_workspace_access("token-value")

            result = await service.authorize_shared_workspace_access(
                "token-value",
                current_user=SimpleNamespace(id="user-1", role="user", authProvider="ldap"),
            )

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(str(raised.exception.detail), "Authentication required")
        self.assertEqual(result.get("workspace_id"), "workspace-a")

    async def test_service_runtime_audit_event_enriches_payload_before_create(self) -> None:
        created: list[dict[str, Any]] = []

        class _FakeAuditModel:
            async def create(self, *, data: dict[str, Any]) -> None:
                created.append(data)

        async def fake_get_db() -> SimpleNamespace:
            return SimpleNamespace(
                userspaceruntimeauditevent=_FakeAuditModel(),
                user=SimpleNamespace(
                    find_unique=mock.AsyncMock(
                        return_value=SimpleNamespace(id="user-1", username="user@example.com", authProvider="ldap", sourceProvider=None, sourceId=None)
                    )
                ),
            )

        service = UserSpaceService()
        with (
            mock.patch("ragtime.userspace.service.get_db", new=fake_get_db),
            mock.patch.object(service, "get_workspace_audit_fingerprint_namespace", new=mock.AsyncMock(return_value="portable-a")),
        ):
            ok = await service._record_runtime_audit_event(
                "workspace-a",
                "user-1",
                "app.action",
                {"action": "save"},
            )

        self.assertTrue(ok)
        event_payload = created[0]["eventPayload"].data
        self.assertEqual(event_payload["action"], "save")
        self.assertEqual(event_payload["user_fingerprint_scope"], "workspace")
        self.assertTrue(str(event_payload["user_fingerprint"]).startswith("v1:"))

    async def test_workspace_audit_identity_manifest_round_trips_namespace(self) -> None:
        service = UserSpaceService()

        with mock.patch.object(service, "get_workspace_audit_fingerprint_namespace", new=mock.AsyncMock(return_value="portable-a")):
            manifest = await service.export_workspace_audit_identity_manifest("workspace-a")

        self.assertEqual(manifest["workspace_fingerprint_namespace"], "portable-a")
        self.assertEqual(manifest["fingerprint_scope"], "workspace")

    async def test_workspace_audit_identity_persists_through_import_manifest(self) -> None:
        with TemporaryDirectory() as source_dir, TemporaryDirectory() as imported_dir:
            source_service = UserSpaceService()
            imported_service = UserSpaceService()
            source_files_dir = Path(source_dir) / "workspace" / "files"
            imported_files_dir = Path(imported_dir) / "workspace" / "files"

            with mock.patch.object(source_service, "_workspace_files_dir", return_value=source_files_dir):
                namespace = await source_service.get_workspace_audit_fingerprint_namespace("source-workspace")
                manifest = await source_service.export_workspace_audit_identity_manifest("source-workspace")

            with mock.patch.object(imported_service, "_workspace_files_dir", return_value=imported_files_dir):
                await imported_service.import_workspace_audit_identity_manifest("imported-workspace", manifest)
                imported_namespace = await imported_service.get_workspace_audit_fingerprint_namespace("imported-workspace")

            self.assertEqual(imported_namespace, namespace)


if __name__ == "__main__":
    unittest.main()
