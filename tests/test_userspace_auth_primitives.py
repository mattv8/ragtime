from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

import ragtime.userspace.preview_host as _PREVIEW_HOST
import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES
from ragtime.core.user_identity import add_workspace_user_fingerprint, build_user_fingerprint_subject, build_workspace_user_fingerprint
from ragtime.userspace.models import UserSpaceBrowserAuthRequest
from ragtime.userspace.service import UserSpaceService


def _build_request(path: str = "/__ragtime/browser-auth") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": [(b"host", b"workspace.ragtime.test")],
            "scheme": "https",
            "server": ("workspace.ragtime.test", 443),
            "client": ("127.0.0.1", 12345),
        }
    )


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
        self.assertEqual(payload["auth"]["binding_strategy"], "browser_capability_token")
        self.assertEqual(payload["auth"]["methods"][0]["key"], "local_managed")
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
        )

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
