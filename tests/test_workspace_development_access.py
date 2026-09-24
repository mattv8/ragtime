import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request

from ragtime.userspace import development_access


def _request(token: str) -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"authorization", f"Bearer {token}".encode())]})


class WorkspaceDevelopmentAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_development_credential_resolves_fresh_user_and_scopes(self) -> None:
        token, selector, token_hash = development_access.build_development_credential_token()
        credential = SimpleNamespace(
            id="credential-1",
            workspaceId="workspace-1",
            userId="user-1",
            tokenHash=token_hash,
            revokedAt=None,
            expiresAt=None,
            scopes=["read", "exec"],
        )
        db = SimpleNamespace(
            workspacedevelopmentcredential=SimpleNamespace(find_unique=mock.AsyncMock(return_value=credential)),
            user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="admin"))),
        )
        with mock.patch.object(development_access, "get_db", mock.AsyncMock(return_value=db)):
            principal = await development_access.resolve_development_principal(_request(token))
        db.workspacedevelopmentcredential.find_unique.assert_awaited_once_with(where={"selector": selector})
        self.assertEqual(principal.workspace_id, "workspace-1")
        self.assertEqual(principal.scopes, frozenset({"read", "exec"}))
        self.assertTrue(principal.is_admin)

    async def test_revoked_development_credential_is_rejected(self) -> None:
        token, selector, token_hash = development_access.build_development_credential_token()
        credential = SimpleNamespace(
            id="credential-1",
            workspaceId="workspace-1",
            userId="user-1",
            tokenHash=token_hash,
            revokedAt=object(),
            expiresAt=None,
            scopes=["read"],
        )
        db = SimpleNamespace(workspacedevelopmentcredential=SimpleNamespace(find_unique=mock.AsyncMock(return_value=credential)))
        with mock.patch.object(development_access, "get_db", mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as raised:
                await development_access.resolve_development_principal(_request(token))
        self.assertEqual(raised.exception.status_code, 401)

    async def test_malformed_credential_expiry_is_rejected(self) -> None:
        token, selector, token_hash = development_access.build_development_credential_token()
        credential = SimpleNamespace(
            id="credential-1",
            workspaceId="workspace-1",
            userId="user-1",
            tokenHash=token_hash,
            revokedAt=None,
            expiresAt="not-a-datetime",
            scopes=["read"],
        )
        db = SimpleNamespace(workspacedevelopmentcredential=SimpleNamespace(find_unique=mock.AsyncMock(return_value=credential)))
        with mock.patch.object(development_access, "get_db", mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as raised:
                await development_access.resolve_development_principal(_request(token))
        self.assertEqual(raised.exception.status_code, 401)

    async def test_session_path_uses_existing_session_parser(self) -> None:
        with mock.patch.object(development_access, "get_current_user", mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user"))) as current_user:
            principal = await development_access.resolve_development_principal(_request("session-jwt"))
        current_user.assert_awaited_once_with(token="session-jwt")
        self.assertIsNone(principal.credential_id)
