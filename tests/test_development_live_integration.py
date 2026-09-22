"""Opt-in local-DB coverage for the external workspace development surface."""

import hashlib
import json
import os
import shutil
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import TypedDict
from unittest import mock
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from mcp.types import TextContent
from prisma.enums import AuthProvider, UserRole, WorkspaceRole
from starlette.requests import Request

from ragtime.api import routes as api_routes
from ragtime.core.database import connect_db, disconnect_db
from ragtime.core.datetimes import utc_now
from ragtime.mcp import server as mcp_server
from ragtime.rag.components import _CopilotChatOpenAI
from ragtime.userspace import development_credentials_routes, development_routes
from ragtime.userspace.development_access import (
    DevelopmentPrincipal,
    create_workspace_development_credential,
    resolve_development_principal,
    revoke_workspace_development_credential,
    rotate_workspace_development_credential,
)
from ragtime.userspace.development_service import development_service
from ragtime.userspace.service import userspace_service


def _request(token: str) -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"authorization", f"Bearer {token}".encode())]})


class _CredentialResponse(TypedDict):
    id: str
    token: str
    scopes: list[str]


@unittest.skipUnless(os.environ.get("RAGTIME_BYO_INTEGRATION") == "1", "requires local Prisma/Postgres opt-in")
class DevelopmentLiveIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db = await connect_db()
        self.owner_id, self.other_id = str(uuid4()), str(uuid4())
        self.workspace_id, self.other_workspace_id = str(uuid4()), str(uuid4())
        for user_id, label in ((self.owner_id, "owner"), (self.other_id, "other")):
            await self.db.user.create(
                data={
                    "id": user_id,
                    "username": f"local:byo-live-{label}-{user_id}",
                    "authProvider": AuthProvider.local,
                    "chatEnabled": False,
                    "userspaceGenerationEnabled": False,
                }
            )
        for workspace_id, owner_id, name in (
            (self.workspace_id, self.owner_id, "byo-live-primary"),
            (self.other_workspace_id, self.other_id, "byo-live-other"),
        ):
            await self.db.workspace.create(data={"id": workspace_id, "name": f"{name}-{workspace_id}", "ownerUserId": owner_id})
            await self.db.workspacemember.create(data={"workspaceId": workspace_id, "userId": owner_id, "role": WorkspaceRole.owner})
            userspace_service._workspace_files_dir(workspace_id).mkdir(parents=True, exist_ok=True)
            await userspace_service._ensure_workspace_git_repo(workspace_id)

    async def asyncTearDown(self) -> None:
        for workspace_id in (self.workspace_id, self.other_workspace_id):
            await self.db.workspace.delete_many(where={"id": workspace_id})
            shutil.rmtree(userspace_service._workspace_dir(workspace_id), ignore_errors=True)
        await self.db.user.delete_many(where={"id": {"in": [self.owner_id, self.other_id]}})
        await disconnect_db()

    async def _credential(self, *, scopes: list[str] | None = None, expires_at: datetime | None = None) -> tuple[_CredentialResponse, DevelopmentPrincipal]:
        created = await create_workspace_development_credential(
            workspace_id=self.workspace_id,
            user_id=self.owner_id,
            name="live integration",
            scopes=scopes or ["read", "write"],
            expires_at=expires_at,
        )
        credential_id = created.get("id")
        token = created.get("token")
        credential_scopes = created.get("scopes")
        assert isinstance(credential_id, str)
        assert isinstance(token, str)
        assert isinstance(credential_scopes, list) and all(isinstance(scope, str) for scope in credential_scopes)
        response: _CredentialResponse = {"id": credential_id, "token": token, "scopes": credential_scopes}
        return response, await resolve_development_principal(_request(response["token"]))

    async def _delete_credential_record(self, workspace_id: str, credential_id: str) -> httpx.Response:
        app = FastAPI()
        app.include_router(development_credentials_routes.router)
        app.dependency_overrides[development_credentials_routes.get_current_user] = lambda: SimpleNamespace(id=self.owner_id, role="user")
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                return await client.delete(
                    f"/indexes/userspace/development/workspaces/{workspace_id}/credentials/{credential_id}/record"
                )
        finally:
            app.dependency_overrides.clear()

    async def test_credential_http_operations_snapshot_context_and_mcp_share_the_real_dispatcher(self) -> None:
        created, principal = await self._credential()
        self.assertTrue(created["token"].startswith("rtdev_"))

        operations = await development_routes.get_operations(self.workspace_id, principal)
        self.assertIn("file_write", {item["name"] for item in operations["operations"]})
        written = await development_routes.execute_operation(
            self.workspace_id,
            "file_write",
            development_routes.DevelopmentOperationRequest(arguments={"path": "src/app.txt", "content": "first", "expected_hash": None}),
            principal,
        )
        read = await development_service.execute(principal, self.workspace_id, "file_read", {"path": "src/app.txt"})
        self.assertEqual(read["content"], "first")
        with self.assertRaises(HTTPException) as conflict:
            await development_service.execute(
                principal, self.workspace_id, "file_patch", {"path": "src/app.txt", "expected_hash": "wrong", "old": "first", "new": "second"}
            )
        self.assertEqual(conflict.exception.status_code, 409)
        patched = await development_service.execute(
            principal, self.workspace_id, "file_patch", {"path": "src/app.txt", "expected_hash": written["content_hash"], "old": "first", "new": "second"}
        )
        snapshot = await development_service.execute(principal, self.workspace_id, "snapshot_create", {"message": "live test"})
        snapshots = await development_service.execute(principal, self.workspace_id, "snapshots_list", {})
        self.assertIn(snapshot["id"], {item["id"] for item in snapshots["snapshots"]})

        with mock.patch.object(_CopilotChatOpenAI, "_agenerate", new=mock.AsyncMock()) as generate:
            context = await development_routes.get_context(self.workspace_id, principal)
        self.assertEqual(context["user"]["username"], f"local:byo-live-owner-{self.owner_id}")
        self.assertTrue(context["system_instructions"])
        generate.assert_not_awaited()

        with mcp_server.development_principal_context(principal):
            result = await mcp_server._execute_development_tool(
                "workspace_development", {"workspace_id": self.workspace_id, "operation": "file_read", "arguments": {"path": "src/app.txt"}}
            )
        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text)["content"], "second")

        deleted = await development_service.execute(
            principal, self.workspace_id, "file_delete", {"path": "src/app.txt", "expected_hash": patched["content_hash"]}
        )
        self.assertTrue(deleted["deleted"])

    async def test_credentials_are_bound_scoped_rotatable_and_invalid_after_revoke_or_expiry(self) -> None:
        created, principal = await self._credential(scopes=["read"])
        self.assertEqual(created["scopes"], ["read"])
        listed = await development_credentials_routes.list_workspace_development_credentials(self.workspace_id, SimpleNamespace(id=self.owner_id, role="user"))
        self.assertNotIn("token", listed["items"][0])
        with self.assertRaises(HTTPException) as rejected_scope:
            await development_service.execute(principal, self.workspace_id, "file_write", {"path": "x", "content": "x", "expected_hash": None})
        self.assertEqual(rejected_scope.exception.status_code, 403)
        with self.assertRaises(HTTPException) as rejected_workspace:
            await development_service.execute(principal, self.other_workspace_id, "files_list", {})
        self.assertEqual(rejected_workspace.exception.status_code, 403)

        await self.db.user.update(where={"id": self.owner_id}, data={"role": UserRole.admin})
        self.assertTrue((await resolve_development_principal(_request(created["token"]))).is_admin)

        rotated = await rotate_workspace_development_credential(workspace_id=self.workspace_id, credential_id=created["id"])
        with self.assertRaises(HTTPException):
            await resolve_development_principal(_request(created["token"]))
        self.assertEqual((await resolve_development_principal(_request(rotated["token"]))).credential_id, created["id"])
        await revoke_workspace_development_credential(workspace_id=self.workspace_id, credential_id=created["id"])
        with self.assertRaises(HTTPException):
            await resolve_development_principal(_request(rotated["token"]))

        expired, _ = await self._credential(expires_at=utc_now() + timedelta(minutes=1))
        await self.db.workspacedevelopmentcredential.update(where={"id": expired["id"]}, data={"expiresAt": utc_now() - timedelta(seconds=1)})
        with self.assertRaises(HTTPException) as rejected_expiry:
            await resolve_development_principal(_request(expired["token"]))
        self.assertEqual(rejected_expiry.exception.status_code, 401)

    async def test_deleting_a_revoked_credential_removes_it_from_the_workspace_list(self) -> None:
        created, _ = await self._credential()
        await revoke_workspace_development_credential(workspace_id=self.workspace_id, credential_id=created["id"])

        response = await self._delete_credential_record(self.workspace_id, created["id"])
        self.assertEqual(response.status_code, 204)
        listed = await development_credentials_routes.list_workspace_development_credentials(
            self.workspace_id, SimpleNamespace(id=self.owner_id, role="user")
        )
        self.assertNotIn(created["id"], {item["id"] for item in listed["items"]})

    async def test_deleting_an_active_credential_is_rejected(self) -> None:
        created, _ = await self._credential()

        response = await self._delete_credential_record(self.workspace_id, created["id"])
        self.assertEqual(response.status_code, 400)

    async def test_deleting_a_credential_from_another_workspace_returns_not_found(self) -> None:
        created, _ = await self._credential()
        await revoke_workspace_development_credential(workspace_id=self.workspace_id, credential_id=created["id"])

        response = await self._delete_credential_record(self.other_workspace_id, created["id"])
        self.assertEqual(response.status_code, 404)

    async def test_development_bearer_is_rejected_by_anonymous_v1_auth_dependency(self) -> None:
        created, _ = await self._credential()
        with mock.patch.object(api_routes.settings, "api_key", None):
            with self.assertRaises(HTTPException) as rejected:
                await api_routes.verify_api_key(authorization=f"Bearer {created['token']}")
        self.assertEqual(rejected.exception.status_code, 401)

    @unittest.skipUnless(os.environ.get("RAGTIME_BYO_BASE_URL"), "requires an isolated current-source HTTP server")
    async def test_bootstrap_downloads_are_complete_authorized_and_revocable(self) -> None:
        created, _ = await self._credential(scopes=["read"])
        base_url = os.environ["RAGTIME_BYO_BASE_URL"].rstrip("/")
        prefix = f"/indexes/userspace/development/workspaces/{self.workspace_id}"
        async with httpx.AsyncClient(base_url=base_url, headers={"Authorization": f"Bearer {created['token']}"}, timeout=30) as client:
            response = await client.get(f"{prefix}/bootstrap")
            self.assertEqual(response.status_code, 200, response.text)
            manifest = response.json()
            self.assertEqual(manifest["scopes"], ["read"])
            artifacts = {item["id"]: item for item in manifest["artifacts"]}
            profile = manifest["profiles"]["opencode"]
            downloaded = {}
            for artifact_id in profile["artifact_ids"]:
                with self.subTest(artifact=artifact_id):
                    artifact = artifacts[artifact_id]
                    result = await client.get(artifact["download_url"])
                    self.assertEqual(result.status_code, 200, result.text)
                    self.assertEqual(len(result.content), artifact["bytes"])
                    self.assertEqual(hashlib.sha256(result.content).hexdigest(), artifact["sha256"])
                    self.assertNotIn(created["token"], result.text)
                    self.assertIn(artifact["content_type"].split(";")[0], result.headers["content-type"])
                    downloaded[artifact_id] = result.text

            config = json.loads(downloaded[profile["config_artifact"]])
            self.assertIn("mcp", config)
            self.assertTrue(config["instructions"])
            self.assertIn(manifest["credential_env_var"], downloaded[profile["config_artifact"]])
            stale = await client.get(f"{prefix}/bootstrap/files/core/rules.md", params={"revision": "stale"})
            self.assertEqual(stale.status_code, 409)
            unknown = await client.get(f"{prefix}/bootstrap/files/not-an-artifact")
            self.assertEqual(unknown.status_code, 404)
            other = await client.get(f"/indexes/userspace/development/workspaces/{self.other_workspace_id}/bootstrap")
            self.assertEqual(other.status_code, 403)
            await revoke_workspace_development_credential(workspace_id=self.workspace_id, credential_id=created["id"])
            revoked = await client.get(f"{prefix}/bootstrap")
            self.assertEqual(revoked.status_code, 401)
