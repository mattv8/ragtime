"""Opt-in smoke coverage for real external-development transport boundaries."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import unittest
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import httpx
from fastapi import FastAPI
from prisma.enums import AuthProvider, WorkspaceRole

from ragtime.core.database import connect_db, disconnect_db
from ragtime.mcp import routes as mcp_routes
from ragtime.mcp import server as mcp_server
from ragtime.rag.components import _CopilotChatOpenAI
from ragtime.userspace.development_access import (
    create_workspace_development_credential,
    revoke_workspace_development_credential,
)
from ragtime.userspace.service import userspace_service


def _mcp_request(method: str, params: dict, request_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


@unittest.skipUnless(os.environ.get("RAGTIME_BYO_INTEGRATION") == "1", "requires local Prisma/Postgres opt-in")
class DevelopmentTransportRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Uses the isolated manager configured by the opt-in verification harness."""

    async def asyncSetUp(self) -> None:
        self.db = await connect_db()
        self.owner_id, self.other_id = str(uuid4()), str(uuid4())
        self.workspace_id, self.other_workspace_id = str(uuid4()), str(uuid4())
        self.workspace_ids = (self.workspace_id, self.other_workspace_id)
        for user_id, label in ((self.owner_id, "owner"), (self.other_id, "other")):
            await self.db.user.create(
                data={
                    "id": user_id,
                    "username": f"local:byo-transport-{label}-{user_id}",
                    "authProvider": AuthProvider.local,
                    "hostedChatEnabled": False,
                }
            )
        for workspace_id, owner_id, label in (
            (self.workspace_id, self.owner_id, "primary"),
            (self.other_workspace_id, self.other_id, "other"),
        ):
            await self.db.workspace.create(data={"id": workspace_id, "name": f"byo-transport-{label}-{workspace_id}", "ownerUserId": owner_id})
            await self.db.workspacemember.create(data={"workspaceId": workspace_id, "userId": owner_id, "role": WorkspaceRole.owner})
            userspace_service._workspace_files_dir(workspace_id).mkdir(parents=True, exist_ok=True)
            userspace_service._seed_runtime_entrypoint_config(workspace_id)
            await userspace_service._ensure_workspace_git_repo(workspace_id)

    async def asyncTearDown(self) -> None:
        for workspace_id in self.workspace_ids:
            await self.db.workspace.delete_many(where={"id": workspace_id})
            shutil.rmtree(userspace_service._workspace_dir(workspace_id), ignore_errors=True)
        await self.db.user.delete_many(where={"id": {"in": [self.owner_id, self.other_id]}})
        await disconnect_db()

    async def _credential(self, workspace_id: str, user_id: str, scopes: list[str]) -> dict:
        return await create_workspace_development_credential(
            workspace_id=workspace_id,
            user_id=user_id,
            name="transport smoke",
            scopes=scopes,
            expires_at=None,
        )

    async def _mcp_json(self, client: httpx.AsyncClient, token: str, body: dict) -> dict:
        response = await client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"},
            json=body,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_mcp_http_transport_keeps_rtdev_principals_request_local(self) -> None:
        read_credential = await self._credential(self.workspace_id, self.owner_id, ["read"])
        exec_credential = await self._credential(self.other_workspace_id, self.other_id, ["read", "exec"])
        app = FastAPI()
        for route in mcp_routes.get_mcp_routes():
            app.router.routes.append(route)
        settings = {
            "mcp_enabled": True,
            "mcp_default_route_auth": True,
            "mcp_default_route_auth_method": "oauth2",
            "server_name": "byo-transport-smoke",
        }
        # This is test-only transport configuration: rtdev validation, routing,
        # principal binding, and operation dispatch remain the real code paths.
        previous_manager = mcp_routes._mcp_http_state["session_manager"]
        mcp_routes._mcp_http_state["session_manager"] = None
        try:
            async with AsyncExitStack() as stack:
                stack.enter_context(mock.patch.object(mcp_routes, "get_app_settings", new=mock.AsyncMock(return_value=settings)))
                stack.enter_context(mock.patch.object(mcp_server, "get_app_settings", new=mock.AsyncMock(return_value=settings)))
                await stack.enter_async_context(mcp_routes.mcp_lifespan_manager())
                client = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp.test"))

                initialization = {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "1"}}
                await self._mcp_json(client, read_credential["token"], _mcp_request("initialize", initialization, 1))
                await self._mcp_json(client, exec_credential["token"], _mcp_request("initialize", initialization, 2))
                read_tools, exec_tools = await asyncio.gather(
                    self._mcp_json(client, read_credential["token"], _mcp_request("tools/list", {}, 3)),
                    self._mcp_json(client, exec_credential["token"], _mcp_request("tools/list", {}, 4)),
                )
                self.assertEqual({tool["name"] for tool in read_tools["result"]["tools"]}, {"workspace_development", "workspace_development_context"})
                self.assertEqual({tool["name"] for tool in exec_tools["result"]["tools"]}, {"workspace_development", "workspace_development_context"})
                read_ops = next(tool for tool in read_tools["result"]["tools"] if tool["name"] == "workspace_development")["inputSchema"]["properties"][
                    "operation"
                ]["enum"]
                exec_ops = next(tool for tool in exec_tools["result"]["tools"] if tool["name"] == "workspace_development")["inputSchema"]["properties"][
                    "operation"
                ]["enum"]
                self.assertNotIn("exec_start", read_ops)
                self.assertIn("exec_start", exec_ops)

                call = _mcp_request("tools/call", {"name": "workspace_development_context", "arguments": {"workspace_id": self.workspace_id}}, 5)
                with mock.patch.object(_CopilotChatOpenAI, "_agenerate", new=mock.AsyncMock()) as generate:
                    context_result = await self._mcp_json(client, read_credential["token"], call)
                self.assertFalse(context_result["result"].get("isError", False))
                self.assertEqual(json.loads(context_result["result"]["content"][0]["text"])["user"]["username"], f"local:byo-transport-owner-{self.owner_id}")
                generate.assert_not_awaited()

                await revoke_workspace_development_credential(workspace_id=self.workspace_id, credential_id=read_credential["id"])
                revoked = await client.post("/mcp", headers={"Authorization": f"Bearer {read_credential['token']}"}, json=_mcp_request("tools/list", {}, 6))
                self.assertEqual(revoked.status_code, 401)
        finally:
            mcp_routes._mcp_http_state["session_manager"] = previous_manager

    async def test_runtime_manager_exec_job_lifecycle_preserves_workspace_file(self) -> None:
        manager_url = os.environ.get("RUNTIME_MANAGER_URL")
        manager_token = os.environ.get("RUNTIME_AUTH_TOKEN")
        if not manager_url or not manager_token:
            self.skipTest("requires isolated runtime manager URL and token")
        provider_session_id = f"byo-runtime-{self.workspace_id[:12]}"
        headers = {"Authorization": f"Bearer {manager_token}"}
        async with httpx.AsyncClient(base_url=manager_url, headers=headers, timeout=30) as client:
            started = await client.post(
                "/sessions/start", json={"workspace_id": self.workspace_id, "leased_by_user_id": self.owner_id, "provider_session_id": provider_session_id}
            )
            self.assertEqual(started.status_code, 200, started.text)
            try:
                await self._wait_for_runtime_startup(client, provider_session_id)
                job = await client.post(
                    f"/sessions/{provider_session_id}/exec-jobs",
                    json={
                        "command": "printf runtime-smoke > .ragtime/exec-smoke.txt; cat .ragtime/exec-smoke.txt",
                        "timeout_seconds": 20,
                        "user_id": self.owner_id,
                        "operation": "transport_smoke",
                    },
                )
                self.assertEqual(job.status_code, 200, job.text)
                completed = await self._terminal_job(client, provider_session_id, job.json()["id"])
                self.assertEqual(completed["status"], "completed")
                self.assertIn("runtime-smoke", completed["output"])
                first = await client.get(f"/sessions/{provider_session_id}/exec-jobs/{job.json()['id']}", params={"cursor": 0, "limit": 7})
                self.assertEqual(first.status_code, 200, first.text)
                self.assertLessEqual(len(first.json()["output"]), 7)
                listed = await client.get(f"/sessions/{provider_session_id}/exec-jobs")
                self.assertIn(job.json()["id"], {item["id"] for item in listed.json()})

                long_job = await client.post(f"/sessions/{provider_session_id}/exec-jobs", json={"command": "sleep 20", "timeout_seconds": 25})
                self.assertEqual(long_job.status_code, 200, long_job.text)
                cancelled = await client.post(f"/sessions/{provider_session_id}/exec-jobs/{long_job.json()['id']}/cancel")
                self.assertEqual(cancelled.status_code, 200, cancelled.text)
                self.assertEqual((await self._terminal_job(client, provider_session_id, long_job.json()["id"]))["status"], "cancelled")
            finally:
                await client.post(f"/sessions/{provider_session_id}/stop")

            restarted = await client.post(
                "/sessions/start", json={"workspace_id": self.workspace_id, "leased_by_user_id": self.owner_id, "provider_session_id": provider_session_id}
            )
            self.assertEqual(restarted.status_code, 200, restarted.text)
            try:
                await self._wait_for_runtime_startup(client, provider_session_id)
                preserved = await client.post(
                    f"/sessions/{provider_session_id}/exec-jobs", json={"command": "cat .ragtime/exec-smoke.txt", "timeout_seconds": 20}
                )
                self.assertEqual(preserved.status_code, 200, preserved.text)
                self.assertIn("runtime-smoke", (await self._terminal_job(client, provider_session_id, preserved.json()["id"]))["output"])
            finally:
                await client.post(f"/sessions/{provider_session_id}/stop")

    async def _wait_for_runtime_startup(self, client: httpx.AsyncClient, provider_session_id: str) -> dict:
        async def wait_for_startup() -> dict:
            while True:
                response = await client.get(f"/sessions/{provider_session_id}")
                self.assertEqual(response.status_code, 200, response.text)
                session = response.json()
                if session["state"] == "running" and session.get("runtime_operation_phase") not in {
                    "queued",
                    "provisioning",
                    "bootstrapping",
                    "deps_install",
                    "launching",
                    "probing",
                }:
                    return session
                await asyncio.sleep(0.2)

        return await asyncio.wait_for(wait_for_startup(), timeout=120)

    async def _terminal_job(self, client: httpx.AsyncClient, provider_session_id: str, job_id: str) -> dict:
        async def wait_for_terminal() -> dict:
            while True:
                response = await client.get(f"/sessions/{provider_session_id}/exec-jobs/{job_id}", params={"cursor": 0, "limit": 16384})
                self.assertEqual(response.status_code, 200, response.text)
                job = response.json()
                if job["status"] != "running":
                    return job
                await asyncio.sleep(0.2)

        return await asyncio.wait_for(wait_for_terminal(), timeout=30)
