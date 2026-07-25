from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest import mock

from fastapi import HTTPException

from ragtime.core.encryption import decrypt_with_password
from ragtime.indexer import repository as repository_module
from ragtime.indexer import routes
from ragtime.indexer.models import ToolAccessEntryUpdate, ToolAccessPolicyResponse, ToolAccessPolicyUpdateRequest, ToolConfig, ToolType

EXPORT_PASSWORD = "correct horse battery staple!9A"


class _FakeToolConfigDelegate:
    def __init__(self, *, exists: bool = True, allow_write: bool = True) -> None:
        self.exists = exists
        self.allow_write = allow_write

    async def find_unique(self, *, where: dict[str, Any], **_kwargs: Any) -> Any:
        if not self.exists or where.get("id") != "tool-1":
            return None
        return SimpleNamespace(id="tool-1", allowWrite=self.allow_write)


class _FakeLookupDelegate:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    async def find_many(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        requested_ids = set(kwargs.get("where", {}).get("id", {}).get("in", []))
        return [row for row in self.rows if getattr(row, "id", None) in requested_ids]


class _FakeCreateManyDelegate:
    def __init__(self, *, raise_exc: Exception | None = None) -> None:
        self.raise_exc = raise_exc
        self.delete_many_calls: list[dict[str, Any]] = []
        self.create_many_calls: list[dict[str, Any]] = []

    async def delete_many(self, **kwargs: Any) -> None:
        self.delete_many_calls.append(kwargs)

    async def create_many(self, **kwargs: Any) -> None:
        self.create_many_calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc


class _FakeToolAccessPolicyDelegate:
    def __init__(self, *, default_chat_access: str = "deny", default_workspace_access: str = "deny") -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self.default_chat_access = default_chat_access
        self.default_workspace_access = default_workspace_access

    async def find_unique(self, *, where: dict[str, Any], **_kwargs: Any) -> Any:
        if where.get("toolConfigId") != "tool-1":
            return None
        return SimpleNamespace(
            id="policy-1",
            toolConfigId="tool-1",
            defaultChatAccess=self.default_chat_access,
            defaultWorkspaceAccess=self.default_workspace_access,
        )

    async def upsert(self, **kwargs: Any) -> Any:
        self.upsert_calls.append(kwargs)
        return SimpleNamespace(id="policy-1", toolConfigId="tool-1")


class _FakeTx:
    def __init__(self, *, create_many_exc: Exception | None = None) -> None:
        self.toolaccesspolicy = _FakeToolAccessPolicyDelegate()
        self.tooluseraccess = _FakeCreateManyDelegate(raise_exc=create_many_exc)
        self.toolauthgroupaccess = _FakeCreateManyDelegate()


class _FakeTxContext:
    def __init__(self, tx: _FakeTx) -> None:
        self.tx = tx

    async def __aenter__(self) -> _FakeTx:
        return self.tx

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class _FakeDb:
    def __init__(
        self,
        *,
        tool_exists: bool = True,
        tool_allow_write: bool = True,
        user_rows: list[dict[str, Any]] | None = None,
        group_rows: list[dict[str, Any]] | None = None,
        existing_users: list[Any] | None = None,
        existing_groups: list[Any] | None = None,
        create_many_exc: Exception | None = None,
    ) -> None:
        self.toolconfig = _FakeToolConfigDelegate(exists=tool_exists, allow_write=tool_allow_write)
        self.toolaccesspolicy = _FakeToolAccessPolicyDelegate()
        self.user = _FakeLookupDelegate(existing_users or [])
        self.authgroup = _FakeLookupDelegate(existing_groups or [])
        self._user_rows = list(user_rows or [])
        self._group_rows = list(group_rows or [])
        self.tx_state = _FakeTx(create_many_exc=create_many_exc)

    async def query_raw(self, query: str, *_params: Any) -> list[dict[str, Any]]:
        normalized = " ".join(query.split())
        if 'FROM "tool_user_access"' in normalized:
            return list(self._user_rows)
        if 'FROM "tool_auth_group_access"' in normalized:
            return list(self._group_rows)
        raise AssertionError(f"Unexpected query: {query}")

    def tx(self) -> _FakeTxContext:
        return _FakeTxContext(self.tx_state)


class ToolAccessRepositoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.repo = repository_module.IndexerRepository()

    async def test_get_tool_access_policy_enriches_users_and_groups(self) -> None:
        fake_db = _FakeDb(
            user_rows=[
                {
                    "principal_id": "user-1",
                    "chat_access": "read",
                    "workspace_access": "read_write",
                    "display_name": "Alex Admin",
                    "username": "local:alex",
                    "auth_provider": "local_managed",
                }
            ],
            group_rows=[
                {
                    "principal_id": "group-1",
                    "chat_access": None,
                    "workspace_access": "read",
                    "display_name": "Ops",
                    "provider": "ldap",
                    "active_member_count": 0,
                }
            ],
        )

        with mock.patch.object(self.repo, "_get_db", mock.AsyncMock(return_value=fake_db)):
            result = await self.repo.get_tool_access_policy("tool-1")

        self.assertEqual(result.tool_id, "tool-1")
        self.assertEqual(result.default_chat_access, "deny")
        self.assertEqual(result.default_workspace_access, "deny")
        self.assertEqual(result.users[0].display_name, "Alex Admin")
        self.assertEqual(result.users[0].principal_detail, "local:alex")
        self.assertFalse(result.users[0].orphaned)
        self.assertEqual(result.groups[0].display_name, "Ops")
        self.assertEqual(result.groups[0].principal_detail, "ldap")
        self.assertTrue(result.groups[0].orphaned)

    async def test_get_tool_access_policy_caps_legacy_read_write_defaults_for_read_only_tool(self) -> None:
        fake_db = _FakeDb(tool_allow_write=False)
        fake_db.toolaccesspolicy.default_chat_access = "read_write"
        fake_db.toolaccesspolicy.default_workspace_access = "read_write"

        with mock.patch.object(self.repo, "_get_db", mock.AsyncMock(return_value=fake_db)):
            result = await self.repo.get_tool_access_policy("tool-1")

        self.assertEqual(result.default_chat_access, "read")
        self.assertEqual(result.default_workspace_access, "read")

    async def test_replace_tool_access_policy_rejects_unknown_principals(self) -> None:
        request = ToolAccessPolicyUpdateRequest(
            default_chat_access="read",
            default_workspace_access="deny",
            users=[ToolAccessEntryUpdate(principal_id="missing-user", chat_access="read", workspace_access=None)],
            groups=[ToolAccessEntryUpdate(principal_id="missing-group", chat_access=None, workspace_access="read")],
        )
        fake_db = _FakeDb(existing_users=[], existing_groups=[])

        with mock.patch.object(self.repo, "_get_db", mock.AsyncMock(return_value=fake_db)):
            with self.assertRaises(HTTPException) as ctx:
                await self.repo.replace_tool_access_policy("tool-1", request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("missing-user", str(ctx.exception.detail))
        self.assertIn("missing-group", str(ctx.exception.detail))

    async def test_replace_tool_access_policy_rejects_null_null_entries(self) -> None:
        request = ToolAccessPolicyUpdateRequest(
            default_chat_access="read",
            default_workspace_access="deny",
            users=[ToolAccessEntryUpdate(principal_id="user-1", chat_access=None, workspace_access=None)],
            groups=[],
        )
        fake_db = _FakeDb(existing_users=[SimpleNamespace(id="user-1")])

        with mock.patch.object(self.repo, "_get_db", mock.AsyncMock(return_value=fake_db)):
            with self.assertRaises(HTTPException) as ctx:
                await self.repo.replace_tool_access_policy("tool-1", request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("at least one surface", str(ctx.exception.detail))

    async def test_replace_tool_access_policy_rejects_duplicate_principals(self) -> None:
        request = ToolAccessPolicyUpdateRequest(
            default_chat_access="read",
            default_workspace_access="deny",
            users=[
                ToolAccessEntryUpdate(principal_id="user-1", chat_access="read", workspace_access=None),
                ToolAccessEntryUpdate(principal_id="user-1", chat_access=None, workspace_access="read"),
            ],
            groups=[],
        )
        fake_db = _FakeDb(existing_users=[SimpleNamespace(id="user-1")])

        with mock.patch.object(self.repo, "_get_db", mock.AsyncMock(return_value=fake_db)):
            with self.assertRaises(HTTPException) as ctx:
                await self.repo.replace_tool_access_policy("tool-1", request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Duplicate user principals", str(ctx.exception.detail))

    async def test_replace_tool_access_policy_replaces_rows_transactionally(self) -> None:
        request = ToolAccessPolicyUpdateRequest(
            default_chat_access="read",
            default_workspace_access="read_write",
            users=[ToolAccessEntryUpdate(principal_id="user-1", chat_access="read", workspace_access=None)],
            groups=[ToolAccessEntryUpdate(principal_id="group-1", chat_access=None, workspace_access="read_write")],
        )
        fake_db = _FakeDb(
            tool_allow_write=True,
            existing_users=[SimpleNamespace(id="user-1")],
            existing_groups=[SimpleNamespace(id="group-1")],
        )

        with mock.patch.object(self.repo, "_get_db", mock.AsyncMock(return_value=fake_db)):
            result = await self.repo.replace_tool_access_policy("tool-1", request)

        upsert = fake_db.tx_state.toolaccesspolicy.upsert_calls[0]
        self.assertEqual(upsert["where"], {"toolConfigId": "tool-1"})
        self.assertEqual(upsert["data"]["create"]["defaultChatAccess"], "read")
        self.assertEqual(upsert["data"]["create"]["defaultWorkspaceAccess"], "read_write")
        self.assertEqual(upsert["data"]["update"], {"defaultChatAccess": "read", "defaultWorkspaceAccess": "read_write"})
        self.assertEqual(
            fake_db.tx_state.tooluseraccess.create_many_calls[0]["data"],
            [{"policyId": "policy-1", "userId": "user-1", "chatAccess": "read", "workspaceAccess": None}],
        )
        self.assertEqual(
            fake_db.tx_state.toolauthgroupaccess.create_many_calls[0]["data"],
            [{"policyId": "policy-1", "authGroupId": "group-1", "chatAccess": None, "workspaceAccess": "read_write"}],
        )
        self.assertEqual(result.users[0].principal_id, "user-1")
        self.assertEqual(result.groups[0].principal_id, "group-1")

    async def test_replace_tool_access_policy_caps_default_read_write_for_read_only_tool_only(self) -> None:
        request = ToolAccessPolicyUpdateRequest(
            default_chat_access="read_write",
            default_workspace_access="read_write",
            users=[ToolAccessEntryUpdate(principal_id="user-1", chat_access="read_write", workspace_access=None)],
            groups=[ToolAccessEntryUpdate(principal_id="group-1", chat_access=None, workspace_access="read_write")],
        )
        fake_db = _FakeDb(
            tool_allow_write=False,
            existing_users=[SimpleNamespace(id="user-1")],
            existing_groups=[SimpleNamespace(id="group-1")],
        )

        with mock.patch.object(self.repo, "_get_db", mock.AsyncMock(return_value=fake_db)):
            result = await self.repo.replace_tool_access_policy("tool-1", request)

        upsert = fake_db.tx_state.toolaccesspolicy.upsert_calls[0]
        self.assertEqual(upsert["data"]["create"]["defaultChatAccess"], "read")
        self.assertEqual(upsert["data"]["create"]["defaultWorkspaceAccess"], "read")
        self.assertEqual(upsert["data"]["update"], {"defaultChatAccess": "read", "defaultWorkspaceAccess": "read"})
        self.assertEqual(
            fake_db.tx_state.tooluseraccess.create_many_calls[0]["data"],
            [{"policyId": "policy-1", "userId": "user-1", "chatAccess": "read_write", "workspaceAccess": None}],
        )
        self.assertEqual(
            fake_db.tx_state.toolauthgroupaccess.create_many_calls[0]["data"],
            [{"policyId": "policy-1", "authGroupId": "group-1", "chatAccess": None, "workspaceAccess": "read_write"}],
        )
        self.assertEqual(result.default_chat_access, "read")
        self.assertEqual(result.default_workspace_access, "read")
        self.assertEqual(result.users[0].chat_access, "read_write")
        self.assertEqual(result.groups[0].workspace_access, "read_write")

    async def test_replace_tool_access_policy_maps_unique_conflict_to_409(self) -> None:
        request = ToolAccessPolicyUpdateRequest(
            default_chat_access="deny",
            default_workspace_access="deny",
            users=[ToolAccessEntryUpdate(principal_id="user-1", chat_access="read", workspace_access=None)],
            groups=[],
        )
        fake_db = _FakeDb(
            existing_users=[SimpleNamespace(id="user-1")],
            create_many_exc=RuntimeError("unique boom"),
        )

        with (
            mock.patch.object(self.repo, "_get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(repository_module, "_is_unique_violation", return_value=True),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await self.repo.replace_tool_access_policy("tool-1", request)

        self.assertEqual(ctx.exception.status_code, 409)


class ToolAccessRouteTests(unittest.IsolatedAsyncioTestCase):
    def _access_route(self, path: str, method: str) -> Any:
        return next(
            (route for route in routes.router.routes if getattr(route, "path", None) == path and method in getattr(route, "methods", set())),
            None,
        )

    def test_access_routes_require_admin(self) -> None:
        for method in ("GET", "PUT"):
            route = self._access_route("/indexes/tools/{tool_id}/access", method)
            self.assertIsNotNone(route, f"{method} /indexes/tools/{{tool_id}}/access route not found")
            dep_calls = [dep.call for dep in getattr(route, "dependant", SimpleNamespace(dependencies=[])).dependencies]
            self.assertIn(routes.require_admin, dep_calls)

    async def test_update_tool_access_policy_does_not_rebuild_or_notify(self) -> None:
        response = ToolAccessPolicyResponse(
            tool_id="tool-1",
            default_chat_access="deny",
            default_workspace_access="deny",
            users=[],
            groups=[],
        )
        request = ToolAccessPolicyUpdateRequest(
            default_chat_access="deny",
            default_workspace_access="deny",
            users=[],
            groups=[],
        )
        mock_repo = mock.AsyncMock()
        mock_repo.replace_tool_access_policy = mock.AsyncMock(return_value=response)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()) as mock_initialize,
            mock.patch("ragtime.indexer.routes.notify_tools_changed") as mock_notify,
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache") as mock_invalidate,
        ):
            result = await routes.update_tool_access_policy("tool-1", request)

        self.assertEqual(result.tool_id, "tool-1")
        mock_initialize.assert_not_awaited()
        mock_notify.assert_not_called()
        mock_invalidate.assert_not_called()

    async def test_create_tool_config_ensures_tool_access_policy(self) -> None:
        request = routes.CreateToolConfigRequest(
            name="Tool One",
            tool_type=ToolType.POSTGRES,
            description="",
            connection_config={"host": "db.example.test"},
        )
        created = ToolConfig(
            id="tool-1",
            name="Tool One",
            tool_type=ToolType.POSTGRES,
            description="",
            connection_config={"host": "db.example.test"},
        )
        mock_repo = mock.AsyncMock()
        mock_repo.create_tool_config = mock.AsyncMock(return_value=created)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.ensure_tool_access_policy", mock.AsyncMock()) as mock_ensure,
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()) as mock_initialize,
            mock.patch("ragtime.indexer.routes.notify_tools_changed") as mock_notify,
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache") as mock_invalidate,
        ):
            result = await routes.create_tool_config(request, cast(Any, SimpleNamespace(id="admin-1")))

        self.assertEqual(result.id, "tool-1")
        mock_ensure.assert_awaited_once_with("tool-1")
        mock_initialize.assert_awaited_once()
        mock_notify.assert_called_once()
        mock_invalidate.assert_called_once()

    async def test_import_tool_config_ensures_tool_access_policy(self) -> None:
        envelope = self._build_export_envelope(
            ToolConfig(
                id="tool-export",
                name="Imported Tool",
                tool_type=ToolType.POSTGRES,
                description="",
                connection_config={"host": "db.example.test"},
                enabled=True,
            )
        )
        created = ToolConfig(
            id="tool-1",
            name="Imported Tool",
            tool_type=ToolType.POSTGRES,
            description="",
            connection_config={"host": "db.example.test"},
            enabled=False,
        )
        mock_repo = mock.AsyncMock()
        mock_repo.list_tool_configs = mock.AsyncMock(return_value=[])
        mock_repo.create_tool_config = mock.AsyncMock(return_value=created)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.ensure_tool_access_policy", mock.AsyncMock()) as mock_ensure,
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()) as mock_initialize,
            mock.patch("ragtime.indexer.routes.notify_tools_changed") as mock_notify,
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache") as mock_invalidate,
        ):
            result = await routes.import_tool_config(routes.ToolImportRequest(password=EXPORT_PASSWORD, file_content=json.dumps(envelope)))

        self.assertEqual(result.id, "tool-1")
        mock_ensure.assert_awaited_once_with("tool-1")
        mock_initialize.assert_awaited_once()
        mock_notify.assert_called_once()
        mock_invalidate.assert_called_once()

    def _build_export_envelope(self, config: ToolConfig) -> dict[str, Any]:
        envelope = routes._build_tool_export_envelope(config, EXPORT_PASSWORD)
        payload = json.loads(decrypt_with_password(envelope["payload"], EXPORT_PASSWORD, envelope["kdf"]["salt"]))
        self.assertNotIn("users", payload)
        self.assertNotIn("groups", payload)
        self.assertNotIn("default_chat_access", payload)
        self.assertNotIn("default_workspace_access", payload)
        return envelope


if __name__ == "__main__":
    unittest.main()
