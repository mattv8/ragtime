import sys
import types
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from fastapi import HTTPException

from ragtime.userspace.models import (
    UpsertWorkspaceAgentGrantRequest,
    UserSpaceWorkspace,
    WorkspaceMember,
)
from ragtime.userspace.service import UserSpaceService

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

_NOW = datetime(2026, 5, 5, tzinfo=timezone.utc)


class _FakeWorkspaceAgentGrantTable:
    def __init__(self, rows: list[SimpleNamespace] | None = None) -> None:
        self.rows = rows or []
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    async def find_many(
        self,
        *,
        where: dict[str, str],
        order: dict[str, str] | None = None,
    ) -> list[SimpleNamespace]:
        return [row for row in self.rows if all(getattr(row, key, None) == value for key, value in where.items())]

    async def find_first(self, *, where: dict[str, str]) -> SimpleNamespace | None:
        for row in self.rows:
            if all(getattr(row, key, None) == value for key, value in where.items()):
                return row
        return None

    async def create(self, *, data: dict[str, Any]) -> SimpleNamespace:
        self.created.append(data)
        row = SimpleNamespace(id="grant-created", **data)
        self.rows.append(row)
        return row

    async def update(
        self,
        *,
        where: dict[str, str],
        data: dict[str, Any],
    ) -> SimpleNamespace:
        self.updated.append({"where": where, "data": data})
        row = await self.find_first(where=where)
        if row is None:
            row = SimpleNamespace(id=where["id"])
        for key, value in data.items():
            setattr(row, key, value)
        return row

    async def delete(self, *, where: dict[str, str]) -> None:
        self.deleted.append(where)


class _FakeWorkspaceTable:
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    async def find_unique(self, *, where: dict[str, str]) -> SimpleNamespace | None:
        workspace_id = str(where.get("id") or "")
        name = self.names.get(workspace_id)
        if name is None:
            return None
        return SimpleNamespace(id=workspace_id, name=name)


class _FakeUserTable:
    async def find_unique(self, *, where: dict[str, str]) -> SimpleNamespace | None:
        user_id = str(where.get("id") or "")
        return SimpleNamespace(id=user_id, username=f"user-{user_id}")


class _GrantService(UserSpaceService):
    def __init__(
        self,
        roles: dict[str, str | None],
        *,
        owner_user_ids: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.roles = roles
        self.owner_user_ids = owner_user_ids or {}
        self.enforcements: list[tuple[str, str | None]] = []
        self.audit_events: list[tuple[str, str | None, str, dict[str, Any]]] = []

    async def _enforce_workspace_access(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str | None = None,
        is_admin: bool = False,
    ) -> UserSpaceWorkspace:
        self.enforcements.append((workspace_id, required_role))
        role = self.roles.get(workspace_id)
        if role is None and not is_admin:
            raise HTTPException(status_code=404, detail="Workspace not found")
        if required_role == "editor" and role not in {"owner", "editor"} and not is_admin:
            raise HTTPException(status_code=403, detail="Editor access required")
        if required_role == "owner" and role != "owner" and not is_admin:
            raise HTTPException(status_code=403, detail="Owner access required")
        return UserSpaceWorkspace(
            id=workspace_id,
            name=f"Workspace {workspace_id}",
            owner_user_id=self.owner_user_ids.get(
                workspace_id,
                "user-1" if role == "owner" else "owner-other",
            ),
            members=([] if role in {None, "owner"} else [WorkspaceMember(user_id=user_id, role=cast(Any, role))]),
            created_at=_NOW,
            updated_at=_NOW,
        )

    async def _record_runtime_audit_event(
        self,
        workspace_id: str,
        user_id: str | None,
        event_type: str,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
        created_at: datetime | None = None,
    ) -> bool:
        self.audit_events.append((workspace_id, user_id, event_type, payload))
        return True


class UserSpaceAgentGrantTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_grant_allows_source_editor_and_target_viewer(self) -> None:
        service = _GrantService({"source": "editor", "target": "viewer"})
        grant_table = _FakeWorkspaceAgentGrantTable()
        fake_db = SimpleNamespace(
            workspaceagentgrant=grant_table,
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            grant = await service.upsert_workspace_agent_grant(
                "source",
                UpsertWorkspaceAgentGrantRequest(
                    target_workspace_id="target",
                    access_mode="read",
                ),
                "user-2",
            )

        self.assertEqual(grant.source_workspace_id, "source")
        self.assertEqual(grant.target_workspace_id, "target")
        self.assertEqual(grant.access_mode, "read")
        self.assertEqual(grant.source_workspace_name, "Workspace source")
        self.assertEqual(grant.target_workspace_name, "Workspace target")
        self.assertEqual(
            service.enforcements,
            [("source", "editor"), ("target", "viewer")],
        )
        self.assertEqual(grant_table.created[0]["accessMode"], "read")
        self.assertEqual(grant.sqlite_access_mode, "none")
        self.assertEqual(grant_table.created[0]["sqliteAccessMode"], "none")
        self.assertEqual(service.audit_events[0][3]["sqlite_access_mode"], "none")

    async def test_read_write_grant_requires_target_editor(self) -> None:
        service = _GrantService({"source": "editor", "target": "viewer"})
        fake_db = SimpleNamespace(
            workspaceagentgrant=_FakeWorkspaceAgentGrantTable(),
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            with self.assertRaises(HTTPException) as raised:
                await service.upsert_workspace_agent_grant(
                    "source",
                    UpsertWorkspaceAgentGrantRequest(
                        target_workspace_id="target",
                        access_mode="read_write",
                    ),
                    "user-2",
                )

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(
            service.enforcements,
            [("source", "editor"), ("target", "editor")],
        )

    async def test_target_listing_for_viewer_omits_read_write_grants(self) -> None:
        service = _GrantService({"target": "viewer"})
        grant_table = _FakeWorkspaceAgentGrantTable(
            [
                SimpleNamespace(
                    id="read-grant",
                    sourceWorkspaceId="source-read",
                    targetWorkspaceId="target",
                    accessMode="read",
                    grantedByUserId="user-1",
                    expiresAt=None,
                    createdAt=_NOW,
                    updatedAt=_NOW,
                ),
                SimpleNamespace(
                    id="write-grant",
                    sourceWorkspaceId="source-write",
                    targetWorkspaceId="target",
                    accessMode="read_write",
                    grantedByUserId="user-1",
                    expiresAt=None,
                    createdAt=_NOW,
                    updatedAt=_NOW,
                ),
            ]
        )
        fake_db = SimpleNamespace(
            workspaceagentgrant=grant_table,
            workspace=_FakeWorkspaceTable(
                {
                    "source-read": "Read Source",
                    "source-write": "Write Source",
                    "target": "Target",
                }
            ),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            grants = await service.list_workspace_agent_grants(
                "target",
                "user-2",
                direction="target",
            )

        self.assertEqual([grant.id for grant in grants], ["read-grant"])
        self.assertEqual(grants[0].source_workspace_name, "Read Source")
        self.assertEqual(grants[0].sqlite_access_mode, "none")

    async def test_legacy_missing_sqlite_mode_maps_to_none(self) -> None:
        service = _GrantService({"source": "editor"})
        grant_table = _FakeWorkspaceAgentGrantTable(
            [
                SimpleNamespace(
                    id="legacy-grant",
                    sourceWorkspaceId="source",
                    targetWorkspaceId="target",
                    accessMode="read",
                    grantedByUserId="user-1",
                    expiresAt=None,
                    createdAt=_NOW,
                    updatedAt=_NOW,
                )
            ]
        )
        fake_db = SimpleNamespace(
            workspaceagentgrant=grant_table,
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            grants = await service.list_workspace_agent_grants(
                "source",
                "user-2",
                direction="source",
            )

        self.assertEqual(len(grants), 1)
        self.assertEqual(grants[0].sqlite_access_mode, "none")

    async def test_omitted_sqlite_mode_preserves_existing_non_none_value(self) -> None:
        service = _GrantService(
            {"source": "editor", "target": "owner"},
            owner_user_ids={"target": "user-2"},
        )
        existing_row = SimpleNamespace(
            id="grant-existing",
            sourceWorkspaceId="source",
            targetWorkspaceId="target",
            accessMode="read",
            sqliteAccessMode="read_write",
            grantedByUserId="user-1",
            expiresAt=None,
            createdAt=_NOW,
            updatedAt=_NOW,
        )
        grant_table = _FakeWorkspaceAgentGrantTable([existing_row])
        fake_db = SimpleNamespace(
            workspaceagentgrant=grant_table,
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            grant = await service.upsert_workspace_agent_grant(
                "source",
                UpsertWorkspaceAgentGrantRequest(
                    target_workspace_id="target",
                    access_mode="read",
                ),
                "user-2",
            )

        self.assertEqual(grant.sqlite_access_mode, "read_write")
        self.assertEqual(existing_row.sqliteAccessMode, "read_write")
        self.assertNotIn("sqliteAccessMode", grant_table.updated[0]["data"])
        self.assertEqual(service.audit_events[0][3]["sqlite_access_mode"], "read_write")

    async def test_explicit_sqlite_mode_requires_real_target_owner(self) -> None:
        for sqlite_access_mode in ("none", "read", "read_write"):
            with self.subTest(sqlite_access_mode=sqlite_access_mode):
                service = _GrantService(
                    {"source": "editor", "target": "editor"},
                    owner_user_ids={"target": "owner-other"},
                )
                fake_db = SimpleNamespace(
                    workspaceagentgrant=_FakeWorkspaceAgentGrantTable(),
                    workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
                    user=_FakeUserTable(),
                )

                async def _fake_get_db() -> SimpleNamespace:
                    return fake_db

                with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
                    with self.assertRaises(HTTPException) as raised:
                        await service.upsert_workspace_agent_grant(
                            "source",
                            UpsertWorkspaceAgentGrantRequest(
                                target_workspace_id="target",
                                access_mode="read",
                                sqlite_access_mode=sqlite_access_mode,
                            ),
                            "user-2",
                        )

                self.assertEqual(raised.exception.status_code, 403)

    async def test_admin_without_target_ownership_is_denied_explicit_sqlite_mode(self) -> None:
        service = _GrantService(
            {"source": None, "target": None},
            owner_user_ids={"target": "owner-other"},
        )
        fake_db = SimpleNamespace(
            workspaceagentgrant=_FakeWorkspaceAgentGrantTable(),
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            with self.assertRaises(HTTPException) as raised:
                await service.upsert_workspace_agent_grant(
                    "source",
                    UpsertWorkspaceAgentGrantRequest(
                        target_workspace_id="target",
                        access_mode="read",
                        sqlite_access_mode="read",
                    ),
                    "user-2",
                    is_admin=True,
                )

        self.assertEqual(raised.exception.status_code, 403)

    async def test_target_owner_can_grant_and_downgrade_sqlite_access(self) -> None:
        service = _GrantService(
            {"source": "editor", "target": "owner"},
            owner_user_ids={"target": "user-2"},
        )
        grant_table = _FakeWorkspaceAgentGrantTable()
        fake_db = SimpleNamespace(
            workspaceagentgrant=grant_table,
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            created = await service.upsert_workspace_agent_grant(
                "source",
                UpsertWorkspaceAgentGrantRequest(
                    target_workspace_id="target",
                    access_mode="read",
                    sqlite_access_mode="read_write",
                ),
                "user-2",
            )
            downgraded = await service.upsert_workspace_agent_grant(
                "source",
                UpsertWorkspaceAgentGrantRequest(
                    target_workspace_id="target",
                    access_mode="read",
                    sqlite_access_mode="none",
                ),
                "user-2",
            )

        self.assertEqual(created.sqlite_access_mode, "read_write")
        self.assertEqual(downgraded.sqlite_access_mode, "none")
        self.assertEqual(grant_table.created[0]["sqliteAccessMode"], "read_write")
        self.assertEqual(grant_table.updated[0]["data"]["sqliteAccessMode"], "none")

    async def test_source_side_revoke_is_denied_for_active_sqlite_grant(self) -> None:
        service = _GrantService(
            {"source": "editor", "target": "viewer"},
            owner_user_ids={"target": "owner-other"},
        )
        grant_table = _FakeWorkspaceAgentGrantTable(
            [
                SimpleNamespace(
                    id="sqlite-grant",
                    sourceWorkspaceId="source",
                    targetWorkspaceId="target",
                    accessMode="read",
                    sqliteAccessMode="read",
                    grantedByUserId="user-1",
                    expiresAt=None,
                    createdAt=_NOW,
                    updatedAt=_NOW,
                )
            ]
        )
        fake_db = SimpleNamespace(
            workspaceagentgrant=grant_table,
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            with self.assertRaises(HTTPException) as raised:
                await service.revoke_workspace_agent_grant(
                    "source",
                    "target",
                    "user-2",
                )

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(grant_table.deleted, [])

    async def test_target_owner_can_revoke_active_sqlite_grant(self) -> None:
        service = _GrantService(
            {"source": None, "target": "owner"},
            owner_user_ids={"target": "user-2"},
        )
        grant_table = _FakeWorkspaceAgentGrantTable(
            [
                SimpleNamespace(
                    id="sqlite-grant",
                    sourceWorkspaceId="source",
                    targetWorkspaceId="target",
                    accessMode="read",
                    sqliteAccessMode="read_write",
                    grantedByUserId="user-1",
                    expiresAt=None,
                    createdAt=_NOW,
                    updatedAt=_NOW,
                )
            ]
        )
        fake_db = SimpleNamespace(
            workspaceagentgrant=grant_table,
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            revoked = await service.revoke_workspace_agent_grant(
                "source",
                "target",
                "user-2",
            )

        self.assertTrue(revoked)
        self.assertEqual(grant_table.deleted, [{"id": "sqlite-grant"}])
        self.assertEqual(service.audit_events[0][3]["sqlite_access_mode"], "read_write")

    async def test_revoke_allows_target_viewer_to_remove_read_grant(self) -> None:
        service = _GrantService({"source": None, "target": "viewer"})
        grant_table = _FakeWorkspaceAgentGrantTable(
            [
                SimpleNamespace(
                    id="read-grant",
                    sourceWorkspaceId="source",
                    targetWorkspaceId="target",
                    accessMode="read",
                    grantedByUserId="user-1",
                    expiresAt=None,
                    createdAt=_NOW,
                    updatedAt=_NOW,
                )
            ]
        )
        fake_db = SimpleNamespace(
            workspaceagentgrant=grant_table,
            workspace=_FakeWorkspaceTable({"source": "Source", "target": "Target"}),
            user=_FakeUserTable(),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            revoked = await service.revoke_workspace_agent_grant(
                "source",
                "target",
                "user-2",
            )

        self.assertTrue(revoked)
        self.assertEqual(grant_table.deleted, [{"id": "read-grant"}])
        self.assertEqual(
            service.enforcements,
            [("source", "editor"), ("target", "viewer")],
        )
        self.assertEqual(service.audit_events[0][3]["sqlite_access_mode"], "none")


if __name__ == "__main__":
    unittest.main()
