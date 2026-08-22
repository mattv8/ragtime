from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace import routes as userspace_routes
from ragtime.userspace import sqlite_inspector as sqlite_inspector_helpers
from ragtime.userspace.models import SqliteInspectorDatabaseSummary
from ragtime.userspace.service import UserSpaceService
from tests.test_userspace_sqlite_shared import _FakeGrantTable, _FakeUserTable, _FakeWorkspaceMemberTable, _WorkspaceRowBase


@dataclass
class _WorkspaceRow(_WorkspaceRowBase):
    """Workspace row for linked SQLite inspector tests."""

    sqlite_persistence_mode: str = "exclude"


class _FakeWorkspaceTable:
    def __init__(self, rows: dict[str, _WorkspaceRow], roles: dict[tuple[str, str], str]) -> None:
        self.rows = rows
        self.roles = roles

    async def find_unique(self, *, where: dict[str, str], include: dict[str, object] | None = None) -> SimpleNamespace | None:
        workspace_id = str(where.get("id") or "")
        row = self.rows.get(workspace_id)
        if row is None:
            return None
        members = [
            SimpleNamespace(userId=user_id, role=role) for (member_workspace_id, user_id), role in self.roles.items() if member_workspace_id == workspace_id
        ]
        return SimpleNamespace(
            id=workspace_id,
            ownerUserId=row.owner_user_id,
            name=row.name,
            description="",
            sqlite_persistence_mode=row.sqlite_persistence_mode,
            sqlitePersistenceMode=row.sqlite_persistence_mode,
            members=members,
            toolSelections=[],
            toolGroupSelections=[],
            toolOptions=[],
            owner=SimpleNamespace(id=row.owner_user_id),
            createdAt=datetime(2026, 8, 6, tzinfo=timezone.utc),
            updatedAt=datetime(2026, 8, 6, tzinfo=timezone.utc),
        )

    async def find_many(self, *, where: dict[str, Any] | None = None, order: dict[str, str] | None = None) -> list[SimpleNamespace]:
        workspace_ids = {str(value) for value in (where or {}).get("id", {}).get("in", [])}
        rows = []
        for workspace_id, row in self.rows.items():
            if workspace_ids and workspace_id not in workspace_ids:
                continue
            rows.append(
                SimpleNamespace(
                    id=workspace_id,
                    ownerUserId=row.owner_user_id,
                    name=row.name,
                    sqlite_persistence_mode=row.sqlite_persistence_mode,
                )
            )
        if order == {"name": "asc"}:
            rows.sort(key=lambda item: (str(item.name).casefold(), str(item.id)))
        return rows


class SqliteInspectorDatabaseSummaryModelTests(unittest.TestCase):
    def test_linked_database_summary_serializes_identity_access_and_missing_state(self) -> None:
        summary = SqliteInspectorDatabaseSummary(
            name="app.sqlite3",
            relative_path=".ragtime/db/app.sqlite3",
            size_bytes=0,
            table_count=0,
            last_modified_ms=None,
            owner_workspace_id="target-ws",
            owner_workspace_name="Target Workspace",
            ownership="linked",
            access_mode="read",
            persistence_mode="exclude",
            initialized=False,
        )

        self.assertEqual(
            summary.model_dump(),
            {
                "name": "app.sqlite3",
                "relative_path": ".ragtime/db/app.sqlite3",
                "size_bytes": 0,
                "table_count": 0,
                "last_modified_ms": None,
                "owner_workspace_id": "target-ws",
                "owner_workspace_name": "Target Workspace",
                "ownership": "linked",
                "access_mode": "read",
                "persistence_mode": "exclude",
                "initialized": False,
            },
        )


class LinkedSqliteInspectorServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = UserSpaceService()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)
        self.promoted_workspaces: list[str] = []
        self.audit_events: list[dict[str, Any]] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _grant(
        self,
        target_workspace_id: str,
        *,
        sqlite_access_mode: str = "read",
        grant_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=grant_id or f"grant-{target_workspace_id}",
            sourceWorkspaceId="source-ws",
            targetWorkspaceId=target_workspace_id,
            accessMode="read_write",
            sqliteAccessMode=sqlite_access_mode,
            expiresAt=expires_at,
        )

    def _db(
        self,
        *,
        workspace_rows: dict[str, _WorkspaceRow],
        roles: dict[tuple[str, str], str],
        grant_rows: list[SimpleNamespace],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            workspace=_FakeWorkspaceTable(workspace_rows, roles),
            workspacemember=_FakeWorkspaceMemberTable(roles),
            workspaceagentgrant=_FakeGrantTable(grant_rows),
            user=_FakeUserTable(),
        )

    def _workspace_files_dir(self, workspace_id: str) -> Path:
        return self.workspace_root / workspace_id / "files"

    def _initialize_workspace_database(self, workspace_id: str, database_name: str = "app.sqlite3") -> None:
        files_dir = self._workspace_files_dir(workspace_id)
        files_dir.mkdir(parents=True, exist_ok=True)
        sqlite_inspector_helpers.initialize_database(files_dir, database_name)

    def _create_table_with_rows(self, workspace_id: str, table_name: str = "items") -> None:
        files_dir = self._workspace_files_dir(workspace_id)
        sqlite_inspector_helpers.create_table(
            files_dir,
            "app.sqlite3",
            table_name,
            [
                sqlite_inspector_helpers.ColumnDefinition(name="id", type="INTEGER", primary_key=True, not_null=True),
                sqlite_inspector_helpers.ColumnDefinition(name="name", type="TEXT"),
            ],
        )
        sqlite_inspector_helpers.insert_row(files_dir, "app.sqlite3", table_name, {"id": 1, "name": "Ada"})

    def _sqlite_upload_copy(self, source_workspace_id: str) -> Path:
        source_path = self.workspace_root / f"{source_workspace_id}-upload.sqlite3"
        sqlite_inspector_helpers.initialize_database(self.workspace_root / source_workspace_id / "upload-files", "upload.sqlite3")
        original = self.workspace_root / source_workspace_id / "upload-files" / ".ragtime" / "db" / "upload.sqlite3"
        source_path.write_bytes(original.read_bytes())
        return source_path

    def _patch_common(self, fake_db: SimpleNamespace, workspace_rows: dict[str, _WorkspaceRow]) -> tuple[mock._patch, ...]:
        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        async def _ensure_sqlite_mode_included(workspace_id: str) -> bool:
            self.promoted_workspaces.append(workspace_id)
            row = workspace_rows[workspace_id]
            if row.sqlite_persistence_mode == "include":
                return False
            row.sqlite_persistence_mode = "include"
            return True

        async def _record_runtime_audit_event(
            workspace_id: str,
            user_id: str | None,
            event_type: str,
            payload: dict[str, object],
            *,
            session_id: str | None = None,
            created_at: datetime | None = None,
        ) -> bool:
            self.audit_events.append(
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "event_type": event_type,
                    "payload": payload,
                    "session_id": session_id,
                    "created_at": created_at,
                }
            )
            return True

        return (
            mock.patch("ragtime.userspace.service.get_db", new=_fake_get_db),
            mock.patch.object(self.service, "_workspace_files_dir", side_effect=self._workspace_files_dir),
            mock.patch.object(self.service, "_ensure_sqlite_mode_included", side_effect=_ensure_sqlite_mode_included),
            mock.patch.object(self.service, "_record_runtime_audit_event", new=_record_runtime_audit_event),
            mock.patch.object(self.service, "_sync_runtime_bootstrap_config", return_value=None),
        )

    async def test_list_sqlite_databases_merges_owned_and_linked_access_modes_and_missing_state(self) -> None:
        workspace_rows = {
            "source-ws": _WorkspaceRow(owner_user_id="owner-source", name="Source Workspace", sqlite_persistence_mode="include"),
            "target-read": _WorkspaceRow(owner_user_id="owner-read", name="Alpha", sqlite_persistence_mode="include"),
            "target-readwrite-viewer": _WorkspaceRow(owner_user_id="owner-rw-viewer", name="Beta", sqlite_persistence_mode="include"),
            "target-readwrite-editor": _WorkspaceRow(owner_user_id="owner-rw-editor", name="beta", sqlite_persistence_mode="exclude"),
            "target-missing": _WorkspaceRow(owner_user_id="owner-missing", name="Delta", sqlite_persistence_mode="exclude"),
            "target-expired": _WorkspaceRow(owner_user_id="owner-expired", name="Gamma", sqlite_persistence_mode="include"),
        }
        roles = {
            ("source-ws", "viewer-user"): "viewer",
            ("target-read", "viewer-user"): "viewer",
            ("target-readwrite-viewer", "viewer-user"): "viewer",
            ("target-readwrite-editor", "viewer-user"): "editor",
            ("target-missing", "viewer-user"): "viewer",
            ("target-expired", "viewer-user"): "editor",
        }
        fake_db = self._db(
            workspace_rows=workspace_rows,
            roles=roles,
            grant_rows=[
                self._grant("target-read", sqlite_access_mode="read"),
                self._grant("target-readwrite-viewer", sqlite_access_mode="read_write"),
                self._grant("target-readwrite-editor", sqlite_access_mode="read_write"),
                self._grant("target-missing", sqlite_access_mode="read"),
                self._grant("target-expired", sqlite_access_mode="read", expires_at=datetime(2026, 8, 5, tzinfo=timezone.utc)),
            ],
        )
        self._initialize_workspace_database("source-ws", "zeta.sqlite3")
        self._initialize_workspace_database("source-ws", "app.sqlite3")
        self._initialize_workspace_database("target-read")
        self._create_table_with_rows("target-read")
        self._initialize_workspace_database("target-readwrite-viewer")
        self._initialize_workspace_database("target-readwrite-editor")

        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db, workspace_rows):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=datetime(2026, 8, 6, tzinfo=timezone.utc)))
            databases, total_bytes, mode = await self.service.list_sqlite_databases("source-ws", "viewer-user")

        self.assertEqual(mode, "include")
        self.assertEqual(
            [(database.ownership, database.owner_workspace_name, database.name, database.access_mode, database.initialized) for database in databases],
            [
                ("owned", "Source Workspace", "app.sqlite3", "read", True),
                ("owned", "Source Workspace", "zeta.sqlite3", "read", True),
                ("linked", "Alpha", "app.sqlite3", "read", True),
                ("linked", "beta", "app.sqlite3", "read_write", True),
                ("linked", "Beta", "app.sqlite3", "read", True),
                ("linked", "Delta", "app.sqlite3", "read", False),
            ],
        )
        self.assertEqual(databases[-1].last_modified_ms, None)
        self.assertEqual(databases[-1].size_bytes, 0)
        self.assertEqual(databases[-1].table_count, 0)
        self.assertEqual(databases[-1].persistence_mode, "exclude")
        self.assertEqual(total_bytes, sum(database.size_bytes for database in databases if database.initialized))

    async def test_list_sqlite_databases_sets_owned_read_write_for_editor_and_omits_linked_for_admin_without_source_membership(self) -> None:
        workspace_rows = {
            "source-ws": _WorkspaceRow(owner_user_id="owner-source", name="Source Workspace", sqlite_persistence_mode="exclude"),
            "target-ws": _WorkspaceRow(owner_user_id="owner-target", name="Target Workspace", sqlite_persistence_mode="include"),
        }
        editor_db = self._db(
            workspace_rows=workspace_rows,
            roles={("source-ws", "editor-user"): "editor", ("target-ws", "editor-user"): "editor"},
            grant_rows=[self._grant("target-ws", sqlite_access_mode="read_write")],
        )
        admin_db = self._db(
            workspace_rows=workspace_rows,
            roles={("target-ws", "admin-user"): "editor"},
            grant_rows=[self._grant("target-ws", sqlite_access_mode="read_write")],
        )
        self._initialize_workspace_database("source-ws")
        self._initialize_workspace_database("target-ws")

        with ExitStack() as stack:
            for patcher in self._patch_common(editor_db, workspace_rows):
                stack.enter_context(patcher)
            editor_databases, _, _ = await self.service.list_sqlite_databases("source-ws", "editor-user")
        self.assertEqual(editor_databases[0].access_mode, "read_write")

        with ExitStack() as stack:
            for patcher in self._patch_common(admin_db, workspace_rows):
                stack.enter_context(patcher)
            admin_databases, _, _ = await self.service.list_sqlite_databases("source-ws", "admin-user", is_admin=True)
        self.assertEqual([(database.ownership, database.owner_workspace_id) for database in admin_databases], [("owned", "source-ws")])

    async def test_linked_read_operations_allow_queries_exports_and_table_reads(self) -> None:
        workspace_rows = {
            "source-ws": _WorkspaceRow(owner_user_id="owner-source", name="Source Workspace"),
            "target-ws": _WorkspaceRow(owner_user_id="owner-target", name="Target Workspace", sqlite_persistence_mode="include"),
        }
        roles = {("source-ws", "viewer-user"): "viewer", ("target-ws", "viewer-user"): "viewer"}
        fake_db = self._db(workspace_rows=workspace_rows, roles=roles, grant_rows=[self._grant("target-ws", sqlite_access_mode="read")])
        self._initialize_workspace_database("target-ws")
        self._create_table_with_rows("target-ws")

        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db, workspace_rows):
                stack.enter_context(patcher)
            summary, tables = await self.service.list_sqlite_tables("source-ws", "viewer-user", "app.sqlite3", owner_workspace_id="target-ws")
            schema = await self.service.get_sqlite_table_schema("source-ws", "viewer-user", "app.sqlite3", "items", owner_workspace_id="target-ws")
            page = await self.service.list_sqlite_rows("source-ws", "viewer-user", "app.sqlite3", "items", owner_workspace_id="target-ws")
            query = await self.service.execute_sqlite_readonly_query(
                "source-ws", "viewer-user", "app.sqlite3", "select id, name from items", owner_workspace_id="target-ws"
            )
            exported_db = await self.service.export_sqlite_database("source-ws", "viewer-user", "app.sqlite3", owner_workspace_id="target-ws")
            exported_csv = await self.service.export_sqlite_table_csv("source-ws", "viewer-user", "app.sqlite3", "items", owner_workspace_id="target-ws")

        self.assertEqual(summary.owner_workspace_id, "target-ws")
        self.assertEqual(summary.access_mode, "read")
        self.assertEqual([table.name for table in tables], ["items"])
        self.assertEqual([column.name for column in schema.columns], ["id", "name"])
        self.assertEqual(page.rows[0]["name"], "Ada")
        self.assertEqual(query.rows, [{"id": 1, "name": "Ada"}])
        self.assertTrue(exported_db.exists())
        self.assertIn("Ada", exported_csv)
        exported_db.unlink(missing_ok=True)

    async def test_linked_read_denies_write_operations_and_non_default_database_targets(self) -> None:
        workspace_rows = {
            "source-ws": _WorkspaceRow(owner_user_id="owner-source", name="Source Workspace"),
            "target-ws": _WorkspaceRow(owner_user_id="owner-target", name="Target Workspace"),
        }
        roles = {("source-ws", "viewer-user"): "viewer", ("target-ws", "viewer-user"): "viewer"}
        fake_db = self._db(workspace_rows=workspace_rows, roles=roles, grant_rows=[self._grant("target-ws", sqlite_access_mode="read_write")])
        self._initialize_workspace_database("target-ws")
        upload_path = self._sqlite_upload_copy("source-ws")

        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db, workspace_rows):
                stack.enter_context(patcher)
            denied_calls = [
                lambda: self.service.initialize_sqlite_database("source-ws", "viewer-user", database_name="app.sqlite3", owner_workspace_id="target-ws"),
                lambda: self.service.import_sqlite_database("source-ws", "viewer-user", "app.sqlite3", upload_path, owner_workspace_id="target-ws"),
                lambda: self.service.delete_sqlite_database("source-ws", "viewer-user", "app.sqlite3", owner_workspace_id="target-ws"),
                lambda: self.service.create_sqlite_table(
                    "source-ws",
                    "viewer-user",
                    "app.sqlite3",
                    "items",
                    [sqlite_inspector_helpers.ColumnDefinition(name="id", type="INTEGER", primary_key=True)],
                    owner_workspace_id="target-ws",
                ),
                lambda: self.service.alter_sqlite_table(
                    "source-ws",
                    "viewer-user",
                    "app.sqlite3",
                    "items",
                    [sqlite_inspector_helpers.TableAlteration(op="rename_table", new_table_name="renamed")],
                    owner_workspace_id="target-ws",
                ),
                lambda: self.service.drop_sqlite_table("source-ws", "viewer-user", "app.sqlite3", "items", owner_workspace_id="target-ws"),
                lambda: self.service.import_sqlite_table_csv(
                    "source-ws", "viewer-user", "app.sqlite3", "items", "id,name\n1,Ada\n", owner_workspace_id="target-ws"
                ),
                lambda: self.service.insert_sqlite_row("source-ws", "viewer-user", "app.sqlite3", "items", {"id": 2}, owner_workspace_id="target-ws"),
                lambda: self.service.update_sqlite_row(
                    "source-ws", "viewer-user", "app.sqlite3", "items", {"id": 1}, {"name": "Grace"}, owner_workspace_id="target-ws"
                ),
                lambda: self.service.delete_sqlite_row("source-ws", "viewer-user", "app.sqlite3", "items", {"id": 1}, owner_workspace_id="target-ws"),
            ]
            for call in denied_calls:
                with self.assertRaises(HTTPException) as raised:
                    await call()
                self.assertEqual(raised.exception.status_code, 403)

            with self.assertRaises(HTTPException) as linked_name_error:
                await self.service.initialize_sqlite_database("source-ws", "viewer-user", database_name="custom.sqlite3", owner_workspace_id="target-ws")
            self.assertEqual(linked_name_error.exception.status_code, 404)

    async def test_linked_read_write_operations_mutate_target_promote_target_and_audit(self) -> None:
        workspace_rows = {
            "source-ws": _WorkspaceRow(owner_user_id="owner-source", name="Source Workspace", sqlite_persistence_mode="exclude"),
            "target-ws": _WorkspaceRow(owner_user_id="owner-target", name="Target Workspace", sqlite_persistence_mode="exclude"),
        }
        roles = {("source-ws", "editor-user"): "viewer", ("target-ws", "editor-user"): "editor"}
        fake_db = self._db(workspace_rows=workspace_rows, roles=roles, grant_rows=[self._grant("target-ws", sqlite_access_mode="read_write")])
        self._initialize_workspace_database("target-ws")
        upload_path = self._sqlite_upload_copy("target-ws")

        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db, workspace_rows):
                stack.enter_context(patcher)
            created_table, promoted = await self.service.create_sqlite_table(
                "source-ws",
                "editor-user",
                "app.sqlite3",
                "items",
                [
                    sqlite_inspector_helpers.ColumnDefinition(name="id", type="INTEGER", primary_key=True, not_null=True),
                    sqlite_inspector_helpers.ColumnDefinition(name="name", type="TEXT"),
                ],
                owner_workspace_id="target-ws",
            )
            inserted_row, _ = await self.service.insert_sqlite_row(
                "source-ws", "editor-user", "app.sqlite3", "items", {"id": 1, "name": "Ada"}, owner_workspace_id="target-ws"
            )
            updated_row, _ = await self.service.update_sqlite_row(
                "source-ws", "editor-user", "app.sqlite3", "items", {"id": 1}, {"name": "Grace"}, owner_workspace_id="target-ws"
            )
            await self.service.delete_sqlite_row("source-ws", "editor-user", "app.sqlite3", "items", {"id": 1}, owner_workspace_id="target-ws")
            imported_table, _ = await self.service.import_sqlite_table_csv(
                "source-ws", "editor-user", "app.sqlite3", "items", "id,name\n2,Linus\n", replace=True, owner_workspace_id="target-ws"
            )
            altered_schema, _ = await self.service.alter_sqlite_table(
                "source-ws",
                "editor-user",
                "app.sqlite3",
                "items",
                [sqlite_inspector_helpers.TableAlteration(op="rename_table", new_table_name="records")],
                owner_workspace_id="target-ws",
            )
            dropped_promoted = await self.service.drop_sqlite_table("source-ws", "editor-user", "app.sqlite3", "records", owner_workspace_id="target-ws")
            await self.service.delete_sqlite_database("source-ws", "editor-user", "app.sqlite3", owner_workspace_id="target-ws")
            initialized_summary, _ = await self.service.initialize_sqlite_database(
                "source-ws", "editor-user", database_name="app.sqlite3", owner_workspace_id="target-ws"
            )
            imported_database, _ = await self.service.import_sqlite_database(
                "source-ws", "editor-user", "app.sqlite3", upload_path, owner_workspace_id="target-ws"
            )
            linked_summary, linked_tables = await self.service.list_sqlite_tables("source-ws", "editor-user", "app.sqlite3", owner_workspace_id="target-ws")

        self.assertTrue(promoted)
        self.assertEqual(created_table.name, "items")
        self.assertEqual(inserted_row["name"], "Ada")
        self.assertEqual(updated_row["name"], "Grace")
        self.assertEqual(imported_table.name, "items")
        self.assertEqual(altered_schema.name, "records")
        self.assertFalse(dropped_promoted)
        self.assertEqual(initialized_summary.owner_workspace_id, "target-ws")
        self.assertEqual(initialized_summary.persistence_mode, "include")
        self.assertEqual(imported_database.owner_workspace_id, "target-ws")
        self.assertEqual(linked_summary.owner_workspace_id, "target-ws")
        self.assertEqual(workspace_rows["source-ws"].sqlite_persistence_mode, "exclude")
        self.assertEqual(workspace_rows["target-ws"].sqlite_persistence_mode, "include")
        self.assertTrue(self.promoted_workspaces)
        self.assertTrue(all(workspace_id == "target-ws" for workspace_id in self.promoted_workspaces))
        self.assertEqual([table.name for table in linked_tables], [])
        linked_audits = [event for event in self.audit_events if event["event_type"] == "sqlite_inspector.linked_mutation"]
        self.assertTrue(linked_audits)
        self.assertEqual(
            linked_audits[0]["payload"],
            {
                "source_workspace_id": "source-ws",
                "target_workspace_id": "target-ws",
                "grant_id": "grant-target-ws",
                "operation": mock.ANY,
                "database_name": "app.sqlite3",
            },
        )
        self.assertNotIn("sql", linked_audits[0]["payload"])
        self.assertNotIn("values", linked_audits[0]["payload"])

    async def test_linked_write_revocation_takes_effect_on_next_request(self) -> None:
        workspace_rows = {
            "source-ws": _WorkspaceRow(owner_user_id="owner-source", name="Source Workspace"),
            "target-ws": _WorkspaceRow(owner_user_id="owner-target", name="Target Workspace"),
        }
        grant = self._grant("target-ws", sqlite_access_mode="read_write")
        roles = {("source-ws", "editor-user"): "viewer", ("target-ws", "editor-user"): "editor"}
        fake_db = self._db(workspace_rows=workspace_rows, roles=roles, grant_rows=[grant])
        self._initialize_workspace_database("target-ws")

        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db, workspace_rows):
                stack.enter_context(patcher)
            await self.service.initialize_sqlite_database("source-ws", "editor-user", owner_workspace_id="target-ws")
            grant.sqliteAccessMode = "read"
            with self.assertRaises(HTTPException) as raised:
                await self.service.create_sqlite_table(
                    "source-ws",
                    "editor-user",
                    "app.sqlite3",
                    "items",
                    [sqlite_inspector_helpers.ColumnDefinition(name="id", type="INTEGER", primary_key=True)],
                    owner_workspace_id="target-ws",
                )
        self.assertEqual(raised.exception.status_code, 403)

    async def test_route_linked_delete_snapshots_target_workspace(self) -> None:
        user = SimpleNamespace(id="user-1", role="editor")
        delete_mock = mock.AsyncMock(return_value=None)
        snapshot_mock = mock.AsyncMock(return_value=None)

        with (
            mock.patch.object(userspace_routes.userspace_service, "delete_sqlite_database", delete_mock),
            mock.patch.object(userspace_routes, "_snapshot_after_sqlite_mutation", snapshot_mock),
        ):
            response = await userspace_routes.delete_workspace_sqlite_database(
                "source-ws",
                "app.sqlite3",
                owner_workspace_id="target-ws",
                user=user,
            )

        delete_mock.assert_awaited_once_with(
            "source-ws",
            "user-1",
            "app.sqlite3",
            owner_workspace_id="target-ws",
            is_admin=False,
        )
        snapshot_mock.assert_awaited_once_with("target-ws", "user-1", "delete database app.sqlite3")
        self.assertEqual(response.workspace_id, "source-ws")
