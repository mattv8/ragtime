from __future__ import annotations

import sys
import types
import unittest
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from fastapi import HTTPException
from prisma import Json

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.models import (  # noqa: E402
    CreateWorkspaceRequest,
    DuplicateWorkspaceRequest,
    UpdateWorkspaceRequest,
    UserSpaceWorkspace,
    UserSpaceWorkspaceArchiveExportRequest,
    WorkspaceMember,
    WorkspaceToolOptionState,
)
from ragtime.userspace.routes import update_workspace as update_workspace_route  # noqa: E402
from ragtime.userspace.service import UserSpaceService  # noqa: E402
from ragtime.userspace.workspace_tool_options import (  # noqa: E402
    load_workspace_tool_options,
    normalize_workspace_tool_options,
    resolve_workspace_tool_write_access,
)

_NOW = datetime(2026, 7, 14, tzinfo=timezone.utc)


class WorkspaceToolOptionHelperTests(unittest.TestCase):
    def test_normalize_filters_to_selected_healthy_write_capable_tools_and_true_flag(self) -> None:
        normalized = normalize_workspace_tool_options(
            {
                "tool-write": {"write_access_enabled": True, "ignored": True},
                "tool-false": {"write_access_enabled": False},
                "tool-readonly": {"write_access_enabled": True},
                "tool-unselected": {"write_access_enabled": True},
                "": {"write_access_enabled": True},
                "tool-invalid": "yes",
            },
            selected_tool_ids={"tool-write", "tool-false", "tool-readonly"},
            write_capable_tool_ids={"tool-write"},
        )

        self.assertEqual(
            normalized,
            {"tool-write": {"write_access_enabled": True}},
        )

    def test_load_and_resolve_workspace_write_grant_with_global_ceiling(self) -> None:
        self.assertEqual(load_workspace_tool_options({"write_access_enabled": True, "ignored": True}), {"write_access_enabled": True})
        self.assertEqual(load_workspace_tool_options({"write_access_enabled": False}), {})
        self.assertFalse(resolve_workspace_tool_write_access(False, None))
        self.assertFalse(resolve_workspace_tool_write_access(True, None))
        self.assertTrue(resolve_workspace_tool_write_access(False, {"write_access_enabled": True}))
        self.assertTrue(resolve_workspace_tool_write_access(True, {"write_access_enabled": True}))


class WorkspaceToolOptionSerializationTests(unittest.TestCase):
    def test_workspace_from_record_serializes_tool_options(self) -> None:
        service = UserSpaceService()

        workspace = service._workspace_from_record(
            SimpleNamespace(
                id="ws-1",
                name="Workspace",
                description="desc",
                sqlitePersistenceMode="include",
                ownerUserId="user-1",
                toolSelectionMode="custom",
                toolSelections=[SimpleNamespace(toolConfigId="tool-1")],
                toolGroupSelections=[],
                toolOptions=[SimpleNamespace(toolConfigId="tool-1", options={"write_access_enabled": True})],
                conversations=[],
                members=[SimpleNamespace(userId="user-1", role="owner")],
                owner=SimpleNamespace(username="owner", displayName="Owner"),
                createdAt=_NOW,
                updatedAt=_NOW,
            )
        )

        self.assertEqual(
            workspace.tool_options,
            {"tool-1": WorkspaceToolOptionState(write_access_enabled=True)},
        )


class _FakeWorkspaceTable:
    def __init__(self) -> None:
        self.update_calls: list[dict[str, Any]] = []
        self.workspace_record = SimpleNamespace(
            id="ws-1",
            name="Workspace",
            description="desc",
            sqlitePersistenceMode="include",
            ownerUserId="owner-1",
            toolSelectionMode="custom",
            toolSelections=[SimpleNamespace(toolConfigId="tool-write"), SimpleNamespace(toolConfigId="tool-read")],
            toolGroupSelections=[],
            toolOptions=[SimpleNamespace(toolConfigId="tool-read", options={"write_access_enabled": True})],
            conversations=[],
            members=[SimpleNamespace(userId="owner-1", role="owner")],
            owner=SimpleNamespace(username="owner", displayName="Owner"),
            createdAt=_NOW,
            updatedAt=_NOW,
        )

    async def update(self, *, where: dict[str, str], data: dict[str, Any]) -> None:
        self.update_calls.append({"where": where, "data": data})

    async def find_unique(self, *, where: dict[str, str], include: dict[str, Any] | None = None) -> SimpleNamespace | None:
        _ = include
        if where.get("id") != "ws-1":
            return None
        return self.workspace_record


class _CaptureTable:
    def __init__(self) -> None:
        self.deleted: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []

    async def delete_many(self, *, where: dict[str, Any]) -> None:
        self.deleted.append(where)

    async def create(self, *, data: dict[str, Any]) -> SimpleNamespace:
        self.created.append(data)
        return SimpleNamespace(**data)


class _WorkspaceMemberTable(_CaptureTable):
    async def find_first(self, *, where: dict[str, Any]) -> SimpleNamespace | None:
        if where.get("userId") == "owner-1":
            return SimpleNamespace(id="member-owner-1", workspaceId=where.get("workspaceId"), userId="owner-1", role="owner")
        return None

    async def update(self, *, where: dict[str, Any], data: dict[str, Any]) -> SimpleNamespace:
        self.created.append({"where": where, "data": data})
        return SimpleNamespace(id=where.get("id"), **data)


class _WorkspaceToolOptionTable(_CaptureTable):
    def __init__(self, workspace_table: _FakeWorkspaceTable) -> None:
        super().__init__()
        self.workspace_table = workspace_table

    async def delete_many(self, *, where: dict[str, Any]) -> None:
        await super().delete_many(where=where)
        self.workspace_table.workspace_record.toolOptions = []

    async def create(self, *, data: dict[str, Any]) -> SimpleNamespace:
        row = await super().create(data=data)
        stored_options = row.options.data if isinstance(row.options, Json) else row.options
        self.workspace_table.workspace_record.toolOptions.append(SimpleNamespace(**{**data, "options": stored_options}))
        return row


class _CreateWorkspaceTable:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.workspace_record = SimpleNamespace(
            id="ws-create-1",
            name="Workspace",
            description="desc",
            sqlitePersistenceMode="include",
            ownerUserId="owner-1",
            toolSelectionMode="custom",
            toolSelections=[],
            toolGroupSelections=[],
            toolOptions=[],
            conversations=[],
            members=[SimpleNamespace(userId="owner-1", role="owner")],
            owner=SimpleNamespace(username="owner", displayName="Owner"),
            createdAt=_NOW,
            updatedAt=_NOW,
        )

    async def create(self, *, data: dict[str, Any], include: dict[str, Any] | None = None) -> SimpleNamespace:
        _ = include
        self.create_calls.append(data)
        self.workspace_record = SimpleNamespace(
            **{
                **self.workspace_record.__dict__,
                "id": data["id"],
                "name": data["name"],
                "description": data["description"],
                "sqlitePersistenceMode": data["sqlitePersistenceMode"],
                "ownerUserId": data["ownerUserId"],
                "toolSelectionMode": data["toolSelectionMode"],
                "createdAt": data["createdAt"],
                "updatedAt": data["updatedAt"],
            }
        )
        return self.workspace_record

    async def find_unique(self, *, where: dict[str, str], include: dict[str, Any] | None = None) -> SimpleNamespace | None:
        _ = include
        if where.get("id") != self.workspace_record.id:
            return None
        return self.workspace_record


class _CreateWorkspaceMemberTable(_CaptureTable):
    def __init__(self, workspace_table: _CreateWorkspaceTable) -> None:
        super().__init__()
        self.workspace_table = workspace_table

    async def create(self, *, data: dict[str, Any]) -> SimpleNamespace:
        row = await super().create(data=data)
        self.workspace_table.workspace_record.members.append(SimpleNamespace(userId=data["userId"], role=data["role"]))
        return row


def _make_workspace_create_db() -> SimpleNamespace:
    workspace_table = _CreateWorkspaceTable()
    return SimpleNamespace(
        workspace=workspace_table,
        workspacemember=_CreateWorkspaceMemberTable(workspace_table),
        workspacetoolselection=_CaptureTable(),
        workspacetoolgroupselection=_CaptureTable(),
        workspacetooloption=_CaptureTable(),
        user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(id="owner-1", role="user"))),
        toolconfig=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(id="tool-read", enabled=True))),
        toolgroup=SimpleNamespace(find_unique=mock.AsyncMock(return_value=None)),
    )


class _WorkspaceUpdateService(UserSpaceService):
    def __init__(
        self,
        role: str,
        *,
        owner_user_id: str = "owner-1",
        tool_options: dict[str, WorkspaceToolOptionState] | None = None,
    ) -> None:
        super().__init__()
        self.role = role
        self.owner_user_id = owner_user_id
        self.tool_options = dict(tool_options) if tool_options is not None else {"tool-read": WorkspaceToolOptionState(write_access_enabled=True)}

    async def _enforce_workspace_access(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str | None = None,
        is_admin: bool = False,
    ) -> UserSpaceWorkspace:
        if required_role == "editor" and self.role not in {"owner", "editor"} and not is_admin:
            raise HTTPException(status_code=403, detail="Editor access required")
        if required_role == "owner" and self.role != "owner" and not is_admin:
            raise HTTPException(status_code=403, detail="Owner access required")
        return UserSpaceWorkspace(
            id=workspace_id,
            name="Workspace",
            owner_user_id=self.owner_user_id,
            members=[] if self.role == "owner" else [WorkspaceMember(user_id=user_id, role=cast(Any, self.role))],
            tool_selection_mode="custom",
            selected_tool_ids=["tool-write", "tool-read"],
            tool_options=dict(self.tool_options),
            created_at=_NOW,
            updated_at=_NOW,
        )


def _make_workspace_update_db() -> SimpleNamespace:
    workspace_table = _FakeWorkspaceTable()
    return SimpleNamespace(
        workspace=workspace_table,
        workspacemember=_WorkspaceMemberTable(),
        workspacetoolselection=_CaptureTable(),
        workspacetoolgroupselection=_CaptureTable(),
        workspacetooloption=_WorkspaceToolOptionTable(workspace_table),
        user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=None)),
    )


@contextmanager
def _patch_workspace_update_dependencies(
    fake_db: SimpleNamespace,
    *,
    tool_configs: list[SimpleNamespace],
    healthy_enabled_tool_ids: list[str],
):
    async def _fake_get_db() -> SimpleNamespace:
        return fake_db

    with ExitStack() as stack:
        stack.enter_context(mock.patch("ragtime.userspace.service.get_db", new=_fake_get_db))
        stack.enter_context(
            mock.patch(
                "ragtime.userspace.service.repository.list_tool_configs",
                mock.AsyncMock(return_value=tool_configs),
            )
        )
        stack.enter_context(
            mock.patch(
                "ragtime.userspace.service.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=healthy_enabled_tool_ids),
            )
        )
        yield


class WorkspaceToolOptionUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_filters_changed_selection_ids_and_groups_for_new_owner(self) -> None:
        service = _WorkspaceUpdateService("owner")
        fake_db = _make_workspace_update_db()
        fake_db.user.find_unique = mock.AsyncMock(return_value=SimpleNamespace(id="owner-2", role="user"))

        with (
            _patch_workspace_update_dependencies(
                fake_db,
                tool_configs=[SimpleNamespace(id="tool-read", enabled=True, allow_write=True)],
                healthy_enabled_tool_ids=["tool-read"],
            ),
            mock.patch.object(
                service,
                "_filter_workspace_selection_inputs_for_owner",
                mock.AsyncMock(return_value=(["tool-read"], ["group-a"])),
            ) as filter_inputs,
            mock.patch.object(
                service,
                "_resolve_workspace_owner_tool_access",
                mock.AsyncMock(return_value={"tool-read": "read_write"}),
            ),
            mock.patch.object(service, "_persist_workspace_tool_selections", mock.AsyncMock()) as persist_tool_selections,
            mock.patch.object(service, "_persist_workspace_tool_group_selections", mock.AsyncMock()) as persist_tool_groups,
        ):
            await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(
                    owner_user_id="owner-2",
                    selected_tool_ids=["tool-read", "tool-write"],
                    selected_tool_group_ids=["group-a", "group-b"],
                ),
                "owner-1",
                is_admin=True,
            )

        filter_inputs.assert_awaited_once_with("owner-2", ["tool-read", "tool-write"], ["group-a", "group-b"])
        persist_tool_selections.assert_awaited_once_with(fake_db, "ws-1", ["tool-read"], replace_existing=True)
        persist_tool_groups.assert_awaited_once_with(fake_db, "ws-1", ["group-a"], replace_existing=True)

    async def test_update_owner_transfer_without_selection_fields_preserves_partial_update_semantics(self) -> None:
        service = _WorkspaceUpdateService("owner")
        fake_db = _make_workspace_update_db()
        fake_db.user.find_unique = mock.AsyncMock(return_value=SimpleNamespace(id="owner-2", role="user"))

        with (
            _patch_workspace_update_dependencies(
                fake_db,
                tool_configs=[SimpleNamespace(id="tool-read", enabled=True, allow_write=True)],
                healthy_enabled_tool_ids=["tool-read"],
            ),
            mock.patch.object(
                service,
                "_filter_workspace_selection_inputs_for_owner",
                mock.AsyncMock(return_value=(["tool-write", "tool-read"], [])),
            ),
            mock.patch.object(
                service,
                "_resolve_workspace_owner_tool_access",
                mock.AsyncMock(return_value={"tool-read": "read_write"}),
            ),
            mock.patch.object(service, "_persist_workspace_tool_selections", mock.AsyncMock()) as persist_tool_selections,
            mock.patch.object(service, "_persist_workspace_tool_group_selections", mock.AsyncMock()) as persist_tool_groups,
        ):
            await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(owner_user_id="owner-2", description="updated"),
                "owner-1",
                is_admin=True,
            )

        persist_tool_selections.assert_not_awaited()
        persist_tool_groups.assert_not_awaited()

    async def test_owner_transfer_removes_write_opt_ins_without_read_write_acl_and_logs_removed_ids(self) -> None:
        service = _WorkspaceUpdateService(
            "owner",
            tool_options={
                "tool-denied": WorkspaceToolOptionState(write_access_enabled=True),
                "tool-read": WorkspaceToolOptionState(write_access_enabled=True),
                "tool-write": WorkspaceToolOptionState(write_access_enabled=True),
            },
        )
        fake_db = _make_workspace_update_db()
        fake_db.user.find_unique = mock.AsyncMock(return_value=SimpleNamespace(id="owner-2", role="user"))
        fake_db.workspace.workspace_record.toolOptions = [
            SimpleNamespace(toolConfigId="tool-denied", options={"write_access_enabled": True}),
            SimpleNamespace(toolConfigId="tool-read", options={"write_access_enabled": True}),
            SimpleNamespace(toolConfigId="tool-write", options={"write_access_enabled": True}),
        ]

        with (
            _patch_workspace_update_dependencies(
                fake_db,
                tool_configs=[
                    SimpleNamespace(id="tool-denied", enabled=True, allow_write=True),
                    SimpleNamespace(id="tool-read", enabled=True, allow_write=True),
                    SimpleNamespace(id="tool-write", enabled=True, allow_write=True),
                ],
                healthy_enabled_tool_ids=["tool-denied", "tool-read", "tool-write"],
            ),
            mock.patch.object(
                service,
                "_filter_workspace_selection_inputs_for_owner",
                mock.AsyncMock(return_value=(["tool-write"], [])),
            ),
            mock.patch.object(
                service,
                "_resolve_workspace_owner_tool_access",
                mock.AsyncMock(return_value={"tool-denied": "deny", "tool-read": "read", "tool-write": "read_write"}),
            ),
            mock.patch("ragtime.userspace.service.logger.info") as log_info,
        ):
            result = await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(owner_user_id="owner-2"),
                "owner-1",
                is_admin=True,
            )

        self.assertEqual(result.tool_options, {"tool-write": WorkspaceToolOptionState(write_access_enabled=True)})
        self.assertEqual(fake_db.workspacetooloption.deleted, [{"workspaceId": "ws-1"}])
        self.assertEqual([created["toolConfigId"] for created in fake_db.workspacetooloption.created], ["tool-write"])
        log_info.assert_called_once()
        self.assertEqual(log_info.call_args.args[1], "ws-1")
        self.assertEqual(set(log_info.call_args.args[2].split(", ")), {"tool-denied", "tool-read"})

    async def test_owner_update_persists_true_rows_only(self) -> None:
        service = _WorkspaceUpdateService("owner")
        fake_db = _make_workspace_update_db()

        with _patch_workspace_update_dependencies(
            fake_db,
            tool_configs=[
                SimpleNamespace(id="tool-write", enabled=True, allow_write=True),
                SimpleNamespace(id="tool-read", enabled=True, allow_write=False),
            ],
            healthy_enabled_tool_ids=["tool-write", "tool-read"],
        ):
            result = await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(
                    tool_options={
                        "tool-write": WorkspaceToolOptionState(write_access_enabled=True),
                        "tool-read": WorkspaceToolOptionState(write_access_enabled=True),
                    }
                ),
                "owner-1",
            )

        self.assertEqual(
            result.tool_options,
            {
                "tool-write": WorkspaceToolOptionState(write_access_enabled=True),
                "tool-read": WorkspaceToolOptionState(write_access_enabled=True),
            },
        )
        self.assertEqual(fake_db.workspacetooloption.deleted, [{"workspaceId": "ws-1"}])
        self.assertEqual(len(fake_db.workspacetooloption.created), 2)
        self.assertEqual(
            {created["toolConfigId"]: created["options"].data for created in fake_db.workspacetooloption.created},
            {
                "tool-write": {"write_access_enabled": True},
                "tool-read": {"write_access_enabled": True},
            },
        )

    async def test_editor_cannot_update_tool_options(self) -> None:
        service = _WorkspaceUpdateService("editor")

        with self.assertRaises(HTTPException) as raised:
            await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(tool_options={"tool-write": WorkspaceToolOptionState(write_access_enabled=True)}),
                "editor-1",
            )

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(raised.exception.detail, "Owner access required")

    async def test_admin_can_update_tool_options(self) -> None:
        service = _WorkspaceUpdateService("editor", owner_user_id="someone-else")
        fake_db = _make_workspace_update_db()

        with _patch_workspace_update_dependencies(
            fake_db,
            tool_configs=[SimpleNamespace(id="tool-write", enabled=True, allow_write=True)],
            healthy_enabled_tool_ids=["tool-write"],
        ):
            await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(tool_options={"tool-write": WorkspaceToolOptionState(write_access_enabled=True)}),
                "admin-1",
                is_admin=True,
            )

        self.assertEqual(len(fake_db.workspacetooloption.created), 1)

    async def test_owner_can_update_tool_options_for_globally_read_only_tool(self) -> None:
        service = _WorkspaceUpdateService("owner")
        fake_db = _make_workspace_update_db()

        with _patch_workspace_update_dependencies(
            fake_db,
            tool_configs=[
                SimpleNamespace(id="tool-write", enabled=True, allow_write=True),
                SimpleNamespace(id="tool-read", enabled=True, allow_write=False),
            ],
            healthy_enabled_tool_ids=["tool-write", "tool-read"],
        ):
            result = await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(
                    description="updated",
                    tool_options={"tool-read": WorkspaceToolOptionState(write_access_enabled=True)},
                ),
                "owner-1",
            )

        self.assertEqual(result.tool_options, {"tool-read": WorkspaceToolOptionState(write_access_enabled=True)})
        self.assertEqual(len(fake_db.workspacetooloption.created), 1)
        created = fake_db.workspacetooloption.created[0]
        self.assertEqual(created["toolConfigId"], "tool-read")

    async def test_owner_can_remove_existing_tool_option_for_globally_read_only_tool(self) -> None:
        service = _WorkspaceUpdateService("owner")
        fake_db = _make_workspace_update_db()

        with _patch_workspace_update_dependencies(
            fake_db,
            tool_configs=[
                SimpleNamespace(id="tool-write", enabled=True, allow_write=True),
                SimpleNamespace(id="tool-read", enabled=True, allow_write=False),
            ],
            healthy_enabled_tool_ids=["tool-write", "tool-read"],
        ):
            result = await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(tool_options={}),
                "owner-1",
            )

        self.assertEqual(result.tool_options, {})
        self.assertEqual(fake_db.workspacetooloption.created, [])

    async def test_owner_can_add_tool_option_for_globally_read_only_tool(self) -> None:
        service = _WorkspaceUpdateService("owner", tool_options={})
        fake_db = _make_workspace_update_db()
        fake_db.workspace.workspace_record.toolOptions = []

        with _patch_workspace_update_dependencies(
            fake_db,
            tool_configs=[
                SimpleNamespace(id="tool-write", enabled=True, allow_write=True),
                SimpleNamespace(id="tool-read", enabled=True, allow_write=False),
            ],
            healthy_enabled_tool_ids=["tool-write", "tool-read"],
        ):
            result = await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(tool_options={"tool-read": WorkspaceToolOptionState(write_access_enabled=True)}),
                "owner-1",
            )

        self.assertEqual(result.tool_options, {"tool-read": WorkspaceToolOptionState(write_access_enabled=True)})

    async def test_omitted_or_false_tool_options_delete_rows(self) -> None:
        service = _WorkspaceUpdateService("owner")
        fake_db = _make_workspace_update_db()

        with _patch_workspace_update_dependencies(
            fake_db,
            tool_configs=[SimpleNamespace(id="tool-write", enabled=True, allow_write=True)],
            healthy_enabled_tool_ids=["tool-write"],
        ):
            await service.update_workspace(
                "ws-1",
                UpdateWorkspaceRequest(tool_options={"tool-write": WorkspaceToolOptionState(write_access_enabled=False)}),
                "owner-1",
            )

        self.assertEqual(fake_db.workspacetooloption.deleted, [{"workspaceId": "ws-1"}])
        self.assertEqual(fake_db.workspacetooloption.created, [])


class WorkspaceToolOptionCreateTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_filters_selected_tool_ids_and_groups_by_owner_acl(self) -> None:
        service = UserSpaceService()
        fake_db = _make_workspace_create_db()
        created_workspace = UserSpaceWorkspace(
            id="ws-create-1",
            name="Workspace",
            description="desc",
            owner_user_id="owner-1",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-read"],
            selected_tool_group_ids=["group-a"],
            created_at=_NOW,
            updated_at=_NOW,
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with (
            mock.patch("ragtime.userspace.service.get_db", new=_fake_get_db),
            mock.patch.object(service, "_is_admin_user", mock.AsyncMock(return_value=False)),
            mock.patch(
                "ragtime.userspace.service.repository.list_tool_configs",
                mock.AsyncMock(
                    return_value=[
                        SimpleNamespace(id="tool-read", enabled=True, allow_write=True),
                        SimpleNamespace(id="tool-write", enabled=True, allow_write=True),
                    ]
                ),
            ),
            mock.patch(
                "ragtime.userspace.service.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-read", "tool-write"]),
            ),
            mock.patch(
                "ragtime.userspace.service.repository.get_tool_ids_for_groups",
                mock.AsyncMock(side_effect=[["tool-read", "tool-write"], ["tool-write"], ["tool-read", "tool-write"]]),
            ),
            mock.patch.object(
                service,
                "filter_tool_ids_for_workspace_owner",
                mock.AsyncMock(side_effect=[["tool-read"], ["tool-read"], []]),
            ),
            mock.patch.object(service, "_persist_workspace_tool_selections", mock.AsyncMock()) as persist_tool_selections,
            mock.patch.object(service, "_persist_workspace_tool_group_selections", mock.AsyncMock()) as persist_tool_groups,
            mock.patch.object(service, "_persist_workspace_tool_options", mock.AsyncMock()) as persist_tool_options,
            mock.patch.object(service, "_workspace_files_dir", return_value=Path(".")),
            mock.patch.object(service, "_ensure_object_storage_config"),
            mock.patch.object(service, "_seed_runtime_bootstrap_config"),
            mock.patch.object(service, "_seed_runtime_entrypoint_config"),
            mock.patch.object(service, "_ensure_workspace_git_repo", mock.AsyncMock()),
            mock.patch.object(service, "_workspace_from_record", return_value=created_workspace),
        ):
            await service.create_workspace(
                CreateWorkspaceRequest(
                    name="Workspace",
                    selected_tool_ids=["tool-read", "tool-write"],
                    selected_tool_group_ids=["group-a", "group-b"],
                    tool_selection_mode="custom",
                ),
                "owner-1",
            )

        persist_tool_selections.assert_awaited_once_with(fake_db, fake_db.workspace.workspace_record.id, ["tool-read"])
        persist_tool_groups.assert_awaited_once_with(fake_db, fake_db.workspace.workspace_record.id, ["group-a"])
        persist_tool_options.assert_awaited_once()

    async def test_owner_can_create_workspace_with_globally_read_only_tool_option(self) -> None:
        service = UserSpaceService()
        fake_db = _make_workspace_create_db()
        created_workspace = UserSpaceWorkspace(
            id="ws-create-1",
            name="Workspace",
            description="desc",
            owner_user_id="owner-1",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-read"],
            tool_options={"tool-read": WorkspaceToolOptionState(write_access_enabled=True)},
            created_at=_NOW,
            updated_at=_NOW,
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with (
            mock.patch("ragtime.userspace.service.get_db", new=_fake_get_db),
            mock.patch.object(service, "_is_admin_user", mock.AsyncMock(return_value=False)),
            mock.patch.object(service, "_filter_workspace_selection_inputs_for_owner", mock.AsyncMock(return_value=(["tool-read"], None))),
            mock.patch(
                "ragtime.userspace.service.repository.list_tool_configs",
                mock.AsyncMock(return_value=[SimpleNamespace(id="tool-read", enabled=True, allow_write=False)]),
            ),
            mock.patch(
                "ragtime.userspace.service.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-read"]),
            ),
            mock.patch.object(service, "_persist_workspace_tool_selections", mock.AsyncMock()) as persist_tool_selections,
            mock.patch.object(service, "_persist_workspace_tool_group_selections", mock.AsyncMock()) as persist_tool_groups,
            mock.patch.object(service, "_persist_workspace_tool_options", mock.AsyncMock()) as persist_tool_options,
            mock.patch.object(service, "_workspace_files_dir", return_value=Path(".")),
            mock.patch.object(service, "_ensure_object_storage_config"),
            mock.patch.object(service, "_seed_runtime_bootstrap_config"),
            mock.patch.object(service, "_seed_runtime_entrypoint_config"),
            mock.patch.object(service, "_ensure_workspace_git_repo", mock.AsyncMock()),
            mock.patch.object(service, "_workspace_from_record", return_value=created_workspace),
        ):
            result = await service.create_workspace(
                CreateWorkspaceRequest(
                    name="Workspace",
                    selected_tool_ids=["tool-read"],
                    tool_selection_mode="custom",
                    tool_options={"tool-read": WorkspaceToolOptionState(write_access_enabled=True)},
                ),
                "owner-1",
            )

        self.assertEqual(result, created_workspace)
        self.assertEqual(len(fake_db.workspace.create_calls), 1)
        self.assertEqual(len(fake_db.workspacemember.created), 1)
        persist_tool_selections.assert_awaited_once()
        persist_tool_groups.assert_awaited_once()
        persist_tool_options.assert_awaited_once_with(
            fake_db,
            fake_db.workspace.workspace_record.id,
            {"tool-read": {"write_access_enabled": True}},
        )

    async def test_admin_can_create_workspace_with_restricted_grant(self) -> None:
        service = UserSpaceService()
        fake_db = _make_workspace_create_db()
        created_workspace = UserSpaceWorkspace(
            id="ws-create-1",
            name="Workspace",
            description="desc",
            owner_user_id="owner-1",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-read"],
            tool_options={"tool-read": WorkspaceToolOptionState(write_access_enabled=True)},
            created_at=_NOW,
            updated_at=_NOW,
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with (
            mock.patch("ragtime.userspace.service.get_db", new=_fake_get_db),
            mock.patch.object(service, "_is_admin_user", mock.AsyncMock(return_value=True)),
            mock.patch.object(service, "_filter_workspace_selection_inputs_for_owner", mock.AsyncMock(return_value=(["tool-read"], None))),
            mock.patch(
                "ragtime.userspace.service.repository.list_tool_configs",
                mock.AsyncMock(return_value=[SimpleNamespace(id="tool-read", enabled=True, allow_write=False)]),
            ),
            mock.patch(
                "ragtime.userspace.service.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-read"]),
            ),
            mock.patch.object(service, "_persist_workspace_tool_selections", mock.AsyncMock()) as persist_tool_selections,
            mock.patch.object(service, "_persist_workspace_tool_group_selections", mock.AsyncMock()) as persist_tool_groups,
            mock.patch.object(service, "_persist_workspace_tool_options", mock.AsyncMock()) as persist_tool_options,
            mock.patch.object(service, "_workspace_files_dir", return_value=Path(".")),
            mock.patch.object(service, "_ensure_object_storage_config"),
            mock.patch.object(service, "_seed_runtime_bootstrap_config"),
            mock.patch.object(service, "_seed_runtime_entrypoint_config"),
            mock.patch.object(service, "_ensure_workspace_git_repo", mock.AsyncMock()),
            mock.patch.object(service, "_workspace_from_record", return_value=created_workspace),
        ):
            result = await service.create_workspace(
                CreateWorkspaceRequest(
                    name="Workspace",
                    selected_tool_ids=["tool-read"],
                    tool_selection_mode="custom",
                    tool_options={"tool-read": WorkspaceToolOptionState(write_access_enabled=True)},
                ),
                "owner-1",
            )

        self.assertEqual(result, created_workspace)
        self.assertEqual(len(fake_db.workspace.create_calls), 1)
        self.assertEqual(len(fake_db.workspacemember.created), 1)
        persist_tool_selections.assert_awaited_once()
        persist_tool_groups.assert_awaited_once()
        persist_tool_options.assert_awaited_once_with(
            fake_db,
            fake_db.workspace.workspace_record.id,
            {"tool-read": {"write_access_enabled": True}},
        )


class WorkspaceToolOptionRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_normalizes_selected_tool_ids_against_enabled_tools_and_passes_admin(self) -> None:
        request = UpdateWorkspaceRequest(
            selected_tool_ids=["tool-write", "tool-unhealthy", "tool-disabled", "tool-write"],
            tool_options={
                "tool-write": WorkspaceToolOptionState(write_access_enabled=True),
                "tool-unhealthy": WorkspaceToolOptionState(write_access_enabled=True),
                "tool-disabled": WorkspaceToolOptionState(write_access_enabled=True),
            },
        )
        fake_user = SimpleNamespace(id="admin-1", role="admin", username="admin")
        fake_workspace = UserSpaceWorkspace(
            id="ws-1",
            name="Workspace",
            owner_user_id="owner-1",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-write"],
            tool_options={"tool-write": WorkspaceToolOptionState(write_access_enabled=True)},
            created_at=_NOW,
            updated_at=_NOW,
        )

        with (
            mock.patch(
                "ragtime.userspace.routes.repository.list_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-write", "tool-unhealthy"]),
            ),
            mock.patch(
                "ragtime.userspace.routes.userspace_service.update_workspace",
                mock.AsyncMock(return_value=fake_workspace),
            ) as update_workspace,
        ):
            result = await update_workspace_route("ws-1", request, fake_user)

        self.assertEqual(result.tool_options, {"tool-write": WorkspaceToolOptionState(write_access_enabled=True)})
        self.assertEqual(request.selected_tool_ids, ["tool-write", "tool-unhealthy"])
        await_args = update_workspace.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertTrue(await_args.kwargs["is_admin"])


class WorkspaceToolOptionDuplicateAndArchiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_with_metadata_copies_tool_options_into_create_request(self) -> None:
        service = UserSpaceService()
        source_workspace = UserSpaceWorkspace(
            id="source-ws",
            name="Source Workspace",
            description="desc",
            owner_user_id="user-1",
            sqlite_persistence_mode="include",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-read"],
            selected_tool_group_ids=["group-1"],
            tool_options={"tool-read": WorkspaceToolOptionState(write_access_enabled=True)},
            created_at=_NOW,
            updated_at=_NOW,
        )
        created_workspace = UserSpaceWorkspace(
            id="created-ws",
            name="Copy",
            owner_user_id="user-1",
            created_at=_NOW,
            updated_at=_NOW,
        )

        async def _fake_enforce(*args: Any, **kwargs: Any) -> UserSpaceWorkspace:
            _ = args, kwargs
            return source_workspace

        async def _fake_create_workspace(request: Any, user_id: str) -> UserSpaceWorkspace:
            _ = user_id
            create_requests.append(request)
            return created_workspace

        create_requests: list[Any] = []

        with (
            mock.patch.object(service, "_enforce_workspace_access", new=_fake_enforce),
            mock.patch("ragtime.userspace.service.repository.get_settings", mock.AsyncMock(return_value=SimpleNamespace())),
            mock.patch(
                "ragtime.userspace.service.repository.list_tool_configs",
                mock.AsyncMock(return_value=[SimpleNamespace(id="tool-read", enabled=True, allow_write=False)]),
            ),
            mock.patch.object(service, "_workspace_duplicate_copy_files_default", return_value=False),
            mock.patch.object(service, "_workspace_duplicate_copy_metadata_default", return_value=True),
            mock.patch.object(service, "_workspace_duplicate_copy_chats_default", return_value=False),
            mock.patch.object(service, "_workspace_duplicate_copy_mounts_default", return_value=False),
            mock.patch.object(service, "_allocate_next_duplicate_workspace_name", mock.AsyncMock(return_value="Copy")),
            mock.patch.object(service, "create_workspace", new=_fake_create_workspace),
            mock.patch.object(service, "upsert_workspace_file", mock.AsyncMock()),
            mock.patch.object(service, "_ensure_object_storage_config"),
            mock.patch.object(service, "_seed_runtime_bootstrap_config"),
            mock.patch.object(service, "_seed_runtime_entrypoint_config"),
            mock.patch.object(service, "_ensure_workspace_git_repo", mock.AsyncMock()),
            mock.patch.object(service, "_copy_workspace_env_vars_for_duplicate", mock.AsyncMock()),
            mock.patch.object(service, "_is_admin_user", mock.AsyncMock(return_value=False)),
            mock.patch("ragtime.userspace.service.repository.create_conversation", mock.AsyncMock(return_value=SimpleNamespace(id="conv-1"))),
            mock.patch.object(service, "_mark_workspace_code_index_dirty", mock.AsyncMock()),
            mock.patch.object(service, "_set_workspace_duplicate_task_phase"),
        ):
            await service._run_workspace_duplicate_task(
                "task-1",
                "source-ws",
                DuplicateWorkspaceRequest(copy_metadata=True, copy_files=False, copy_chats=False, copy_mounts=False),
                "user-1",
                "gpt-test",
            )

        self.assertEqual(len(create_requests), 1)
        self.assertEqual(create_requests[0].tool_options, {"tool-read": WorkspaceToolOptionState(write_access_enabled=True)})

    async def test_archive_export_manifest_serializes_tool_options(self) -> None:
        service = UserSpaceService()
        task_id = "task-1"
        captured_manifest: dict[str, Any] = {}

        async def _fake_run_guarded(
            workspace_id: str,
            user_id: str,
            task_body,
            on_failure,
            **_: object,
        ) -> None:
            _unused = workspace_id, user_id, on_failure
            await task_body(
                SimpleNamespace(
                    name="Workspace",
                    description=None,
                    sqlite_persistence_mode="include",
                    selected_tool_ids=["tool-write"],
                    selected_tool_group_ids=[],
                    tool_options={"tool-write": WorkspaceToolOptionState(write_access_enabled=True)},
                    scm=None,
                )
            )

        def _fake_write_archive(
            source_root: Any,
            archive_path: Any,
            archive_format: str,
            manifest: dict[str, object],
            ignored_prefixes: list[str],
            extra_files: dict[str, Any],
            progress_callback,
        ) -> None:
            _unused = source_root, archive_format, ignored_prefixes, extra_files, progress_callback
            captured_manifest.update(manifest)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(b"archive")

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workspace_files = tmp / "workspace-files"
            workspace_files.mkdir()
            (workspace_files / "notes.txt").write_text("hello\n", encoding="utf-8")

            with (
                mock.patch.object(service, "_run_guarded_workspace_archive_task", new=_fake_run_guarded),
                mock.patch.object(service, "list_workspace_mounts", new=mock.AsyncMock(return_value=[])),
                mock.patch.object(service, "_update_workspace_archive_export_task_phase", new=mock.AsyncMock()),
                mock.patch.object(service, "_serialize_workspace_env_var_placeholders", new=mock.AsyncMock(return_value=[])),
                mock.patch.object(service, "export_workspace_audit_identity_manifest", new=mock.AsyncMock(return_value={})),
                mock.patch.object(service, "_workspace_files_dir", return_value=workspace_files),
                mock.patch.object(service, "_workspace_archive_task_dir", return_value=tmp / task_id),
                mock.patch.object(service, "_write_workspace_archive_sync", new=_fake_write_archive),
            ):
                await service._run_workspace_archive_export_task(
                    task_id,
                    "workspace-1",
                    UserSpaceWorkspaceArchiveExportRequest(
                        archive_format="zip",
                        include_snapshots=False,
                        include_chat_history=False,
                    ),
                    "user-1",
                )

        workspace_manifest = cast(dict[str, Any], captured_manifest["workspace"])
        self.assertEqual(
            workspace_manifest.get("tool_options"),
            {"tool-write": {"write_access_enabled": True}},
        )

    async def test_archive_import_restores_normalized_tool_options(self) -> None:
        service = UserSpaceService()
        manifest = {
            "workspace": {
                "description": "Imported workspace",
                "sqlite_persistence_mode": "include",
                "selected_tool_ids": ["tool-write", "tool-read"],
                "selected_tool_group_ids": [],
                "tool_options": {
                    "tool-write": {"write_access_enabled": True},
                    "tool-read": {"write_access_enabled": True},
                },
            }
        }

        with (
            mock.patch.object(service, "import_workspace_audit_identity_manifest", new=mock.AsyncMock()),
            mock.patch.object(
                service, "_resolve_workspace_archive_selection_id_sets", new=mock.AsyncMock(return_value=({"tool-write", "tool-read"}, set(), []))
            ),
            mock.patch.object(service, "update_workspace", new=mock.AsyncMock()) as update_workspace,
            mock.patch.object(service, "_import_workspace_env_var_placeholders", new=mock.AsyncMock(return_value=(0, 0))),
            mock.patch.object(service, "_import_workspace_mount_placeholders", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(service, "_restore_workspace_archive_scm_metadata", new=mock.AsyncMock()),
            mock.patch(
                "ragtime.userspace.service.repository.list_tool_configs",
                mock.AsyncMock(
                    return_value=[
                        SimpleNamespace(id="tool-write", enabled=True, allow_write=True),
                        SimpleNamespace(id="tool-read", enabled=True, allow_write=False),
                    ]
                ),
            ),
            mock.patch(
                "ragtime.userspace.service.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-write", "tool-read"]),
            ),
        ):
            await service._apply_workspace_archive_manifest(
                "workspace-1",
                "user-1",
                manifest,
                include_snapshots=False,
                include_chat_history=False,
                extract_dir=Path("."),
                is_admin=True,
            )

        update_args = update_workspace.await_args
        assert update_args is not None
        request = update_args.args[1]
        self.assertEqual(
            request.tool_options,
            {
                "tool-write": WorkspaceToolOptionState(write_access_enabled=True),
                "tool-read": WorkspaceToolOptionState(write_access_enabled=True),
            },
        )

    async def test_archive_import_restores_globally_read_only_tool_options_for_non_admin(self) -> None:
        service = UserSpaceService()
        manifest = {
            "workspace": {
                "description": "Imported workspace",
                "sqlite_persistence_mode": "include",
                "selected_tool_ids": ["tool-read"],
                "selected_tool_group_ids": [],
                "tool_options": {
                    "tool-read": {"write_access_enabled": True},
                },
            }
        }

        with (
            mock.patch.object(service, "import_workspace_audit_identity_manifest", new=mock.AsyncMock()),
            mock.patch.object(service, "_resolve_workspace_archive_selection_id_sets", new=mock.AsyncMock(return_value=({"tool-read"}, set(), []))),
            mock.patch.object(service, "update_workspace", new=mock.AsyncMock()) as update_workspace,
            mock.patch.object(service, "_import_workspace_env_var_placeholders", new=mock.AsyncMock(return_value=(0, 0))),
            mock.patch.object(service, "_import_workspace_mount_placeholders", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(service, "_restore_workspace_archive_scm_metadata", new=mock.AsyncMock()),
            mock.patch(
                "ragtime.userspace.service.repository.list_tool_configs",
                mock.AsyncMock(return_value=[SimpleNamespace(id="tool-read", enabled=True, allow_write=False)]),
            ),
            mock.patch(
                "ragtime.userspace.service.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-read"]),
            ),
        ):
            await service._apply_workspace_archive_manifest(
                "workspace-1",
                "user-1",
                manifest,
                include_snapshots=False,
                include_chat_history=False,
                extract_dir=Path("."),
                is_admin=False,
            )

        update_args = update_workspace.await_args
        assert update_args is not None
        request = update_args.args[1]
        self.assertEqual(
            request.tool_options,
            {"tool-read": WorkspaceToolOptionState(write_access_enabled=True)},
        )

    async def test_archive_import_without_tool_options_clears_to_read_only_defaults(self) -> None:
        service = UserSpaceService()
        manifest = {
            "workspace": {
                "description": "Imported workspace",
                "sqlite_persistence_mode": "include",
                "selected_tool_ids": ["tool-write"],
                "selected_tool_group_ids": [],
            }
        }

        with (
            mock.patch.object(service, "import_workspace_audit_identity_manifest", new=mock.AsyncMock()),
            mock.patch.object(service, "_resolve_workspace_archive_selection_id_sets", new=mock.AsyncMock(return_value=({"tool-write"}, set(), []))),
            mock.patch.object(service, "update_workspace", new=mock.AsyncMock()) as update_workspace,
            mock.patch.object(service, "_import_workspace_env_var_placeholders", new=mock.AsyncMock(return_value=(0, 0))),
            mock.patch.object(service, "_import_workspace_mount_placeholders", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(service, "_restore_workspace_archive_scm_metadata", new=mock.AsyncMock()),
            mock.patch(
                "ragtime.userspace.service.repository.list_tool_configs",
                mock.AsyncMock(return_value=[SimpleNamespace(id="tool-write", enabled=True, allow_write=True)]),
            ),
            mock.patch(
                "ragtime.userspace.service.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-write"]),
            ),
        ):
            await service._apply_workspace_archive_manifest(
                "workspace-1",
                "user-1",
                manifest,
                include_snapshots=False,
                include_chat_history=False,
                extract_dir=Path("."),
                is_admin=True,
            )

        update_args = update_workspace.await_args
        assert update_args is not None
        request = update_args.args[1]
        self.assertEqual(request.tool_options, {})


if __name__ == "__main__":
    unittest.main()
