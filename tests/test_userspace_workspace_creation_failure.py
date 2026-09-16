from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.models import CreateWorkspaceRequest  # noqa: E402
from ragtime.userspace.service import UserSpaceService  # noqa: E402


class _WorkspaceTable:
    def __init__(self) -> None:
        self.records: dict[str, SimpleNamespace] = {}
        self.fail_member_create = False
        self.deleted_ids: list[str] = []

    async def create(self, *, data: dict[str, Any], include: dict[str, Any]) -> SimpleNamespace:
        del include
        if any(record.ownerUserId == data["ownerUserId"] and record.nameNormalized == data["nameNormalized"] for record in self.records.values()):
            raise RuntimeError("unique workspace name")
        record = SimpleNamespace(
            **data,
            members=[],
            toolSelections=[],
            toolGroupSelections=[],
            toolOptions=[],
            owner=SimpleNamespace(username="owner", displayName="Owner"),
        )
        self.records[data["id"]] = record
        return record

    async def delete(self, *, where: dict[str, str]) -> None:
        workspace_id = where["id"]
        self.deleted_ids.append(workspace_id)
        self.records.pop(workspace_id, None)

    async def find_unique(self, *, where: dict[str, str], include: dict[str, Any]) -> SimpleNamespace | None:
        del include
        return self.records.get(where["id"])


class _WorkspaceMemberTable:
    def __init__(self, workspace: _WorkspaceTable) -> None:
        self.workspace = workspace

    async def create(self, *, data: dict[str, Any]) -> None:
        if self.workspace.fail_member_create:
            raise RuntimeError("member insert failed")
        self.workspace.records[data["workspaceId"]].members.append(SimpleNamespace(userId=data["userId"], role=data["role"]))


class _TransactionalWorkspaceDb:
    def __init__(self) -> None:
        self.workspace = _WorkspaceTable()
        self.workspacemember = _WorkspaceMemberTable(self.workspace)
        self.workspacetoolselection = SimpleNamespace(create=mock.AsyncMock())
        self.workspacetoolgroupselection = SimpleNamespace(create=mock.AsyncMock())
        self.workspacetooloption = SimpleNamespace(create=mock.AsyncMock())
        self.toolconfig = SimpleNamespace(find_unique=mock.AsyncMock(return_value=None))
        self.toolgroup = SimpleNamespace(find_unique=mock.AsyncMock(return_value=None))

    @asynccontextmanager
    async def tx(self) -> AsyncIterator[_TransactionalWorkspaceDb]:
        records_before = dict(self.workspace.records)
        try:
            yield self
        except BaseException:
            self.workspace.records = records_before
            raise


class WorkspaceCreationFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.service = UserSpaceService()
        self.service._workspaces_dir = Path(self.temp_dir.name) / "workspaces"
        self.service._workspaces_dir.mkdir()
        self.db = _TransactionalWorkspaceDb()
        self.request = CreateWorkspaceRequest(
            name="Retry workspace",
            selected_tool_ids=[],
            selected_tool_group_ids=[],
            tool_selection_mode="custom",
        )

    def _patches(self, ensure_workspace: Any, delete_workspace: Any) -> Any:
        async def get_db() -> _TransactionalWorkspaceDb:
            return self.db

        return mock.patch.multiple(
            "ragtime.userspace.service",
            get_db=get_db,
            object_storage_control=SimpleNamespace(
                ensure_workspace=ensure_workspace,
                delete_workspace=delete_workspace,
            ),
        )

    async def _create(self, ensure_workspace: Any, delete_workspace: Any) -> Any:
        with (
            self._patches(ensure_workspace, delete_workspace),
            mock.patch.object(self.service, "_is_admin_user", mock.AsyncMock(return_value=False)),
            mock.patch.object(self.service, "_filter_workspace_selection_inputs_for_owner", mock.AsyncMock(return_value=([], []))),
            mock.patch.object(self.service, "_normalize_workspace_tool_options_for_request", mock.AsyncMock(return_value={})),
            mock.patch.object(self.service, "_ensure_workspace_git_repo", mock.AsyncMock()),
        ):
            return await self.service.create_workspace(self.request, "owner-1")

    async def test_storage_403_cleans_exact_workspace_and_retry_seeds_runtime_files(self) -> None:
        unrelated = SimpleNamespace(id="unrelated", ownerUserId="other", nameNormalized="other")
        self.db.workspace.records[unrelated.id] = unrelated
        cleanup_storage = mock.AsyncMock()

        with self.assertRaises(HTTPException) as raised:
            await self._create(mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="denied")), cleanup_storage)

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(self.db.workspace.records, {"unrelated": unrelated})
        self.assertEqual(list(self.service._workspaces_dir.iterdir()), [])
        cleanup_storage.assert_awaited_once()

        result = await self._create(mock.AsyncMock(return_value={}), mock.AsyncMock())
        workspace_dir = self.service._workspace_dir(result.id)
        self.assertTrue((workspace_dir / "files" / ".ragtime" / "runtime-bootstrap.json").is_file())
        self.assertTrue((workspace_dir / "files" / ".ragtime" / "runtime-entrypoint.json").is_file())

    async def test_child_database_failure_rolls_back_before_external_provisioning(self) -> None:
        self.db.workspace.fail_member_create = True
        ensure_workspace = mock.AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "member insert failed"):
            await self._create(ensure_workspace, mock.AsyncMock())

        self.assertEqual(self.db.workspace.records, {})
        ensure_workspace.assert_not_awaited()

    async def test_cancellation_after_relational_creation_cleans_and_reraises(self) -> None:
        cleanup_storage = mock.AsyncMock()

        with self.assertRaises(asyncio.CancelledError):
            await self._create(mock.AsyncMock(side_effect=asyncio.CancelledError()), cleanup_storage)

        self.assertEqual(self.db.workspace.records, {})
        cleanup_storage.assert_awaited_once()

    async def test_git_failure_after_storage_cleans_workspace_and_storage(self) -> None:
        cleanup_storage = mock.AsyncMock()

        with (
            self._patches(mock.AsyncMock(return_value={}), cleanup_storage),
            mock.patch.object(self.service, "_is_admin_user", mock.AsyncMock(return_value=False)),
            mock.patch.object(self.service, "_filter_workspace_selection_inputs_for_owner", mock.AsyncMock(return_value=([], []))),
            mock.patch.object(self.service, "_normalize_workspace_tool_options_for_request", mock.AsyncMock(return_value={})),
            mock.patch.object(self.service, "_ensure_workspace_git_repo", mock.AsyncMock(side_effect=RuntimeError("git failed"))),
        ):
            with self.assertRaisesRegex(RuntimeError, "git failed"):
                await self.service.create_workspace(self.request, "owner-1")

        self.assertEqual(self.db.workspace.records, {})
        cleanup_storage.assert_awaited_once()
        self.assertEqual(list(self.service._workspaces_dir.iterdir()), [])

    async def test_cleanup_failure_does_not_mask_storage_failure(self) -> None:
        cleanup_storage = mock.AsyncMock(side_effect=RuntimeError("cleanup failed"))

        with self.assertRaises(HTTPException) as raised:
            await self._create(mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="denied")), cleanup_storage)

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(self.db.workspace.records, {})
