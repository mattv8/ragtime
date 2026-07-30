"""Tests for externally submitted build briefs and task lifecycle."""

import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TypedDict, Unpack
from unittest import mock

from fastapi import HTTPException

from ragtime.indexer.tool_selection import resolve_effective_tool_ids
from ragtime.userspace.agent_briefs import (
    BuildBriefInput,
    compute_brief_payload_hash,
    render_build_brief,
)


class _BriefOverrides(TypedDict, total=False):
    idempotency_key: str
    title: str
    objective: str
    requirements: list[str]
    acceptance_criteria: list[str]
    constraints: list[str]
    preserve_paths: list[str]
    data_component_ids: list[str]
    non_goals: list[str]
    context_revision: str | None


def _brief(**overrides: Unpack[_BriefOverrides]) -> BuildBriefInput:
    base: _BriefOverrides = {
        "idempotency_key": "cowork-req-001",
        "title": "Add revenue chart",
        "objective": "Add a monthly revenue line chart to the dashboard.",
        "requirements": ["Chart shows last 12 months"],
        "acceptance_criteria": ["Chart renders with live data"],
        "preserve_paths": [],
        "data_component_ids": [],
    }
    base.update(overrides)
    return BuildBriefInput(**base)


class BriefRenderingTests(unittest.TestCase):
    def test_render_contains_sections_and_autonomy_note(self) -> None:
        text = render_build_brief(_brief(non_goals=["No auth changes"]), workspace_name="Sales")
        self.assertIn("## Objective", text)
        self.assertIn("## Requirements", text)
        self.assertIn("## Acceptance Criteria", text)
        self.assertIn("## Non-Goals", text)
        self.assertIn("Work autonomously", text)
        self.assertIn('"Sales"', text)

    def test_payload_hash_ignores_idempotency_key_only(self) -> None:
        a = compute_brief_payload_hash(_brief())
        b = compute_brief_payload_hash(_brief(idempotency_key="different-key"))
        c = compute_brief_payload_hash(_brief(title="Other title"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class StartBuildTaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        from ragtime.userspace import build_task_service as module

        self.module = module
        self.service = module.build_task_service

    def _base_patches(self):
        workspace = SimpleNamespace(
            id="ws-1",
            name="Sales",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-1"],
            selected_tool_group_ids=[],
        )
        return [
            mock.patch.object(
                type(self.service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(
                type(self.service),
                "_enforce_editor",
                mock.AsyncMock(return_value=workspace),
            ),
        ]

    async def test_duplicate_key_same_payload_returns_original(self) -> None:
        brief = _brief()
        existing = SimpleNamespace(
            payloadHash=compute_brief_payload_hash(brief),
            workspaceId="ws-1",
            conversationId="conv-1",
            taskId="task-1",
        )
        patches = self._base_patches() + [
            mock.patch.object(type(self.service), "_validate_brief", mock.AsyncMock()),
            mock.patch.object(
                self.module,
                "find_external_build_request",
                mock.AsyncMock(return_value=existing),
            ),
            mock.patch.object(
                self.module.repository,
                "get_chat_task",
                mock.AsyncMock(return_value=SimpleNamespace(status="completed")),
            ),
        ]
        for p in patches:
            p.start()
        try:
            result = await self.service.start_build_task("ws-1", "user-1", brief)
        finally:
            for p in patches:
                p.stop()
        self.assertTrue(result["deduplicated"])
        self.assertEqual(result["conversation_id"], "conv-1")
        self.assertEqual(result["task_id"], "task-1")

    async def test_duplicate_key_different_payload_conflicts(self) -> None:
        existing = SimpleNamespace(
            payloadHash="other-hash",
            workspaceId="ws-1",
            conversationId="conv-1",
            taskId="task-1",
        )
        patches = self._base_patches() + [
            mock.patch.object(type(self.service), "_validate_brief", mock.AsyncMock()),
            mock.patch.object(
                self.module,
                "find_external_build_request",
                mock.AsyncMock(return_value=existing),
            ),
        ]
        for p in patches:
            p.start()
        try:
            with self.assertRaises(HTTPException) as ctx:
                await self.service.start_build_task("ws-1", "user-1", _brief())
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_duplicate_key_for_other_workspace_conflicts_without_leaking_ids(self) -> None:
        brief = _brief()
        existing = SimpleNamespace(
            payloadHash=compute_brief_payload_hash(brief),
            workspaceId="ws-2",
            conversationId="conv-secret",
            taskId="task-secret",
        )
        patches = self._base_patches() + [
            mock.patch.object(type(self.service), "_validate_brief", mock.AsyncMock()),
            mock.patch.object(
                self.module,
                "find_external_build_request",
                mock.AsyncMock(return_value=existing),
            ),
        ]
        for p in patches:
            p.start()
        try:
            with self.assertRaises(HTTPException) as ctx:
                await self.service.start_build_task("ws-1", "user-1", brief)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertNotIn("conv-secret", str(ctx.exception.detail))
        self.assertNotIn("task-secret", str(ctx.exception.detail))

    async def test_unique_insert_race_for_other_workspace_conflicts_without_leaking_ids(self) -> None:
        brief = _brief()
        raced = SimpleNamespace(
            payloadHash=compute_brief_payload_hash(brief),
            workspaceId="ws-2",
            conversationId="conv-secret",
            taskId="task-secret",
        )
        patches = self._base_patches() + [
            mock.patch.object(type(self.service), "_validate_brief", mock.AsyncMock()),
            mock.patch.object(
                self.module,
                "find_external_build_request",
                mock.AsyncMock(side_effect=[None, raced]),
            ),
            mock.patch.object(self.module, "_is_unique_violation", return_value=True),
            mock.patch.object(
                self.module,
                "create_external_build_request",
                mock.AsyncMock(side_effect=RuntimeError("race")),
            ),
        ]
        for p in patches:
            p.start()
        try:
            with self.assertRaises(HTTPException) as ctx:
                await self.service.start_build_task("ws-1", "user-1", brief)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertNotIn("conv-secret", str(ctx.exception.detail))
        self.assertNotIn("task-secret", str(ctx.exception.detail))

    async def test_missing_preserve_path_rejected(self) -> None:
        from ragtime.userspace.service import userspace_service

        patches = self._base_patches() + [
            mock.patch.object(
                userspace_service,
                "list_workspace_files",
                mock.AsyncMock(return_value=[SimpleNamespace(path="index.html")]),
            ),
        ]
        for p in patches:
            p.start()
        try:
            with self.assertRaises(HTTPException) as ctx:
                await self.service.start_build_task("ws-1", "user-1", _brief(preserve_paths=["missing.ts"]))
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("missing.ts", str(ctx.exception.detail))

    async def test_data_component_ids_respect_workspace_owner_acl(self) -> None:
        from ragtime.userspace.service import userspace_service

        workspace = SimpleNamespace(
            id="ws-1",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-1", "tool-2"],
            selected_tool_group_ids=[],
        )

        with (
            mock.patch.object(
                self.module,
                "resolve_effective_tool_ids",
                mock.AsyncMock(return_value=["tool-1", "tool-2"]),
            ),
            mock.patch.object(
                userspace_service,
                "filter_tool_ids_for_workspace_owner",
                mock.AsyncMock(return_value=["tool-1"]),
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await self.service._validate_brief(workspace, _brief(data_component_ids=["tool-2"]), SimpleNamespace(id="user-1", role="user"))

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("tool-2", str(ctx.exception.detail))

    async def test_data_component_ids_accept_enabled_unhealthy_custom_tools(self) -> None:
        from ragtime.userspace.service import userspace_service

        workspace = SimpleNamespace(
            id="ws-1",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-3"],
            selected_tool_group_ids=[],
        )

        with (
            mock.patch.object(
                self.module,
                "resolve_effective_tool_ids",
                mock.AsyncMock(side_effect=resolve_effective_tool_ids),
            ),
            mock.patch.object(self.module.repository, "list_healthy_enabled_tool_ids", mock.AsyncMock(return_value=["tool-1"])),
            mock.patch.object(self.module.repository, "list_enabled_tool_ids", mock.AsyncMock(return_value=["tool-1", "tool-3"])),
            mock.patch.object(self.module.repository, "get_tool_ids_for_groups", mock.AsyncMock(return_value=[])),
            mock.patch.object(userspace_service, "filter_tool_ids_for_workspace_owner", mock.AsyncMock(return_value=["tool-3"])),
        ):
            await self.service._validate_brief(workspace, _brief(data_component_ids=["tool-3"]), SimpleNamespace(id="user-1", role="user"))

    async def test_builder_start_failure_releases_idempotency_claim(self) -> None:
        from ragtime.indexer import routes as indexer_routes

        conversation = SimpleNamespace(id="conv-1")
        ledger = SimpleNamespace(id="ebr-1")
        delete_claim = mock.AsyncMock()
        delete_conversation = mock.AsyncMock(return_value=True)
        patches = self._base_patches() + [
            mock.patch.object(type(self.service), "_validate_brief", mock.AsyncMock()),
            mock.patch.object(self.module, "find_external_build_request", mock.AsyncMock(return_value=None)),
            mock.patch.object(self.module, "create_external_build_request", mock.AsyncMock(return_value=ledger)),
            mock.patch.object(self.module, "delete_external_build_request", delete_claim),
            mock.patch.object(self.module.repository, "create_conversation", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(self.module.repository, "delete_conversation", delete_conversation),
            mock.patch.object(
                indexer_routes,
                "_send_background_message_to_loaded_conversation",
                mock.AsyncMock(side_effect=RuntimeError("provider unavailable")),
            ),
        ]
        for p in patches:
            p.start()
        try:
            with self.assertRaises(RuntimeError):
                await self.service.start_build_task("ws-1", "user-1", _brief())
            delete_conversation.assert_awaited_once_with("conv-1")
            delete_claim.assert_awaited_once_with("ebr-1")
        finally:
            for p in patches:
                p.stop()

    async def test_new_brief_starts_builder_and_finalizes_ledger(self) -> None:
        from ragtime.indexer import routes as indexer_routes

        conversation = SimpleNamespace(id="conv-1")
        ledger = SimpleNamespace(id="ebr-1")
        task = SimpleNamespace(id="task-1", status="pending")
        bind = mock.AsyncMock()
        finalize = mock.AsyncMock()
        patches = self._base_patches() + [
            mock.patch.object(type(self.service), "_validate_brief", mock.AsyncMock()),
            mock.patch.object(self.module, "find_external_build_request", mock.AsyncMock(return_value=None)),
            mock.patch.object(self.module, "create_external_build_request", mock.AsyncMock(return_value=ledger)),
            mock.patch.object(self.module, "bind_external_build_request_conversation", bind),
            mock.patch.object(self.module, "finalize_external_build_request", finalize),
            mock.patch.object(self.module.repository, "create_conversation", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(
                indexer_routes,
                "_send_background_message_to_loaded_conversation",
                mock.AsyncMock(return_value={"task": task}),
            ),
        ]
        for p in patches:
            p.start()
        try:
            result = await self.service.start_build_task("ws-1", "user-1", _brief())
            bind.assert_awaited_once_with("ebr-1", conversation_id="conv-1")
            finalize.assert_awaited_once_with("ebr-1", conversation_id="conv-1", task_id="task-1")
        finally:
            for p in patches:
                p.stop()

        self.assertFalse(result["deduplicated"])
        self.assertEqual(result["task_id"], "task-1")


class GetBuildTaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        from ragtime.userspace import build_task_service as module

        self.module = module
        self.service = module.build_task_service

    def _task(self, **overrides):
        base = dict(
            id="task-1",
            conversation_id="conv-1",
            status="completed",
            response_content="R" * 5000,
            error_message=None,
            created_at=datetime.now(timezone.utc),
            started_at=None,
            completed_at=None,
            last_update_at=datetime.now(timezone.utc) - timedelta(seconds=600),
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    async def test_result_truncation_and_workspace_scoping(self) -> None:
        conversation_mock = mock.AsyncMock(return_value=SimpleNamespace(workspace_id="ws-1"))
        patches = [
            mock.patch.object(
                type(self.service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(
                self.module.repository,
                "get_chat_task",
                mock.AsyncMock(return_value=self._task()),
            ),
            mock.patch.object(
                self.module.repository,
                "get_conversation",
                conversation_mock,
            ),
            mock.patch.object(
                self.module,
                "find_external_build_request_by_conversation",
                mock.AsyncMock(return_value=SimpleNamespace(id="ebr-1")),
            ),
            mock.patch.object(
                self.module.repository,
                "check_conversation_access",
                mock.AsyncMock(return_value=True),
            ),
        ]
        for p in patches:
            p.start()
        try:
            result = await self.service.get_build_task("ws-1", "user-1", "task-1", max_result_chars=1000)
            self.assertTrue(result["result_truncated"])
            self.assertEqual(len(result["result"]), 1000)

            conversation_mock.return_value = SimpleNamespace(workspace_id="other-ws")
            with self.assertRaises(HTTPException) as ctx:
                await self.service.get_build_task("ws-1", "user-1", "task-1")
            self.assertEqual(ctx.exception.status_code, 404)
        finally:
            for p in patches:
                p.stop()

    async def test_running_task_reports_possible_stall(self) -> None:
        task = self._task(status="running", response_content=None)
        patches = [
            mock.patch.object(
                type(self.service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(self.module.repository, "get_chat_task", mock.AsyncMock(return_value=task)),
            mock.patch.object(
                self.module.repository,
                "get_conversation",
                mock.AsyncMock(return_value=SimpleNamespace(workspace_id="ws-1")),
            ),
            mock.patch.object(
                self.module,
                "find_external_build_request_by_conversation",
                mock.AsyncMock(return_value=SimpleNamespace(id="ebr-1")),
            ),
            mock.patch.object(
                self.module.repository,
                "check_conversation_access",
                mock.AsyncMock(return_value=True),
            ),
        ]
        for p in patches:
            p.start()
        try:
            result = await self.service.get_build_task("ws-1", "user-1", "task-1")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(result["status"], "running")
        self.assertTrue(result["possibly_stalled"])
        self.assertIsNone(result["result"])

    async def test_task_not_owned_by_external_ledger_returns_404(self) -> None:
        with (
            mock.patch.object(
                type(self.service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(
                self.module.repository,
                "get_chat_task",
                mock.AsyncMock(return_value=self._task()),
            ),
            mock.patch.object(
                self.module.repository,
                "get_conversation",
                mock.AsyncMock(return_value=SimpleNamespace(workspace_id="ws-1")),
            ),
            mock.patch.object(
                self.module,
                "find_external_build_request_by_conversation",
                mock.AsyncMock(return_value=None),
            ),
            mock.patch.object(
                self.module.repository,
                "check_conversation_access",
                mock.AsyncMock(return_value=True),
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await self.service.get_build_task("ws-1", "user-1", "unrelated-task")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_older_task_in_owned_conversation_still_polls(self) -> None:
        task = self._task(id="task-older")
        with (
            mock.patch.object(
                type(self.service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(
                self.module.repository,
                "get_chat_task",
                mock.AsyncMock(return_value=task),
            ),
            mock.patch.object(
                self.module.repository,
                "get_conversation",
                mock.AsyncMock(return_value=SimpleNamespace(workspace_id="ws-1")),
            ),
            mock.patch.object(
                self.module,
                "find_external_build_request_by_conversation",
                mock.AsyncMock(return_value=SimpleNamespace(id="ebr-1", taskId="task-newer")),
            ),
            mock.patch.object(
                self.module.repository,
                "check_conversation_access",
                mock.AsyncMock(return_value=True),
            ),
        ):
            result = await self.service.get_build_task("ws-1", "user-1", "task-older")

        self.assertEqual(result["task_id"], "task-older")


class ReplyBuildTaskTests(unittest.IsolatedAsyncioTestCase):
    @contextmanager
    def _reply_patches(
        self,
        service,
        module,
        *,
        task,
        conversation,
        ledger_task_id="task-1",
        existing=None,
        access=True,
        task_side_effect=None,
        send=None,
    ):
        from ragtime.indexer import routes as indexer_routes

        get_task = mock.AsyncMock(
            side_effect=task_side_effect if task_side_effect is not None else None,
            return_value=task if task_side_effect is None else None,
        )
        patches = [
            mock.patch.object(
                type(service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(type(service), "_enforce_editor", mock.AsyncMock()),
            mock.patch.object(module.repository, "get_chat_task", get_task),
            mock.patch.object(module.repository, "get_conversation", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(
                module,
                "find_external_build_request_by_conversation",
                mock.AsyncMock(return_value=SimpleNamespace(id="ebr-1", taskId=ledger_task_id)),
            ),
            mock.patch.object(module, "find_external_build_request", mock.AsyncMock(return_value=existing)),
            mock.patch.object(module.repository, "check_conversation_access", mock.AsyncMock(return_value=access)),
        ]
        if send is not None:
            patches.append(mock.patch.object(indexer_routes, "_send_background_message_to_loaded_conversation", send))
        for patcher in patches:
            patcher.start()
        try:
            yield
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    async def test_reply_rejects_running_task(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        with self._reply_patches(
            service,
            module,
            task=SimpleNamespace(id="task-1", conversation_id="conv-1", status="running"),
            conversation=SimpleNamespace(workspace_id="ws-1", parent_conversation_id=None),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_reply_advances_ledger_to_new_task_and_finalizes_reply_claim(self) -> None:
        from ragtime.indexer import routes as indexer_routes
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        ledger = SimpleNamespace(id="ebr-1", taskId="task-1", conversationId="conv-1")
        task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        reply_ledger = SimpleNamespace(id="reply-ebr-1")
        with (
            mock.patch.object(
                type(service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(type(service), "_enforce_editor", mock.AsyncMock()),
            mock.patch.object(module, "find_external_build_request_by_conversation", mock.AsyncMock(return_value=ledger)),
            mock.patch.object(module, "find_external_build_request", mock.AsyncMock(return_value=None)),
            mock.patch.object(module, "create_external_build_request", mock.AsyncMock(return_value=reply_ledger)),
            mock.patch.object(module.repository, "get_chat_task", mock.AsyncMock(return_value=task)),
            mock.patch.object(module.repository, "get_conversation", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(module.repository, "check_conversation_access", mock.AsyncMock(return_value=True)),
            mock.patch.object(
                indexer_routes,
                "_send_background_message_to_loaded_conversation",
                mock.AsyncMock(return_value={"task": SimpleNamespace(id="task-2", status="pending")}),
            ),
            mock.patch.object(module, "finalize_external_build_request", mock.AsyncMock()) as finalize,
        ):
            result = await service.reply_to_build_task(
                "ws-1",
                "user-1",
                "task-1",
                "Use the existing chart library",
                "reply-001",
            )

        self.assertEqual(result["task_id"], "task-2")
        self.assertEqual(
            finalize.await_args_list,
            [
                mock.call("reply-ebr-1", conversation_id="conv-1", task_id="task-2"),
                mock.call("ebr-1", conversation_id="conv-1", task_id="task-2"),
            ],
        )

    async def test_reply_to_older_task_returns_current_ledger_task_without_enqueuing(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        older_task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        current_task = SimpleNamespace(id="task-2", conversation_id="conv-1", status="pending")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        send = mock.AsyncMock()
        with self._reply_patches(
            service,
            module,
            task=current_task,
            conversation=conversation,
            ledger_task_id="task-2",
            task_side_effect=[older_task, current_task],
            send=send,
        ):
            result = await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")

        self.assertEqual(result["task_id"], "task-2")
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["deduplicated"])
        send.assert_not_awaited()

    async def test_reply_rejects_stale_task_when_current_pointer_task_missing(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        requested_task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        send = mock.AsyncMock()
        with self._reply_patches(
            service,
            module,
            task=None,
            conversation=conversation,
            ledger_task_id="task-2",
            task_side_effect=[requested_task, None],
            send=send,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")

        self.assertEqual(ctx.exception.status_code, 409)
        send.assert_not_awaited()

    async def test_reply_rejects_when_conversation_access_denied(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        with self._reply_patches(
            service,
            module,
            task=task,
            conversation=conversation,
            access=False,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")

        self.assertEqual(ctx.exception.status_code, 404)

    async def test_reply_duplicate_key_same_payload_returns_existing_task_without_sending(self) -> None:
        from ragtime.indexer import routes as indexer_routes
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        reply_task = SimpleNamespace(id="task-2", status="pending")
        existing = SimpleNamespace(
            id="reply-ebr-1",
            workspaceId="ws-1",
            payloadHash=module._compute_reply_payload_hash("ws-1", "task-1", "Proceed"),
            conversationId="conv-1",
            taskId="task-2",
        )
        send = mock.AsyncMock()
        with self._reply_patches(
            service,
            module,
            task=reply_task,
            conversation=conversation,
            existing=existing,
            task_side_effect=[task, reply_task],
            send=send,
        ):
            result = await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")

        self.assertTrue(result["deduplicated"])
        self.assertEqual(result["task_id"], "task-2")
        send.assert_not_awaited()

    async def test_reply_retry_prefers_finalized_reply_ledger_over_advanced_original_pointer(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        requested_task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        followup_task = SimpleNamespace(id="task-2", conversation_id="conv-1", status="pending")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        existing_reply = SimpleNamespace(
            id="reply-ebr-1",
            workspaceId="ws-1",
            payloadHash=module._compute_reply_payload_hash("ws-1", "task-1", "Proceed"),
            conversationId="conv-1",
            taskId="task-2",
        )
        send = mock.AsyncMock()
        with self._reply_patches(
            service,
            module,
            task=followup_task,
            conversation=conversation,
            ledger_task_id="task-2",
            existing=existing_reply,
            task_side_effect=[requested_task, followup_task],
            send=send,
        ):
            result = await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")

        self.assertEqual(
            result,
            {
                "deduplicated": True,
                "conversation_id": "conv-1",
                "task_id": "task-2",
                "status": "pending",
            },
        )
        send.assert_not_awaited()

    async def test_reply_duplicate_key_different_payload_conflicts(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        existing = SimpleNamespace(
            id="reply-ebr-1",
            workspaceId="ws-1",
            payloadHash="other-hash",
            conversationId="conv-1",
            taskId="task-2",
        )
        with (
            mock.patch.object(
                type(service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(type(service), "_enforce_editor", mock.AsyncMock()),
            mock.patch.object(module.repository, "get_chat_task", mock.AsyncMock(return_value=task)),
            mock.patch.object(module.repository, "get_conversation", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(module, "find_external_build_request_by_conversation", mock.AsyncMock(return_value=SimpleNamespace(id="ebr-1", taskId="task-1"))),
            mock.patch.object(module, "find_external_build_request", mock.AsyncMock(return_value=existing)),
            mock.patch.object(module.repository, "check_conversation_access", mock.AsyncMock(return_value=True)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")

        self.assertEqual(ctx.exception.status_code, 409)

    async def test_reply_duplicate_key_for_other_workspace_conflicts(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        existing = SimpleNamespace(
            id="reply-ebr-1",
            workspaceId="ws-2",
            payloadHash=module._compute_reply_payload_hash("ws-1", "task-1", "Proceed"),
            conversationId="conv-secret",
            taskId="task-secret",
        )
        with (
            mock.patch.object(
                type(service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(type(service), "_enforce_editor", mock.AsyncMock()),
            mock.patch.object(module.repository, "get_chat_task", mock.AsyncMock(return_value=task)),
            mock.patch.object(module.repository, "get_conversation", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(module, "find_external_build_request_by_conversation", mock.AsyncMock(return_value=SimpleNamespace(id="ebr-1", taskId="task-1"))),
            mock.patch.object(module, "find_external_build_request", mock.AsyncMock(return_value=existing)),
            mock.patch.object(module.repository, "check_conversation_access", mock.AsyncMock(return_value=True)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")

        self.assertEqual(ctx.exception.status_code, 409)

    async def test_reply_duplicate_key_unfinished_claim_conflicts(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        task = SimpleNamespace(id="task-1", conversation_id="conv-1", status="completed")
        conversation = SimpleNamespace(id="conv-1", workspace_id="ws-1", parent_conversation_id=None)
        existing = SimpleNamespace(
            id="reply-ebr-1",
            workspaceId="ws-1",
            payloadHash=module._compute_reply_payload_hash("ws-1", "task-1", "Proceed"),
            conversationId="conv-1",
            taskId=None,
        )
        with (
            mock.patch.object(
                type(service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(type(service), "_enforce_editor", mock.AsyncMock()),
            mock.patch.object(module.repository, "get_chat_task", mock.AsyncMock(return_value=task)),
            mock.patch.object(module.repository, "get_conversation", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(module, "find_external_build_request_by_conversation", mock.AsyncMock(return_value=SimpleNamespace(id="ebr-1", taskId="task-1"))),
            mock.patch.object(module, "find_external_build_request", mock.AsyncMock(return_value=existing)),
            mock.patch.object(module.repository, "check_conversation_access", mock.AsyncMock(return_value=True)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await service.reply_to_build_task("ws-1", "user-1", "task-1", "Proceed", "reply-001")

        self.assertEqual(ctx.exception.status_code, 409)

    async def test_retry_recovers_latest_task_when_conversation_bound_but_task_pointer_missing(self) -> None:
        from ragtime.userspace import build_task_service as module

        service = module.build_task_service
        brief = _brief()
        existing = SimpleNamespace(
            id="ebr-1",
            payloadHash=compute_brief_payload_hash(brief),
            workspaceId="ws-1",
            conversationId="conv-1",
            taskId=None,
        )
        db = SimpleNamespace(chattask=SimpleNamespace(find_first=mock.AsyncMock(return_value=SimpleNamespace(id="task-2", status="completed"))))
        finalize = mock.AsyncMock()
        workspace = SimpleNamespace(
            id="ws-1",
            name="Sales",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-1"],
            selected_tool_group_ids=[],
        )
        patches = [
            mock.patch.object(
                type(service),
                "_load_prisma_user",
                mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user")),
            ),
            mock.patch.object(
                type(service),
                "_enforce_editor",
                mock.AsyncMock(return_value=workspace),
            ),
            mock.patch.object(type(service), "_validate_brief", mock.AsyncMock()),
            mock.patch.object(module, "find_external_build_request", mock.AsyncMock(return_value=existing)),
            mock.patch.object(module, "finalize_external_build_request", finalize),
            mock.patch.object(module, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(
                module.repository,
                "get_chat_task",
                mock.AsyncMock(return_value=SimpleNamespace(id="task-2", status="completed")),
            ),
        ]
        for p in patches:
            p.start()
        try:
            result = await service.start_build_task("ws-1", "user-1", brief)
        finally:
            for p in patches:
                p.stop()

        self.assertTrue(result["deduplicated"])
        self.assertEqual(result["task_id"], "task-2")
        finalize.assert_awaited_once_with("ebr-1", conversation_id="conv-1", task_id="task-2")


if __name__ == "__main__":
    unittest.main()
