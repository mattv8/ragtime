import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TypedDict
from unittest import mock

from ragtime.indexer.models import ConversationBranchKind, WorkspaceChatStateResponse
from ragtime.indexer.repository import repository
from ragtime.userspace.models import UserSpaceRuntimeStatusResponse
from ragtime.userspace.runtime_service import userspace_runtime_service

NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)


class _AccessKwargsRequired(TypedDict):
    conversation_id: str
    user_id: str | None


class _AccessKwargs(_AccessKwargsRequired, total=False):
    is_admin: bool
    workspace_id: str | None


class _AccessCase(TypedDict):
    name: str
    side_effect: list[list[dict[str, object]]]
    kwargs: _AccessKwargs
    expected: bool
    query_count: int


class ConversationAccessPerfRefactorTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_conversation_access_semantics(self) -> None:
        cases: list[_AccessCase] = [
            {
                "name": "not-found",
                "side_effect": [[]],
                "kwargs": {"conversation_id": "conv-1", "user_id": "user-1"},
                "expected": False,
                "query_count": 1,
            },
            {
                "name": "workspace param mismatch",
                "side_effect": [[{"user_id": "owner-1", "workspace_id": "ws-1"}]],
                "kwargs": {
                    "conversation_id": "conv-1",
                    "user_id": "user-1",
                    "workspace_id": "ws-2",
                },
                "expected": False,
                "query_count": 1,
            },
            {
                "name": "personal param but workspace conversation",
                "side_effect": [[{"user_id": "owner-1", "workspace_id": "ws-1"}]],
                "kwargs": {"conversation_id": "conv-1", "user_id": "user-1"},
                "expected": False,
                "query_count": 1,
            },
            {
                "name": "admin after workspace gate",
                "side_effect": [[{"user_id": "owner-1", "workspace_id": "ws-1"}]],
                "kwargs": {
                    "conversation_id": "conv-1",
                    "user_id": "admin-1",
                    "workspace_id": "ws-1",
                    "is_admin": True,
                },
                "expected": True,
                "query_count": 1,
            },
            {
                "name": "admin with personal conversation",
                "side_effect": [[{"user_id": "owner-1", "workspace_id": None}]],
                "kwargs": {
                    "conversation_id": "conv-1",
                    "user_id": "admin-1",
                    "is_admin": True,
                },
                "expected": True,
                "query_count": 1,
            },
            {
                "name": "no user_id",
                "side_effect": [[{"user_id": "owner-1", "workspace_id": None}]],
                "kwargs": {"conversation_id": "conv-1", "user_id": None},
                "expected": False,
                "query_count": 1,
            },
            {
                "name": "workspace owner",
                "side_effect": [
                    [{"user_id": "owner-1", "workspace_id": "ws-1"}],
                    [{"owner_user_id": "user-1", "is_member": False}],
                ],
                "kwargs": {
                    "conversation_id": "conv-1",
                    "user_id": "user-1",
                    "workspace_id": "ws-1",
                },
                "expected": True,
                "query_count": 2,
            },
            {
                "name": "workspace member",
                "side_effect": [
                    [{"user_id": "owner-1", "workspace_id": "ws-1"}],
                    [{"owner_user_id": "owner-1", "is_member": True}],
                ],
                "kwargs": {
                    "conversation_id": "conv-1",
                    "user_id": "user-2",
                    "workspace_id": "ws-1",
                },
                "expected": True,
                "query_count": 2,
            },
            {
                "name": "workspace non-member",
                "side_effect": [
                    [{"user_id": "owner-1", "workspace_id": "ws-1"}],
                    [{"owner_user_id": "owner-1", "is_member": False}],
                ],
                "kwargs": {
                    "conversation_id": "conv-1",
                    "user_id": "user-3",
                    "workspace_id": "ws-1",
                },
                "expected": False,
                "query_count": 2,
            },
            {
                "name": "workspace row missing",
                "side_effect": [
                    [{"user_id": "owner-1", "workspace_id": "ws-1"}],
                    [],
                ],
                "kwargs": {
                    "conversation_id": "conv-1",
                    "user_id": "user-3",
                    "workspace_id": "ws-1",
                },
                "expected": False,
                "query_count": 2,
            },
            {
                "name": "legacy conversation without owner",
                "side_effect": [[{"user_id": None, "workspace_id": None}]],
                "kwargs": {"conversation_id": "conv-1", "user_id": "user-1"},
                "expected": True,
                "query_count": 1,
            },
            {
                "name": "own conversation",
                "side_effect": [[{"user_id": "user-1", "workspace_id": None}]],
                "kwargs": {"conversation_id": "conv-1", "user_id": "user-1"},
                "expected": True,
                "query_count": 1,
            },
            {
                "name": "other conversation with membership",
                "side_effect": [
                    [{"user_id": "owner-1", "workspace_id": None}],
                    [{"present": 1}],
                ],
                "kwargs": {"conversation_id": "conv-1", "user_id": "user-2"},
                "expected": True,
                "query_count": 2,
            },
            {
                "name": "other conversation without membership",
                "side_effect": [
                    [{"user_id": "owner-1", "workspace_id": None}],
                    [],
                ],
                "kwargs": {"conversation_id": "conv-1", "user_id": "user-2"},
                "expected": False,
                "query_count": 2,
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                query_raw = mock.AsyncMock(side_effect=case["side_effect"])
                db = SimpleNamespace(query_raw=query_raw)
                with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
                    result = await repository.check_conversation_access(**case["kwargs"])

                self.assertEqual(result, case["expected"])
                self.assertEqual(query_raw.await_count, case["query_count"])
                for call in query_raw.await_args_list:
                    sql = call.args[0]
                    self.assertNotIn("c.messages", sql)

    async def test_get_conversation_branches_maps_raw_sql_rows(self) -> None:
        query_raw = mock.AsyncMock(
            return_value=[
                {
                    "id": "branch-1",
                    "conversation_id": "conv-1",
                    "parent_branch_id": None,
                    "branch_point_index": "3",
                    "branch_kind": "replay",
                    "message_count": "7",
                    "associated_snapshot_id": "snap-1",
                    "created_by_user_id": "user-1",
                    "created_by_username": "alice",
                    "created_at": NOW,
                }
            ]
        )
        db = SimpleNamespace(query_raw=query_raw)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            branches = await repository.get_conversation_branches("conv-1")

        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0].id, "branch-1")
        self.assertEqual(branches[0].branch_point_index, 3)
        self.assertEqual(branches[0].branch_kind, ConversationBranchKind.REPLAY)
        self.assertEqual(branches[0].message_count, 7)
        self.assertEqual(branches[0].created_by_username, "alice")
        await_args = query_raw.await_args
        if await_args is None:
            self.fail("query_raw was not awaited")
        self.assertIn("jsonb_array_length", await_args.args[0])

    async def test_get_conversation_branches_returns_empty_list_on_query_error(self) -> None:
        db = SimpleNamespace(query_raw=mock.AsyncMock(side_effect=RuntimeError("boom")))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            branches = await repository.get_conversation_branches("conv-1")

        self.assertEqual(branches, [])

    async def test_get_conversation_active_branch_id_variants(self) -> None:
        cases = [
            ([{"active_branch_id": "branch-1"}], "branch-1"),
            ([{"active_branch_id": None}], None),
            ([], None),
        ]

        for rows, expected in cases:
            with self.subTest(expected=expected):
                db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=rows))
                with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
                    result = await repository.get_conversation_active_branch_id("conv-1")
                self.assertEqual(result, expected)

    async def test_get_workspace_tab_state_uses_authorized_devserver_helper_once(self) -> None:
        runtime_status = UserSpaceRuntimeStatusResponse(
            workspace_id="ws-1",
            session_state="running",
            session_id="session-1",
            devserver_running=True,
            devserver_port=5173,
        )
        chat_state = WorkspaceChatStateResponse()

        with (
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ) as enforce_mock,
            mock.patch(
                "ragtime.userspace.runtime_service.build_workspace_chat_state",
                mock.AsyncMock(return_value=chat_state),
            ) as chat_state_mock,
            mock.patch.object(
                userspace_runtime_service,
                "_get_devserver_status_authorized",
                mock.AsyncMock(return_value=runtime_status),
                create=True,
            ) as authorized_mock,
            mock.patch.object(
                userspace_runtime_service,
                "get_devserver_status",
                mock.AsyncMock(side_effect=AssertionError("public get_devserver_status should not be used")),
            ),
        ):
            result = await userspace_runtime_service.get_workspace_tab_state(
                "ws-1",
                "user-1",
                selected_conversation_id="conv-1",
                is_admin=True,
            )

        enforce_mock.assert_awaited_once_with("ws-1", "user-1", "viewer", is_admin=True)
        authorized_mock.assert_awaited_once_with("ws-1", "user-1")
        chat_state_mock.assert_awaited_once_with(
            workspace_id="ws-1",
            user_id="user-1",
            is_admin=True,
            selected_conversation_id="conv-1",
            include_selected_conversation=True,
        )
        self.assertIs(result.runtime_status, runtime_status)
        self.assertIs(result.chat_state, chat_state)


if __name__ == "__main__":
    unittest.main()
