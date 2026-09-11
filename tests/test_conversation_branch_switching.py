import copy
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User

from ragtime.indexer.models import SwitchConversationBranchRequest
from ragtime.indexer.repository import ConversationBranchMutationError, IndexerRepository


def _plain(value):
    return copy.deepcopy(getattr(value, "data", value))


class _BranchStore:
    def __init__(self, rows):
        self.rows = {row.id: row for row in rows}

    async def find_many(self, **_kwargs):
        return list(self.rows.values())

    async def find_unique(self, *, where, **_kwargs):
        return self.rows.get(where["id"])

    async def update(self, *, where, data):
        row = self.rows[where["id"]]
        for key, value in data.items():
            setattr(row, key, _plain(value))
        return row

    async def create(self, *, data, **_kwargs):
        row_data = {key: _plain(value) for key, value in data.items()}
        now = datetime.now(timezone.utc)
        row_data.setdefault("createdAt", now)
        row_data.setdefault("updatedAt", now)
        row = SimpleNamespace(**row_data)
        self.rows[row.id] = row
        return row


class _ConversationStore:
    def __init__(self, row):
        self.row = row

    async def find_unique(self, **_kwargs):
        return self.row

    async def update(self, *, data, **_kwargs):
        for key, value in data.items():
            setattr(self.row, key, _plain(value))
        return self.row


class _Tx:
    def __init__(self, conversation, branches):
        self.conversation = _ConversationStore(conversation)
        self.conversationbranch = _BranchStore(branches)

    async def query_raw(self, _query):
        return [{"id": self.conversation.row.id}]


class _Transaction:
    def __init__(self, tx):
        self.tx = tx

    async def __aenter__(self):
        return self.tx

    async def __aexit__(self, *_args):
        return False


class ConversationBranchPrefixTests(unittest.IsolatedAsyncioTestCase):
    def _repo(self, conversation, branches):
        tx = _Tx(conversation, branches)
        repo = IndexerRepository()
        repo._get_db = mock.AsyncMock(return_value=SimpleNamespace(tx=mock.Mock(return_value=_Transaction(tx))))
        repo._prisma_conversation_to_model = mock.Mock(side_effect=lambda row: row)
        return repo, tx

    @staticmethod
    def _row(branch_id, point, base, suffix, *, parent=None, kind: str | None = "edit"):
        return SimpleNamespace(
            id=branch_id,
            conversationId="conversation",
            branchPointIndex=point,
            parentBranchId=parent,
            branchKind=kind,
            baseMessages=copy.deepcopy(base),
            preservedMessages=copy.deepcopy(suffix),
            userId="user",
        )

    async def test_cross_depth_round_trip_preserves_each_active_source(self):
        root = [{"content": str(index)} for index in range(6)]
        a_messages = root[:2] + [{"content": "A2"}, {"content": "A3"}]
        b_messages = root[:4] + [{"content": "B4"}, {"content": "B5"}]
        a = self._row("A", 2, a_messages[:2], a_messages[2:], parent="parent-a")
        b = self._row("B", 4, b_messages[:4], b_messages[4:], parent="parent-b")
        conversation = SimpleNamespace(id="conversation", messages=copy.deepcopy(a_messages), activeBranchId="A", activeTaskId=None, userId="user")
        repo, tx = self._repo(conversation, [a, b])

        await repo.switch_conversation_branch("conversation", "B")
        self.assertEqual(conversation.messages, b_messages)
        self.assertEqual(tx.conversationbranch.rows["A"].preservedMessages, a_messages[2:])
        await repo.release_conversation_branch("conversation")
        self.assertEqual(conversation.messages, a_messages)
        await repo.switch_conversation_branch("conversation", "B")
        await repo.switch_conversation_branch("conversation", "A")
        self.assertEqual(conversation.messages, a_messages)
        self.assertEqual(tx.conversationbranch.rows["B"].preservedMessages, b_messages[4:])
        await repo.switch_conversation_branch("conversation", "B")
        await repo.switch_conversation_branch("conversation", "A")
        self.assertEqual(conversation.messages, a_messages)
        self.assertEqual(tx.conversationbranch.rows["A"].preservedMessages, a_messages[2:])
        self.assertEqual(tx.conversationbranch.rows["B"].preservedMessages, b_messages[4:])

    async def test_create_returns_stored_branch_and_truncates_conversation(self):
        messages = [{"content": label} for label in ("root", "answer", "edited", "reply")]
        conversation = SimpleNamespace(
            id="conversation",
            messages=copy.deepcopy(messages),
            activeBranchId=None,
            activeTaskId=None,
            userId="user",
        )
        repo, tx = self._repo(conversation, [])

        created = await repo.create_conversation_branch("conversation", 2, user_id="user")

        if created is None:
            self.fail("Expected the newly stored branch to be returned")
        self.assertIn(created.id, tx.conversationbranch.rows)
        stored = tx.conversationbranch.rows[created.id]
        self.assertEqual(stored.baseMessages, messages[:2])
        self.assertEqual(stored.preservedMessages, messages[2:])
        self.assertEqual(conversation.messages, messages[:2])

    async def test_same_group_current_switch_never_creates_or_overwrites_sibling(self):
        messages = [{"content": str(index)} for index in range(4)]
        current = self._row("Current", 2, messages[:2], messages[2:], parent="parent", kind=None)
        edited = self._row("Edited", 2, messages[:2], [{"content": "edited"}], parent="parent")
        conversation = SimpleNamespace(id="conversation", messages=copy.deepcopy(messages), activeBranchId="Current", activeTaskId=None, userId="user")
        repo, tx = self._repo(conversation, [current, edited])

        for _ in range(10):
            await repo.switch_conversation_branch("conversation", "Edited")
            await repo.switch_conversation_branch("conversation", "Current")

        self.assertEqual(set(tx.conversationbranch.rows), {"Current", "Edited"})
        self.assertEqual(tx.conversationbranch.rows["Current"].preservedMessages, messages[2:])

    async def test_legacy_active_source_is_frozen_from_live_edited_content_first(self):
        live = [{"content": "root"}, {"content": "edited"}, {"content": "new"}]
        active = self._row("active", 1, None, [{"content": "stale"}], parent="parent")
        active.baseMessages = None
        target = self._row("target", 1, [{"content": "root"}], [{"content": "target"}])
        conversation = SimpleNamespace(id="conversation", messages=copy.deepcopy(live), activeBranchId="active", activeTaskId=None, userId="user")
        repo, tx = self._repo(conversation, [active, target])

        await repo.switch_conversation_branch("conversation", "target")
        self.assertEqual(tx.conversationbranch.rows["active"].baseMessages, live[:1])
        self.assertEqual(tx.conversationbranch.rows["active"].preservedMessages, live[1:])

    async def test_active_task_rejects_switch_without_mutating_rows(self):
        branch = self._row("target", 0, [], [{"content": "target"}])
        conversation = SimpleNamespace(
            id="conversation",
            messages=[{"content": "live"}],
            activeBranchId=None,
            activeTaskId="task",
            userId="user",
        )
        repo, tx = self._repo(conversation, [branch])
        with self.assertRaises(ConversationBranchMutationError) as context:
            await repo.switch_conversation_branch("conversation", "target")
        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(tx.conversationbranch.rows["target"].preservedMessages, [{"content": "target"}])

    async def test_release_active_current_clears_pointer_without_truncating_messages(self):
        messages = [{"content": "one"}, {"content": "two"}]
        current = self._row("Current", 1, messages[:1], messages[1:], kind=None)
        conversation = SimpleNamespace(id="conversation", messages=copy.deepcopy(messages), activeBranchId="Current", activeTaskId=None, userId="user")
        repo, _tx = self._repo(conversation, [current])
        await repo.release_conversation_branch("conversation")
        self.assertEqual(conversation.messages, messages)
        self.assertIsNone(conversation.activeBranchId)

    async def test_cross_group_switch_never_overwrites_inactive_current_with_same_prefix(self):
        root = [{"content": "root"}, {"content": "shared"}]
        old_view = root + [{"content": "old"}]
        target = self._row("target", 2, root, [{"content": "target"}], parent="target-group")
        current_new = self._row("CurrentNew", 2, root, [{"content": "new"}], parent="target-group", kind=None)
        source = self._row("ancestor", 1, old_view[:1], old_view[1:], parent="ancestor-group")
        conversation = SimpleNamespace(id="conversation", messages=copy.deepcopy(old_view), activeBranchId="ancestor", activeTaskId=None, userId="user")
        repo, tx = self._repo(conversation, [source, target, current_new])

        await repo.switch_conversation_branch("conversation", "target")

        self.assertEqual(tx.conversationbranch.rows["CurrentNew"].preservedMessages, [{"content": "new"}])
        self.assertEqual(len(tx.conversationbranch.rows), 4)

    async def test_malformed_or_cyclic_lineage_fails_before_mutation(self):
        first = self._row("first", 0, None, [], parent="second")
        second = self._row("second", 0, None, [], parent="first")
        tx = SimpleNamespace(conversationbranch=SimpleNamespace(find_many=mock.AsyncMock(return_value=[first, second]), update=mock.AsyncMock()))
        with self.assertRaises(ConversationBranchMutationError):
            await IndexerRepository()._freeze_and_validate_branch_bases(tx, "conversation", [])
        tx.conversationbranch.update.assert_not_awaited()


class ConversationBranchRouteConflictTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _user() -> User:
        now = datetime.now(timezone.utc)
        return User(
            id="user",
            username="user",
            authProvider=AuthProvider.local,
            cachedGroups=cast(Json, "[]"),
            role=UserRole.user,
            roleManuallySet=False,
            createdAt=now,
            updatedAt=now,
        )

    async def test_switch_surfaces_active_task_conflict(self):
        from ragtime.indexer import routes

        repository = SimpleNamespace(
            check_conversation_access=mock.AsyncMock(return_value=True),
            switch_conversation_branch=mock.AsyncMock(side_effect=ConversationBranchMutationError("Cannot switch branches while a chat is in progress")),
        )
        with (
            mock.patch.object(routes, "repository", repository),
            mock.patch.object(routes, "_assert_workspace_access", mock.AsyncMock()),
        ):
            with self.assertRaises(HTTPException) as context:
                await routes.switch_conversation_branch(
                    "conversation",
                    SwitchConversationBranchRequest(branch_id="target"),
                    None,
                    self._user(),
                )
        self.assertEqual(context.exception.status_code, 409)

    async def test_delete_surfaces_child_branch_conflict(self):
        from ragtime.indexer import routes

        repository = SimpleNamespace(
            check_conversation_access=mock.AsyncMock(return_value=True),
            delete_conversation_branch=mock.AsyncMock(side_effect=ConversationBranchMutationError("Cannot delete a branch that has child branches")),
        )
        with (
            mock.patch.object(routes, "repository", repository),
            mock.patch.object(routes, "_assert_workspace_access", mock.AsyncMock()),
        ):
            with self.assertRaises(HTTPException) as context:
                await routes.delete_conversation_branch("conversation", "parent", None, self._user())
        self.assertEqual(context.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
