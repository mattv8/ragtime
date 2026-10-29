import asyncio
import os
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from prisma.enums import AuthProvider

from ragtime.core.database import connect_db, disconnect_db
from ragtime.indexer import routes
from ragtime.indexer.models import ChatQueuedMessageResponse, QueueMessageRequest
from ragtime.indexer.repository import repository


def test_queue_request_preserves_canonical_attachment_message_and_display_segments() -> None:
    message = '[{"type":"file","attachment_id":"file-1","attachment_source":"chat_upload"}]'
    request = QueueMessageRequest(
        message=message,
        segments=[{"type": "text", "text": "see attachment"}],
    )

    assert request.message == message
    assert request.segments == [{"type": "text", "text": "see attachment"}]


def test_queue_request_rejects_malformed_reference_segment() -> None:
    with __import__("pytest").raises(ValueError, match="reference path"):
        QueueMessageRequest(message="prompt", segments=[{"type": "ref", "reference": {"id": "ref-1", "source": "file"}}])


def test_queue_request_drops_display_segments_when_reference_path_is_oversized() -> None:
    request = QueueMessageRequest(
        message="prompt",
        segments=[{"type": "ref", "reference": {"id": "ref-1", "path": "p" * 4097, "source": "file"}}],
    )

    assert request.segments == []


def test_queue_request_still_rejects_malformed_segments_when_oversized() -> None:
    segments = [{"type": "text", "text": "x"} for _ in range(200)] + [{"type": "bogus"}]
    with __import__("pytest").raises(ValueError, match="text or reference"):
        QueueMessageRequest(message="prompt", segments=segments)


def test_queue_response_serializes_bigint_sequence_as_string() -> None:
    response = ChatQueuedMessageResponse(
        id="item-1",
        conversation_id="conversation-1",
        message="hello",
        segments=[],
        request_context={},
        seq="9223372036854775807",
        dispatch_error=None,
        created_at=datetime.now(timezone.utc),
    )

    assert response.model_dump()["seq"] == "9223372036854775807"


@unittest.skipUnless(os.environ.get("RAGTIME_QUEUE_TEST_DATABASE") == "1", "requires isolated queue test database")
class ChatMessageQueueRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db = await connect_db()
        suffix = uuid.uuid4().hex
        self.user_id = f"queue-user-{suffix}"
        self.conversation_id = f"queue-conversation-{suffix}"
        await self.db.user.create(data={"id": self.user_id, "username": f"queue-{suffix}", "authProvider": AuthProvider.local})
        await self.db.conversation.create(data={"id": self.conversation_id, "userId": self.user_id})

    async def asyncTearDown(self) -> None:
        await self.db.conversation.delete(where={"id": self.conversation_id})
        await self.db.user.delete(where={"id": self.user_id})
        await disconnect_db()

    async def _enqueue(self, message: str):
        return await repository.enqueue_chat_message(
            self.conversation_id,
            user_id=self.user_id,
            workspace_id=None,
            message=message,
            segments=[],
            request_context={},
        )

    async def test_fifo_claims_only_head_after_active_task_finishes(self) -> None:
        status, (_, active_task) = await self._enqueue("active")
        self.assertEqual(status, "started")
        self.assertEqual((await self._enqueue("first"))[0], "queued")
        self.assertEqual((await self._enqueue("second"))[0], "queued")
        self.assertEqual((await repository.claim_queued_chat_message(self.conversation_id))[0], "busy")

        await repository.cancel_chat_task(active_task.id)
        status, (item, _, claimed_task) = await repository.claim_queued_chat_message(self.conversation_id)
        self.assertEqual(status, "started")
        self.assertEqual(item.message, "first")
        await repository.cancel_chat_task(claimed_task.id)

        status, (item, _, _) = await repository.claim_queued_chat_message(self.conversation_id)
        self.assertEqual(status, "started")
        self.assertEqual(item.message, "second")

    async def test_queue_cap_is_transactional(self) -> None:
        _, (_, active_task) = await self._enqueue("active")
        results = await asyncio.gather(*(self._enqueue(f"queued-{index}") for index in range(20)))
        self.assertTrue(all(status == "queued" for status, _item in results))
        with self.assertRaises(HTTPException) as raised:
            await self._enqueue("overflow")
        self.assertEqual(raised.exception.status_code, 409)
        await repository.cancel_chat_task(active_task.id)

    async def test_delete_and_claim_cannot_both_consume_the_same_item(self) -> None:
        _, (_, active_task) = await self._enqueue("active")
        _, queued_item = await self._enqueue("queued")
        await repository.cancel_chat_task(active_task.id)
        removed, claimed = await asyncio.gather(
            repository.remove_queued_chat_message(self.conversation_id, queued_item.id),
            repository.claim_queued_chat_message(self.conversation_id),
        )
        self.assertTrue((removed is not None) ^ (claimed[0] == "started"))

    async def test_late_cancel_does_not_rewrite_completed_task_or_clear_new_active_task(self) -> None:
        _, (_, first_task) = await self._enqueue("first")
        await self._enqueue("second")
        await repository.complete_chat_task(first_task.id, "done", [], [])
        _, (_item, _conversation, second_task) = await repository.claim_queued_chat_message(self.conversation_id)

        self.assertIsNone(await repository.cancel_chat_task(first_task.id))
        first = await repository.get_chat_task(first_task.id)
        active = await repository.get_active_task_for_conversation(self.conversation_id)
        assert first is not None
        assert active is not None
        self.assertEqual(first.status.value, "completed")
        self.assertEqual(active.id, second_task.id)

    async def test_stale_terminal_pointer_allows_enqueue_and_dispatch_claim(self) -> None:
        _, (_, completed_task) = await self._enqueue("completed")
        await repository.complete_chat_task(completed_task.id, "done", [], [])
        await self.db.conversation.update(where={"id": self.conversation_id}, data={"activeTaskId": completed_task.id})

        status, (_, replacement_task) = await self._enqueue("replacement")
        self.assertEqual(status, "started")
        queued_status, _queued = await self._enqueue("queued")
        self.assertEqual(queued_status, "queued")
        await repository.cancel_chat_task(replacement_task.id)
        await self.db.conversation.update(where={"id": self.conversation_id}, data={"activeTaskId": completed_task.id})

        status, (_item, _conversation, dispatched_task) = await repository.claim_queued_chat_message(self.conversation_id)
        self.assertEqual(status, "started")
        self.assertEqual(dispatched_task.user_message, "queued")


class QueueDispatchPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_automatic_dispatch_retains_head_when_stored_user_access_is_revoked(self) -> None:
        head = SimpleNamespace(id="queued-1")
        db = SimpleNamespace(chatqueuedmessage=SimpleNamespace(update_many=mock.AsyncMock()))
        with (
            mock.patch.object(routes.repository, "list_queued_chat_messages", new=mock.AsyncMock(return_value=([head], None, True))),
            mock.patch.object(routes.repository, "claim_queued_chat_message", new=mock.AsyncMock()) as claim,
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(
                routes, "_prepare_queued_message_dispatch", new=mock.AsyncMock(side_effect=HTTPException(status_code=404, detail="Conversation not found"))
            ),
        ):
            await routes._dispatch_queue_after_normal_completion("conversation-1")

        claim.assert_not_awaited()
        db.chatqueuedmessage.update_many.assert_awaited_once_with(
            where={"id": "queued-1", "conversationId": "conversation-1"},
            data={"dispatchError": "Conversation not found"},
        )

    async def test_automatic_dispatch_retains_head_when_content_protection_rejects_preflight(self) -> None:
        head = SimpleNamespace(id="queued-1")
        db = SimpleNamespace(chatqueuedmessage=SimpleNamespace(update_many=mock.AsyncMock()))
        with (
            mock.patch.object(routes.repository, "list_queued_chat_messages", new=mock.AsyncMock(return_value=([head], None, True))),
            mock.patch.object(routes.repository, "claim_queued_chat_message", new=mock.AsyncMock()) as claim,
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(
                routes, "_prepare_queued_message_dispatch", new=mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="Content blocked"))
            ),
        ):
            await routes._dispatch_queue_after_normal_completion("conversation-1")

        claim.assert_not_awaited()
        db.chatqueuedmessage.update_many.assert_awaited_once_with(
            where={"id": "queued-1", "conversationId": "conversation-1"},
            data={"dispatchError": "Content blocked"},
        )
