import unittest
from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from fastapi import HTTPException
from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User

from ragtime.content_protection.models import ContentProtectionError
from ragtime.indexer import routes
from ragtime.indexer.models import ChatTaskStatus, QueueMessageRequest


def _user(user_id: str = "user-1") -> User:
    now = datetime.now(timezone.utc)
    return User(
        id=user_id,
        username=user_id,
        authProvider=AuthProvider.local,
        cachedGroups=cast(Json, "[]"),
        role=UserRole.user,
        roleManuallySet=False,
        createdAt=now,
        updatedAt=now,
        securityGeneration=0,
    )


def _awaited_kwargs(mock_call: mock.AsyncMock) -> Mapping[str, Any]:
    assert mock_call.await_args is not None
    return mock_call.await_args.kwargs


def _task(task_id: str = "task-1", status: ChatTaskStatus = ChatTaskStatus.pending) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=task_id,
        conversation_id="conversation-1",
        status=status,
        user_message="message",
        streaming_state={},
        response_content=None,
        error_message=None,
        created_at=now,
        started_at=None,
        completed_at=None,
        last_update_at=now,
    )


class QueueRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_keeps_canonical_multimodal_message_when_started(self) -> None:
        message = '[{"type":"text","text":"See [a.py:1]"},{"type":"file","attachment_id":"file-1","attachment_source":"chat_upload"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]'
        user = _user()
        conversation = SimpleNamespace(user_id="user-1", workspace_id=None, model="model", disabled_builtin_tool_ids=[])
        request = QueueMessageRequest(
            message=message,
            segments=[{"type": "ref", "reference": {"id": "ref-1", "path": "a.py", "source": "file", "startLine": 1}}],
        )
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        with (
            mock.patch.object(routes, "_assert_queue_access", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(routes, "resolve_chat_attachment", return_value=({}, None)),
            mock.patch.object(routes, "_resolve_workspace_runtime_scope", new=mock.AsyncMock(return_value=(None, set(), None))),
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(routes, "_validate_generation_ready_after_user_message", new=mock.AsyncMock(return_value="model")),
            mock.patch.object(routes.repository, "enqueue_chat_message", new=mock.AsyncMock(return_value=("started", (conversation, _task())))) as enqueue,
            mock.patch.object(routes, "_apply_validated_conversation_model", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "schedule_title_generation"),
            mock.patch.object(routes, "_create_background_chat_task_after_user_message", new=mock.AsyncMock(return_value=_task())),
        ):
            response = await routes.enqueue_conversation_message("conversation-1", request, user=user)

        self.assertEqual(response["status"], "started")
        self.assertEqual(_awaited_kwargs(enqueue)["message"], message)
        self.assertEqual(_awaited_kwargs(enqueue)["segments"], request.segments)

    async def test_enqueue_rejects_missing_attachment_before_claim(self) -> None:
        request = QueueMessageRequest(message='[{"type":"file","attachment_id":"gone","attachment_source":"chat_upload"}]')
        user = _user()
        conversation = SimpleNamespace(user_id="user-1", workspace_id=None, model="model")
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        with (
            mock.patch.object(routes, "_assert_queue_access", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(routes, "resolve_chat_attachment", side_effect=FileNotFoundError("gone")),
            mock.patch.object(routes.repository, "enqueue_chat_message", new=mock.AsyncMock()) as enqueue,
        ):
            with self.assertRaises(HTTPException) as raised:
                await routes.enqueue_conversation_message("conversation-1", request, user=user)

        self.assertEqual(raised.exception.status_code, 409)
        enqueue.assert_not_awaited()

    async def test_enqueue_rejects_malformed_attachment_content_with_a_readable_error(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await routes._validate_queue_attachments(
                '[{"type":"text","text":"valid"},"invalid"]',
                "conversation-1",
                "user-1",
                None,
            )

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail, "Attachment content is invalid")

    async def test_dispatch_revoked_stored_user_retains_error_without_claiming(self) -> None:
        head = SimpleNamespace(id="queued-1", userId="stored-user")
        db = SimpleNamespace(
            user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=None)),
            chatqueuedmessage=SimpleNamespace(update_many=mock.AsyncMock()),
        )
        with (
            mock.patch.object(routes, "_assert_queue_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "list_queued_chat_messages", new=mock.AsyncMock(return_value=([head], None, True))),
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(routes.repository, "claim_queued_chat_message", new=mock.AsyncMock()) as claim,
            mock.patch.object(routes, "_assert_workspace_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "check_conversation_access", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(routes.repository, "get_conversation", new=mock.AsyncMock()),
        ):
            with self.assertRaises(HTTPException) as raised:
                await routes.dispatch_conversation_queue("conversation-1", user=_user("caller"))

        self.assertEqual(raised.exception.status_code, 403)
        claim.assert_not_awaited()
        db.chatqueuedmessage.update_many.assert_awaited_once()

    async def test_manual_dispatch_persists_and_reraises_content_protection_denial(self) -> None:
        head = SimpleNamespace(id="queued-1", userId="stored-user")
        denial = ContentProtectionError("content_denied", "request-1")
        db = SimpleNamespace(chatqueuedmessage=SimpleNamespace(update_many=mock.AsyncMock()))
        with (
            mock.patch.object(routes, "_assert_queue_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "list_queued_chat_messages", new=mock.AsyncMock(return_value=([head], None, True))),
            mock.patch.object(routes, "_prepare_queued_message_dispatch", new=mock.AsyncMock(side_effect=denial)),
            mock.patch.object(routes, "_persist_content_protection_denial", new=mock.AsyncMock()) as persist,
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await routes.dispatch_conversation_queue("conversation-1", user=_user("caller"))

        self.assertIs(raised.exception, denial)
        persist.assert_awaited_once_with("conversation-1", denial)
        self.assertEqual(db.chatqueuedmessage.update_many.await_args.kwargs["data"]["dispatchError"], "content_denied")

    async def test_auto_dispatch_persists_content_protection_denial_and_holds_item(self) -> None:
        head = SimpleNamespace(id="queued-1", userId="stored-user")
        denial = ContentProtectionError("content_denied", "request-1")
        db = SimpleNamespace(chatqueuedmessage=SimpleNamespace(update_many=mock.AsyncMock()))
        with (
            mock.patch.object(routes.repository, "list_queued_chat_messages", new=mock.AsyncMock(return_value=([head], None, True))),
            mock.patch.object(routes, "_prepare_queued_message_dispatch", new=mock.AsyncMock(side_effect=denial)),
            mock.patch.object(routes, "_persist_content_protection_denial", new=mock.AsyncMock()) as persist,
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(routes.repository, "claim_queued_chat_message", new=mock.AsyncMock()) as claim,
        ):
            await routes._dispatch_queue_after_normal_completion("conversation-1")

        persist.assert_awaited_once_with("conversation-1", denial)
        self.assertEqual(db.chatqueuedmessage.update_many.await_args.kwargs["data"]["dispatchError"], "content_denied")
        claim.assert_not_awaited()

    async def test_enqueue_drops_oversized_display_segments_without_changing_canonical_message(self) -> None:
        message = "canonical message"
        display_text = "x" * 100_001
        request = QueueMessageRequest(message=message, segments=[{"type": "text", "text": display_text}])
        enqueue = await self._enqueue_with_request(request)

        self.assertEqual(_awaited_kwargs(enqueue)["message"], message)
        self.assertEqual(_awaited_kwargs(enqueue)["segments"], [])

    async def test_enqueue_drops_all_display_segments_when_more_than_200_are_sent(self) -> None:
        message = "canonical message"
        segments = [{"type": "text", "text": f"part {index}"} for index in range(201)]
        request = QueueMessageRequest(message=message, segments=segments)
        enqueue = await self._enqueue_with_request(request)

        self.assertEqual(_awaited_kwargs(enqueue)["message"], message)
        self.assertEqual(_awaited_kwargs(enqueue)["segments"], [])

    async def _enqueue_with_request(self, request: QueueMessageRequest) -> mock.AsyncMock:
        user = _user()
        conversation = SimpleNamespace(user_id="user-1", workspace_id=None, model="model", disabled_builtin_tool_ids=[])
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        with (
            mock.patch.object(routes, "_assert_queue_access", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(routes, "_resolve_workspace_runtime_scope", new=mock.AsyncMock(return_value=(None, set(), None))),
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(routes, "_validate_generation_ready_after_user_message", new=mock.AsyncMock(return_value="model")),
            mock.patch.object(routes.repository, "enqueue_chat_message", new=mock.AsyncMock(return_value=("started", (conversation, _task())))) as enqueue,
            mock.patch.object(routes, "_apply_validated_conversation_model", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "schedule_title_generation"),
            mock.patch.object(routes, "_create_background_chat_task_after_user_message", new=mock.AsyncMock(return_value=_task())),
        ):
            await routes.enqueue_conversation_message("conversation-1", request, user=user)
        return enqueue

    async def test_cancel_returns_terminal_task_when_guarded_cancel_loses_race(self) -> None:
        running = _task(status=ChatTaskStatus.running)
        completed = _task(status=ChatTaskStatus.completed)
        user = _user()
        with (
            mock.patch.object(routes.repository, "get_chat_task", new=mock.AsyncMock(side_effect=[running, completed])),
            mock.patch.object(routes, "_assert_workspace_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "check_conversation_access", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(routes.background_task_service, "cancel_task"),
            mock.patch.object(routes.repository, "get_conversation", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(routes.repository, "cancel_chat_task", new=mock.AsyncMock(return_value=None)),
        ):
            response = await routes.cancel_chat_task("task-1", user=user)

        self.assertEqual(response.status, ChatTaskStatus.completed)
