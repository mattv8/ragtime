from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from ragtime.content_protection.models import ContentProtectionError
from ragtime.indexer import routes


class ChatDenialRouteTests(IsolatedAsyncioTestCase):
    async def test_user_write_protection_failure_keeps_the_original_error_when_notice_storage_fails(self) -> None:
        """Replacing a guarded user-write denial with storage failure loses the protection outcome."""
        error = ContentProtectionError("content_denied", "request-1")
        conversation = SimpleNamespace(id="conversation-1", user_id="owner", workspace_id=None)
        user = SimpleNamespace(id="caller")

        with (
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "add_message", new=mock.AsyncMock(side_effect=error)),
            mock.patch.object(routes, "persist_chat_denial", new=mock.AsyncMock(side_effect=RuntimeError("storage unavailable"))),
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await routes._send_message_to_loaded_conversation(
                    conversation,
                    routes.SendMessageRequest(message="allowed"),
                    user,
                    workspace_id=None,
                    blocked_tool_names=set(),
                )

        self.assertIs(raised.exception, error)

    async def test_assistant_write_protection_failure_persists_only_a_safe_notice(self) -> None:
        """Dropping the output write guard would leave protected model output on the normal path."""
        error = ContentProtectionError("content_denied", "request-1")
        conversation = SimpleNamespace(
            id="conversation-1",
            user_id="owner",
            workspace_id=None,
            model="test",
            messages=[],
            disabled_builtin_tool_ids=[],
        )
        user = SimpleNamespace(id="caller", username="caller", displayName="Caller", role="user")

        with (
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "add_message", new=mock.AsyncMock(side_effect=[conversation, error])) as add_message,
            mock.patch.object(routes, "_validate_generation_ready_after_user_message", new=mock.AsyncMock(return_value="test")),
            mock.patch.object(routes, "_apply_validated_conversation_model", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "_build_chat_history_for_conversation", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(routes.rag, "process_query", new=mock.AsyncMock(return_value="protected model output")),
            mock.patch.object(routes, "create_usage_attempt", new=mock.AsyncMock(return_value="attempt-1")),
            mock.patch.object(routes, "finalize_usage_attempt", new=mock.AsyncMock()),
            mock.patch.object(routes, "schedule_title_generation"),
            mock.patch.object(routes, "persist_chat_denial", new=mock.AsyncMock(return_value="notice-1")) as persist_notice,
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await routes._send_message_to_loaded_conversation(
                    conversation,
                    routes.SendMessageRequest(message="allowed"),
                    user,
                    workspace_id=None,
                    blocked_tool_names=set(),
                )

        self.assertIs(raised.exception, error)
        self.assertEqual(add_message.await_args_list[1].args[2], "protected model output")
        persist_notice.assert_awaited_once_with("conversation-1", error)

    async def test_background_admission_denial_persists_notice_without_creating_atomic_user_or_task_rows(self) -> None:
        """Removing route admission would let rejected input reach the atomic writer."""
        error = ContentProtectionError("content_denied", "request-1")
        conversation = SimpleNamespace(id="conversation-1", user_id="owner", workspace_id=None)
        user = SimpleNamespace(id="caller", role="user")

        with (
            mock.patch.object(routes, "_assert_workspace_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "check_conversation_access", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(routes.repository, "get_conversation", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "_resolve_workspace_runtime_scope", new=mock.AsyncMock(return_value=(None, set(), None))),
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock(side_effect=error)),
            mock.patch.object(routes, "persist_chat_denial", new=mock.AsyncMock(return_value="notice-1")) as persist_notice,
            mock.patch.object(routes.repository, "add_user_message_and_create_chat_task_if_idle", new=mock.AsyncMock()) as atomic_write,
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await routes.send_message_background("conversation-1", routes.SendMessageRequest(message="restricted"), user=user)

        self.assertIs(raised.exception, error)
        persist_notice.assert_awaited_once_with("conversation-1", error)
        atomic_write.assert_not_awaited()

    async def test_unauthorized_background_route_does_not_persist_or_append_a_notice(self) -> None:
        """Moving the admission catch before access checks would disclose a conversation's notice stream."""
        user = SimpleNamespace(id="caller", role="user")

        with (
            mock.patch.object(routes, "_assert_workspace_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "check_conversation_access", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(routes, "persist_chat_denial", new=mock.AsyncMock()) as persist_notice,
            mock.patch.object(routes.repository, "add_user_message_and_create_chat_task_if_idle", new=mock.AsyncMock()) as atomic_write,
        ):
            with self.assertRaises(routes.HTTPException) as raised:
                await routes.send_message_background("conversation-1", routes.SendMessageRequest(message="restricted"), user=user)

        self.assertEqual(raised.exception.status_code, 404)
        persist_notice.assert_not_awaited()
        atomic_write.assert_not_awaited()
