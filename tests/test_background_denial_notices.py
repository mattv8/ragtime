from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from langchain_core.messages import AIMessage

from ragtime.content_protection.models import ContentProtectionError
from ragtime.indexer import background_tasks
from ragtime.indexer.models import ChatMessage, ChatTaskStatus


class BackgroundDenialNoticeTests(IsolatedAsyncioTestCase):
    async def test_protection_failure_fails_task_even_when_notice_storage_fails(self) -> None:
        """Coupling terminal status to notice storage would leave a denied task running."""
        task_id = "task-1"
        conversation = SimpleNamespace(id="conversation-1", user_id="owner", workspace_id=None, model="test", messages=[])
        error = ContentProtectionError("content_denied", "request-1")

        async def denied_stream(*_args, **_kwargs):
            raise error
            yield None

        service = background_tasks.BackgroundTaskService()
        with (
            mock.patch.object(background_tasks.repository, "get_conversation", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(background_tasks.repository, "get_chat_task", new=mock.AsyncMock(return_value=SimpleNamespace(execution_policy=None))),
            mock.patch.object(background_tasks.repository, "update_chat_task_status", new=mock.AsyncMock()) as update_status,
            mock.patch.object(background_tasks, "persist_chat_denial", new=mock.AsyncMock(side_effect=RuntimeError("storage unavailable"))),
            mock.patch.object(background_tasks.SettingsCache, "get_instance", return_value=SimpleNamespace(get_settings=mock.AsyncMock(return_value={}))),
            mock.patch.object(background_tasks, "require_generation", new=mock.AsyncMock()),
            mock.patch.object(background_tasks, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(background_tasks, "authorize_history", new=mock.AsyncMock()),
            mock.patch.object(background_tasks, "rag", SimpleNamespace(is_ready=True, process_query_stream=denied_stream)),
            mock.patch.object(background_tasks.task_event_bus, "publish", new=mock.AsyncMock()),
        ):
            service.start_task("conversation-1", "allowed", existing_task_id=task_id)
            await service._running_tasks[task_id]

        failed_call = next(call for call in update_status.await_args_list if call.args[1] == ChatTaskStatus.failed)
        self.assertEqual(failed_call.kwargs["response_content"], error.public_detail()["message"])
        self.assertTrue(failed_call.kwargs["clear_streaming_state"])
        self.assertEqual(failed_call.kwargs["outcome_summary"]["refusal"], error.public_detail())

    async def test_background_history_excludes_typed_denial_notices(self) -> None:
        """Treating a denial notice as an assistant turn would feed its safe text to the model."""
        task_id = "task-1"
        notice = ChatMessage(
            role="assistant",
            content="Request blocked.",
            events=[{"type": "content_protection_denial", "channel": "final"}],
        )
        conversation = SimpleNamespace(
            id="conversation-1",
            user_id="owner",
            workspace_id=None,
            model="test",
            messages=[notice, ChatMessage(role="user", content="allowed")],
        )
        captured_history = []

        async def completed_stream(_message, history, **_kwargs):
            captured_history.extend(history)
            yield "safe response"

        service = background_tasks.BackgroundTaskService()
        with (
            mock.patch.object(background_tasks.repository, "get_conversation", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(background_tasks.repository, "get_chat_task", new=mock.AsyncMock(return_value=SimpleNamespace(execution_policy=None))),
            mock.patch.object(background_tasks.repository, "update_chat_task_status", new=mock.AsyncMock()),
            mock.patch.object(background_tasks.repository, "add_message", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(background_tasks.repository, "complete_chat_task", new=mock.AsyncMock()),
            mock.patch.object(background_tasks.repository, "link_assistant_snapshot_tool_calls", new=mock.AsyncMock()),
            mock.patch.object(background_tasks.SettingsCache, "get_instance", return_value=SimpleNamespace(get_settings=mock.AsyncMock(return_value={}))),
            mock.patch.object(background_tasks, "require_generation", new=mock.AsyncMock()),
            mock.patch.object(background_tasks, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(background_tasks, "authorize_history", new=mock.AsyncMock()),
            mock.patch.object(background_tasks, "rag", SimpleNamespace(is_ready=True, process_query_stream=completed_stream)),
            mock.patch.object(background_tasks.task_event_bus, "publish", new=mock.AsyncMock()),
        ):
            service.start_task("conversation-1", "allowed", existing_task_id=task_id)
            await service._running_tasks[task_id]

        self.assertFalse(any(isinstance(message, AIMessage) and message.content == "Request blocked." for message in captured_history))
