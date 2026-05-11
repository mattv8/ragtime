from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import ragtime.indexer.background_tasks as background_tasks


class _HangingAsyncStream:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._never = asyncio.Event()
        self.aclose_called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        await self._never.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_called = True


class BackgroundTaskCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_task_closes_stream_iterator(self) -> None:
        service = background_tasks.BackgroundTaskService()
        stream = _HangingAsyncStream()

        fake_conversation = SimpleNamespace(
            messages=[
                SimpleNamespace(
                    role="user",
                    content="hello",
                    events=None,
                )
            ],
            user_id="user-1",
            model="lmstudio::openai/gpt-oss-20b",
            workspace_id=None,
        )

        fake_repository = SimpleNamespace(
            create_chat_task=mock.AsyncMock(),
            get_chat_task=mock.AsyncMock(return_value=SimpleNamespace(id="task-1")),
            update_chat_task_status=mock.AsyncMock(),
            get_conversation=mock.AsyncMock(return_value=fake_conversation),
            update_chat_task_streaming_state=mock.AsyncMock(),
            cancel_chat_task=mock.AsyncMock(),
            complete_chat_task=mock.AsyncMock(),
            add_message=mock.AsyncMock(),
            link_assistant_snapshot_tool_calls=mock.AsyncMock(),
            update_chat_task=mock.AsyncMock(),
        )

        fake_event_bus = SimpleNamespace(publish=mock.AsyncMock())
        fake_rag = SimpleNamespace(
            is_ready=True,
            process_query_stream=mock.Mock(return_value=stream),
        )
        fake_settings_cache = SimpleNamespace(
            get_settings=mock.AsyncMock(return_value={"max_tool_output_chars": 5000})
        )

        with (
            mock.patch.object(background_tasks, "repository", fake_repository),
            mock.patch.object(background_tasks, "task_event_bus", fake_event_bus),
            mock.patch.object(background_tasks, "rag", fake_rag),
            mock.patch.object(
                background_tasks.SettingsCache,
                "get_instance",
                return_value=fake_settings_cache,
            ),
        ):
            task_id = service.start_task(
                conversation_id="conv-1",
                user_message="hello",
                existing_task_id="task-1",
            )
            self.assertEqual(task_id, "task-1")

            running_task = service._running_tasks["task-1"]
            await asyncio.wait_for(stream.started.wait(), timeout=1.0)

            self.assertTrue(service.cancel_task("task-1"))

            with self.assertRaises(asyncio.CancelledError):
                await running_task

        self.assertTrue(stream.aclose_called)


if __name__ == "__main__":
    unittest.main()
