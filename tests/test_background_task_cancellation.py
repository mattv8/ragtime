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
        fake_settings_cache = SimpleNamespace(get_settings=mock.AsyncMock(return_value={"max_tool_output_chars": 5000}))

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


class BackgroundTaskStreamActivityTests(unittest.TestCase):
    def test_recoverable_stream_activity_includes_structured_events(self) -> None:
        self.assertTrue(
            background_tasks._has_recoverable_stream_activity(
                "",
                [{"type": "reasoning", "channel": "analysis", "content": "thinking"}],
                [],
                {},
            )
        )

    def test_recoverable_stream_activity_rejects_fully_silent_stream(self) -> None:
        self.assertFalse(
            background_tasks._has_recoverable_stream_activity(
                "",
                [],
                [],
                {},
            )
        )

    def test_mark_running_tools_completed_closes_open_tool_events(self) -> None:
        events = [
            {"type": "tool", "channel": "commentary", "tool": "query_example"},
        ]
        running_tool_indices = {"run-1": 0}

        background_tasks._mark_running_tools_completed(events, running_tool_indices)

        self.assertEqual(
            events[0]["output"],
            "Tool completed or stream ended without a final tool_end event.",
        )
        self.assertEqual(running_tool_indices, {})


class AgentStreamInactivityTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """Verify the in-component post-tool inactivity guard cancels stalled streams.

    This exercises only the wait_for + aclose pattern that the new
    ``process_query_stream`` code path relies on, which is the actual mechanism
    that prevents the upstream "model hangs after tool_end" failure mode.
    Full integration testing of ``process_query_stream`` is out of scope here
    because it depends on the live executor/LLM/tool wiring.
    """

    def test_constant_is_defined_and_positive(self) -> None:
        from ragtime.rag.components import (
            AGENT_STREAM_POST_TOOL_INACTIVITY_TIMEOUT_SECONDS,
        )

        self.assertIsInstance(AGENT_STREAM_POST_TOOL_INACTIVITY_TIMEOUT_SECONDS, float)
        self.assertGreater(AGENT_STREAM_POST_TOOL_INACTIVITY_TIMEOUT_SECONDS, 0)

    async def test_wait_for_aborts_and_closes_stalled_agent_stream(self) -> None:
        stall_event = asyncio.Event()
        closed = asyncio.Event()

        async def hanging_event_stream():
            try:
                yield {"event": "on_tool_end", "name": "query_production_infoscan_database"}
                await stall_event.wait()  # simulate provider hang after tool_end
                yield {"event": "on_chain_end"}  # never reached
            finally:
                closed.set()

        gen = hanging_event_stream()
        iterator = gen.__aiter__()

        first = await iterator.__anext__()
        self.assertEqual(first["event"], "on_tool_end")

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(iterator.__anext__(), timeout=0.05)

        aclose = getattr(iterator, "aclose", None)
        self.assertTrue(callable(aclose))
        if callable(aclose):
            close_result = aclose()
            if asyncio.iscoroutine(close_result) or isinstance(close_result, asyncio.Future):
                await close_result
        else:
            self.fail("Expected async iterator to expose aclose()")
        await asyncio.wait_for(closed.wait(), timeout=1.0)


if __name__ == "__main__":
    unittest.main()
