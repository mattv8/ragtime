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


class _ActiveToolTimeoutExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = asyncio.Event()
        self.tools: list[object] = []

    def astream_events(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return self._stalled_tool_stream()
        return self._final_text_stream()

    async def _stalled_tool_stream(self):
        try:
            yield {
                "event": "on_tool_start",
                "run_id": "run-1",
                "name": "query_production_infoscan_database",
                "data": {"input": {"query": "select pg_sleep(999)", "timeout": 1}},
            }
            await asyncio.Event().wait()
        finally:
            self.closed.set()

    async def _final_text_stream(self):
        yield {
            "event": "on_chat_model_stream",
            "run_id": "model-1",
            "data": {"chunk": SimpleNamespace(content="The query tool timed out.")},
        }


class _ProviderFailureAfterToolExecutor:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.tools: list[object] = []

    def astream_events(self, *_args, **_kwargs):
        return self._stream()

    async def _stream(self):
        try:
            yield {
                "event": "on_tool_start",
                "run_id": "run-1",
                "name": "read_userspace_file",
                "data": {"input": {"path": "dashboard/main.ts"}},
            }
            yield {
                "event": "on_tool_end",
                "run_id": "run-1",
                "name": "read_userspace_file",
                "data": {"output": "export default function Dashboard() { return null; }"},
            }
            raise RuntimeError("Upstream idle timeout exceeded")
        finally:
            self.closed.set()


class _SynthesisLLM:
    async def astream(self, _messages):
        yield SimpleNamespace(content="Recovered final answer.")


class _FirstChunkRetryLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def astream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Upstream idle timeout exceeded")
        yield SimpleNamespace(content="Retried answer.")


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

    def test_contextual_provider_error_is_wrapped_processing_error(self) -> None:
        detail = background_tasks._extract_wrapped_processing_error(
            "I encountered an error processing your request while using OpenRouter for model 'deepseek/deepseek-v4-pro': Upstream idle timeout exceeded"
        )

        self.assertEqual(
            detail,
            "while using OpenRouter for model 'deepseek/deepseek-v4-pro': Upstream idle timeout exceeded",
        )


class AgentStreamInactivityTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """Verify the in-component post-tool inactivity guard cancels stalled streams.

    This exercises only the wait_for + aclose pattern that the new
    ``process_query_stream`` code path relies on, which is the actual mechanism
    that prevents the upstream "model hangs after tool_end" failure mode.
    Full integration testing of ``process_query_stream`` is out of scope here
    because it depends on the live executor/LLM/tool wiring.
    """

    def test_constant_is_defined_and_positive(self) -> None:
        from ragtime.rag.components import AGENT_STREAM_INACTIVITY_TIMEOUT_SECONDS

        self.assertIsInstance(AGENT_STREAM_INACTIVITY_TIMEOUT_SECONDS, float)
        self.assertGreater(AGENT_STREAM_INACTIVITY_TIMEOUT_SECONDS, 0)

    def test_active_tool_timeout_output_is_agent_actionable(self) -> None:
        from ragtime.rag import components

        output = components._format_active_tool_timeout_output("query_production_infoscan_database")

        self.assertIn("Tool query_production_infoscan_database took too long and timed out", output)
        self.assertIn("Treat this as a failed tool result", output)

    async def test_process_query_stream_synthesizes_active_tool_timeout_result(self) -> None:
        from ragtime.rag import components

        rag_components = components.RAGComponents()
        executor = _ActiveToolTimeoutExecutor()
        rag_components.agent_executor_ui = executor  # type: ignore[assignment]
        request_tool_state: dict[str, object] = {}
        request_context = {
            "prompt_is_ui": True,
            "mode": "chat",
            "allowed_tool_config_ids": set(),
            "runtime_tools": [],
            "request_tool_state": request_tool_state,
            "prompt_additions": "",
            "user_identity_turn_line": "",
            "include_sqlite_persistence": False,
            "userspace_env_var_turn_hint": "",
            "userspace_runtime_status_turn_hint": "",
            "workspace_id": None,
        }
        llm_resolution = components.RequestLLMResolution(
            llm=object(),
            provider="test_provider",
            model="test-model",
        )

        async def collect_outputs() -> list[object]:
            outputs: list[object] = []
            async for item in rag_components.process_query_stream("hello", is_ui=True):
                outputs.append(item)
            return outputs

        with (
            mock.patch.object(rag_components, "_convert_message_to_langchain_async", mock.AsyncMock(return_value="hello")),
            mock.patch.object(rag_components, "_get_request_scoped_llm", mock.AsyncMock(return_value=llm_resolution)),
            mock.patch.object(rag_components, "_build_request_runtime_context", mock.AsyncMock(return_value=request_context)),
            mock.patch.object(rag_components, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag_components, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag_components, "_build_context_headroom_prompt", mock.AsyncMock(return_value="")),
            mock.patch.object(rag_components, "_prepare_chat_context_window", mock.AsyncMock(return_value=(llm_resolution, [], ""))),
            mock.patch.object(rag_components, "_build_runtime_executor", return_value=executor),
            mock.patch.object(
                rag_components,
                "_get_tool_connection_metadata",
                return_value={"timeout_max_seconds": "1"},
            ),
            mock.patch.object(rag_components, "_persist_provider_prompt_debug_record", mock.AsyncMock()),
            mock.patch.object(components, "AGENT_STREAM_INACTIVITY_TIMEOUT_SECONDS", 0.05),
        ):
            outputs = await asyncio.wait_for(collect_outputs(), timeout=1.0)

        self.assertTrue(executor.closed.is_set())
        self.assertIsInstance(outputs[0], dict)
        self.assertIsInstance(outputs[1], dict)
        tool_start = outputs[0]
        tool_end = outputs[1]
        assert isinstance(tool_start, dict)
        assert isinstance(tool_end, dict)
        self.assertEqual(tool_start["type"], "tool_start")
        self.assertEqual(tool_end["type"], "tool_end")
        self.assertIn("took too long and timed out", tool_end["output"])
        self.assertIn("The query tool timed out.", outputs)
        self.assertTrue(request_tool_state["active_tool_stream_timed_out"])

    async def test_process_query_stream_recovers_openrouter_idle_timeout_after_tool_result(self) -> None:
        from ragtime.rag import components

        rag_components = components.RAGComponents()
        executor = _ProviderFailureAfterToolExecutor()
        rag_components.agent_executor_ui = executor  # type: ignore[assignment]
        request_tool_state: dict[str, object] = {}
        request_context = {
            "prompt_is_ui": True,
            "mode": "chat",
            "allowed_tool_config_ids": set(),
            "runtime_tools": [],
            "request_tool_state": request_tool_state,
            "prompt_additions": "",
            "user_identity_turn_line": "",
            "include_sqlite_persistence": False,
            "userspace_env_var_turn_hint": "",
            "userspace_runtime_status_turn_hint": "",
            "workspace_id": None,
        }
        llm_resolution = components.RequestLLMResolution(
            llm=_SynthesisLLM(),
            provider="openrouter",
            model="deepseek/deepseek-v4-pro",
            attempted_providers=("openrouter",),
        )

        async def collect_outputs() -> list[object]:
            outputs: list[object] = []
            async for item in rag_components.process_query_stream("hello", is_ui=True):
                outputs.append(item)
            return outputs

        with (
            mock.patch.object(rag_components, "_convert_message_to_langchain_async", mock.AsyncMock(return_value="hello")),
            mock.patch.object(rag_components, "_get_request_scoped_llm", mock.AsyncMock(return_value=llm_resolution)),
            mock.patch.object(rag_components, "_build_request_runtime_context", mock.AsyncMock(return_value=request_context)),
            mock.patch.object(rag_components, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag_components, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag_components, "_build_context_headroom_prompt", mock.AsyncMock(return_value="")),
            mock.patch.object(rag_components, "_prepare_chat_context_window", mock.AsyncMock(return_value=(llm_resolution, [], ""))),
            mock.patch.object(rag_components, "_build_runtime_executor", return_value=executor),
            mock.patch.object(rag_components, "_persist_provider_prompt_debug_record", mock.AsyncMock()),
        ):
            outputs = await asyncio.wait_for(collect_outputs(), timeout=1.0)

        self.assertTrue(executor.closed.is_set())
        tool_start = outputs[0]
        tool_end = outputs[1]
        assert isinstance(tool_start, dict)
        assert isinstance(tool_end, dict)
        self.assertEqual(tool_start["type"], "tool_start")
        self.assertEqual(tool_end["type"], "tool_end")
        self.assertIn("Recovered final answer.", outputs)
        self.assertTrue(request_tool_state["provider_stream_failed"])
        self.assertEqual(request_tool_state["internal_continue_stop_reason"], "provider_stream_failed")

    async def test_direct_stream_retries_first_chunk_openrouter_idle_timeout(self) -> None:
        from ragtime.rag import components

        rag_components = components.RAGComponents()
        request_llm = _FirstChunkRetryLLM()
        request_tool_state: dict[str, object] = {}
        request_context = {
            "prompt_is_ui": False,
            "mode": "chat",
            "allowed_tool_config_ids": set(),
            "runtime_tools": [],
            "request_tool_state": request_tool_state,
            "prompt_additions": "",
            "user_identity_turn_line": "",
            "include_sqlite_persistence": False,
            "userspace_env_var_turn_hint": "",
            "userspace_runtime_status_turn_hint": "",
            "workspace_id": None,
        }
        llm_resolution = components.RequestLLMResolution(
            llm=request_llm,
            provider="openrouter",
            model="deepseek/deepseek-v4-pro",
            attempted_providers=("openrouter",),
        )

        async def collect_outputs() -> list[object]:
            outputs: list[object] = []
            async for item in rag_components.process_query_stream("hello", is_ui=False):
                outputs.append(item)
            return outputs

        with (
            mock.patch.object(rag_components, "_convert_message_to_langchain_async", mock.AsyncMock(return_value="hello")),
            mock.patch.object(rag_components, "_get_request_scoped_llm", mock.AsyncMock(return_value=llm_resolution)),
            mock.patch.object(rag_components, "_build_request_runtime_context", mock.AsyncMock(return_value=request_context)),
            mock.patch.object(rag_components, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag_components, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag_components, "_build_context_headroom_prompt", mock.AsyncMock(return_value="")),
            mock.patch.object(rag_components, "_prepare_chat_context_window", mock.AsyncMock(return_value=(llm_resolution, [], ""))),
            mock.patch.object(rag_components, "_persist_provider_prompt_debug_record", mock.AsyncMock()),
        ):
            outputs = await asyncio.wait_for(collect_outputs(), timeout=1.0)

        self.assertEqual(request_llm.calls, 2)
        self.assertEqual(outputs, ["Retried answer."])

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
