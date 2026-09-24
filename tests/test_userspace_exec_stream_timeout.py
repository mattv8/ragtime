import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from ragtime.rag.components import RAGComponents
from tests.content_protection_support import use_disabled_content_protection
from tests.generation_policy_test_support import enabled_generation_policy


class _EventExecutor:
    tools: list[Any] = []

    def astream_events(self, _input, version, config):
        assert version == "v2"
        assert config is not None

        async def events():
            yield {
                "event": "on_tool_start",
                "name": "run_terminal_command",
                "run_id": "terminal-run",
                "data": {"input": {"command": "npm run build"}},
            }
            yield {
                "event": "on_tool_end",
                "name": "run_terminal_command",
                "run_id": "terminal-run",
                "data": {"output": '{"exit_code": 0}'},
            }
            yield {
                "event": "on_chat_model_stream",
                "run_id": "chat-run",
                "data": {"chunk": SimpleNamespace(content="done", tool_call_chunks=[])},
            }

        return events()


class UserSpaceExecStreamTimeoutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        use_disabled_content_protection(self)

    async def test_omitted_terminal_timeout_survives_admin_shrink_but_post_tool_uses_normal_guard(self) -> None:
        rag = RAGComponents()
        executor = _EventExecutor()
        llm_resolution = SimpleNamespace(llm=object(), model="test-model", provider="test")
        request_context = {
            "prompt_is_ui": False,
            "mode": "userspace",
            "allowed_tool_config_ids": set(),
            "runtime_tools": [],
            "request_tool_state": {},
            "prompt_additions": "",
            "user_identity_turn_line": "",
            "current_time_turn_line": "",
            "include_sqlite_persistence": False,
            "userspace_env_var_turn_hint": "",
            "userspace_runtime_status_turn_hint": "",
            "userspace_diagnostics_turn_hint": "",
            "tool_skill_mode": "disabled",
        }
        observed_timeouts = []
        original_wait_for = __import__("asyncio").wait_for

        async def capture_wait_for(awaitable, *, timeout):
            observed_timeouts.append(timeout)
            return await original_wait_for(awaitable, timeout=timeout)

        with (
            enabled_generation_policy(surface="userspace"),
            mock.patch.object(rag, "agent_executor", executor),
            # The terminal began when 1800 was allowed, then the administrator
            # lowered the maximum before the event-stream watchdog armed.
            mock.patch(
                "ragtime.rag.components.get_app_settings",
                new=mock.AsyncMock(return_value={"userspace_exec_timeout_default_seconds": 120, "userspace_exec_timeout_max_seconds": 600}),
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="hello")),
            mock.patch.object(rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=llm_resolution)),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda value, *_args, **_kwargs: value)),
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(return_value=request_context)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value=""),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_prepare_chat_context_window", new=mock.AsyncMock(return_value=(llm_resolution, [], ""))),
            mock.patch.object(rag, "_build_runtime_executor", return_value=executor),
            mock.patch.object(rag, "_has_image_content", return_value=False),
            mock.patch("ragtime.rag.components.asyncio.wait_for", new=capture_wait_for),
        ):
            stream = rag.process_query_stream("hello")
            first = await anext(stream)
            second = await anext(stream)
            third = await anext(stream)
            await stream.aclose()

        first_event = cast(dict[str, Any], first)
        second_event = cast(dict[str, Any], second)
        self.assertEqual(first_event["type"], "tool_start")
        self.assertEqual(second_event["type"], "tool_end")
        self.assertEqual(third, "done")
        self.assertEqual(observed_timeouts, [3645.0, 315.0])

    async def test_concurrent_terminal_guards_cleanup_independently(self) -> None:
        class ConcurrentExecutor(_EventExecutor):
            def astream_events(self, _input, version, config):
                assert version == "v2"
                assert config is not None

                async def events():
                    for event in (
                        {"event": "on_tool_start", "name": "run_terminal_command", "run_id": "long", "data": {"input": {"timeout_seconds": 1800}}},
                        # Malformed input cannot inflate the guard beyond the
                        # current configured maximum after the long run ends.
                        {"event": "on_tool_start", "name": "run_terminal_command", "run_id": "short", "data": {"input": {"timeout_seconds": "not-a-timeout"}}},
                        {"event": "on_tool_end", "name": "run_terminal_command", "run_id": "long", "data": {"output": "{}"}},
                        {"event": "on_chat_model_stream", "run_id": "chat", "data": {"chunk": SimpleNamespace(content="done", tool_call_chunks=[])}},
                    ):
                        yield event

                return events()

        rag = RAGComponents()
        executor = ConcurrentExecutor()
        executor_patch = mock.patch.object(rag, "agent_executor", executor)
        resolution = SimpleNamespace(llm=object(), model="test-model", provider="test")
        context = {
            "prompt_is_ui": False,
            "mode": "userspace",
            "allowed_tool_config_ids": set(),
            "runtime_tools": [],
            "request_tool_state": {},
            "prompt_additions": "",
            "user_identity_turn_line": "",
            "current_time_turn_line": "",
            "include_sqlite_persistence": False,
            "userspace_env_var_turn_hint": "",
            "userspace_runtime_status_turn_hint": "",
            "userspace_diagnostics_turn_hint": "",
            "tool_skill_mode": "disabled",
        }
        observed = []
        original_wait_for = __import__("asyncio").wait_for

        async def capture(awaitable, *, timeout):
            observed.append(timeout)
            return await original_wait_for(awaitable, timeout=timeout)

        with (
            enabled_generation_policy(surface="userspace"),
            executor_patch,
            mock.patch(
                "ragtime.rag.components.get_app_settings",
                new=mock.AsyncMock(return_value={"userspace_exec_timeout_default_seconds": 120, "userspace_exec_timeout_max_seconds": 600}),
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="hello")),
            mock.patch.object(rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=resolution)),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda value, *_args, **_kwargs: value)),
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(return_value=context)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value=""),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_prepare_chat_context_window", new=mock.AsyncMock(return_value=(resolution, [], ""))),
            mock.patch.object(rag, "_build_runtime_executor", return_value=executor),
            mock.patch.object(rag, "_has_image_content", return_value=False),
            mock.patch("ragtime.rag.components.asyncio.wait_for", new=capture),
        ):
            stream = rag.process_query_stream("hello")
            for _ in range(4):
                await anext(stream)
            await stream.aclose()

        self.assertEqual(observed, [1845.0, 1845.0, 645.0])
