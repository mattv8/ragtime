import sys
import types
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from fastapi import HTTPException
from langchain_core.messages import AIMessage

_inserted_fake_indexer_service = False
if "ragtime.indexer.service" not in sys.modules:
    fake_indexer_service = types.ModuleType("ragtime.indexer.service")
    setattr(fake_indexer_service, "IndexerService", object)
    sys.modules["ragtime.indexer.service"] = fake_indexer_service
    _inserted_fake_indexer_service = True

from ragtime.rag.components import (
    RAGComponents,
    RequestLLMResolution,
    _is_userspace_workspace_not_found,
)

if _inserted_fake_indexer_service:
    inserted_module = sys.modules.get("ragtime.indexer.service")
    if "ragtime.indexer.service" in sys.modules:
        del sys.modules["ragtime.indexer.service"]
    indexer_package = sys.modules.get("ragtime.indexer")
    if indexer_package is not None and getattr(indexer_package, "service", None) is inserted_module:
        delattr(indexer_package, "service")

from ragtime.userspace.service import UserSpaceService


class ChatRuntimeErrorClassificationTests(unittest.TestCase):
    def test_platform_workspace_404_is_not_labeled_as_provider_failure(self) -> None:
        resolution = RequestLLMResolution(
            llm=object(),
            provider="github_copilot",
            model="gpt-5.4",
            attempted_providers=("github_copilot",),
        )
        exc = HTTPException(status_code=404, detail="Workspace not found")

        self.assertTrue(_is_userspace_workspace_not_found(exc))

        message = RAGComponents()._chat_runtime_error_message(exc, resolution)

        self.assertIn("workspace this chat tried to access", message)
        self.assertNotIn("GitHub Copilot", message)
        self.assertNotIn("gpt-5.4", message)

    def test_non_userspace_exception_preserves_existing_contextual_message(self) -> None:
        resolution = RequestLLMResolution(
            llm=object(),
            provider="github_copilot",
            model="gpt-5.4",
            attempted_providers=("github_copilot",),
        )
        exc = RuntimeError("404: Workspace not found")

        self.assertFalse(_is_userspace_workspace_not_found(exc))

        message = RAGComponents()._chat_runtime_error_message(exc, resolution)

        self.assertIn("I encountered an error processing your request", message)
        self.assertIn("while using GitHub Copilot for model 'gpt-5.4'", message)
        self.assertIn("404: Workspace not found", message)

    def test_unknown_error_preserves_existing_contextual_message(self) -> None:
        resolution = RequestLLMResolution(
            llm=object(),
            provider="github_copilot",
            model="gpt-5.4",
            attempted_providers=("github_copilot",),
        )

        message = RAGComponents()._chat_runtime_error_message(
            RuntimeError("boom"),
            resolution,
        )

        self.assertIn("I encountered an error processing your request", message)
        self.assertIn("while using GitHub Copilot for model 'gpt-5.4'", message)
        self.assertIn("boom", message)

    def test_userspace_runtime_hint_includes_live_data_warning(self) -> None:
        status = SimpleNamespace(
            session_state="running",
            runtime_operation_phase=None,
            devserver_running=True,
            last_error=None,
            live_data_warning="Error: column invoice_created_date does not exist",
        )

        hint = RAGComponents._build_userspace_runtime_status_turn_hint(status)

        self.assertIn("WARNING Possible live data query issue", hint)
        self.assertIn("column invoice_created_date does not exist", hint)

    def test_userspace_runtime_hint_flags_timeout_warning_with_optimization_guidance(self) -> None:
        status = SimpleNamespace(
            session_state="running",
            runtime_operation_phase=None,
            devserver_running=True,
            last_error=None,
            live_data_warning=(
                "Live data query exceeded the request timeout of 90s before a response could be returned. "
                "An admin can increase the selected tool timeout in Settings > Tools."
            ),
        )

        hint = RAGComponents._build_userspace_runtime_status_turn_hint(status)

        self.assertIn("WARNING Live data query TIMEOUT", hint)
        self.assertIn("Do NOT just retry execute-component", hint)
        self.assertIn("Optimize", hint)
        self.assertIn("narrow the WHERE clause", hint)
        # Should not fall back to the generic "Possible live data query issue" line.
        self.assertNotIn("Possible live data query issue", hint)

    def test_context_window_provider_error_is_actionable(self) -> None:
        resolution = RequestLLMResolution(
            llm=object(),
            provider="openrouter",
            model="moonshotai/kimi-k2.6",
            attempted_providers=("openrouter",),
        )
        exc = RuntimeError(
            "Error code: 400 - This endpoint's maximum context length is 262144 tokens. However, you requested about 274315 tokens. Please reduce the length."
        )

        message = RAGComponents()._chat_runtime_error_message(exc, resolution)

        self.assertIn("context window", message)
        self.assertIn("OpenRouter", message)
        self.assertIn("moonshotai/kimi-k2.6", message)
        self.assertIn("fewer or smaller file attachments", message)
        self.assertNotIn("I encountered an error processing your request", message)

    def test_payment_provider_error_uses_safe_message_and_is_not_retryable(self) -> None:
        response = SimpleNamespace(status_code=402)
        exc = RuntimeError("provider response")
        exc.response = response  # type: ignore[attr-defined]

        rag = RAGComponents()
        message = rag._chat_runtime_error_message(exc, RequestLLMResolution(llm=object(), provider="openrouter", model="model"))

        self.assertEqual(message, "The provider requires available payment credit before this request can continue.")
        self.assertFalse(rag._is_transient_provider_runtime_error(exc))

    def test_payment_error_stream_event_is_machine_readable_and_safe(self) -> None:
        exc = RuntimeError("raw provider payload must not escape")
        exc.response = SimpleNamespace(status_code=402)  # type: ignore[attr-defined]

        event = RAGComponents()._provider_error_stream_event(
            exc,
            RequestLLMResolution(llm=object(), provider="openrouter", model="model"),
        )

        self.assertEqual(event["type"], "error")
        self.assertEqual(event["code"], "payment_required")
        self.assertEqual(event["content"], "The provider requires available payment credit before this request can continue.")
        self.assertNotIn("raw provider", event["content"])


class ChatContextWindowBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_near_limit_request_caps_output_budget(self) -> None:
        rag = RAGComponents()
        user_content = "x" * 248_622

        with (
            mock.patch(
                "ragtime.rag.components.get_context_limit",
                new=mock.AsyncMock(return_value=262_144),
            ),
            mock.patch(
                "ragtime.rag.components.count_tokens",
                side_effect=lambda text: len(text),
            ),
        ):
            fit = await rag._fit_chat_request_context_window(
                provider="openrouter",
                model="moonshotai/kimi-k2.6",
                requested_max_tokens=16_384,
                system_prompt="",
                tool_scope_prompt="",
                turn_system_content="",
                chat_history=[],
                user_content=user_content,
                tools=[],
            )

        self.assertLess(fit.max_tokens, 16_384)
        self.assertGreaterEqual(fit.max_tokens, 128)
        self.assertIn("Capped this turn's response budget", fit.notice)

    async def test_too_large_request_raises_friendly_context_error(self) -> None:
        rag = RAGComponents()

        with (
            mock.patch(
                "ragtime.rag.components.get_context_limit",
                new=mock.AsyncMock(return_value=1_000),
            ),
            mock.patch(
                "ragtime.rag.components.count_tokens",
                side_effect=lambda text: len(text),
            ),
        ):
            with self.assertRaisesRegex(Exception, "too close to its context window"):
                await rag._fit_chat_request_context_window(
                    provider="openrouter",
                    model="tiny-context-model",
                    requested_max_tokens=512,
                    system_prompt="",
                    tool_scope_prompt="",
                    turn_system_content="",
                    chat_history=[],
                    user_content="x" * 5_000,
                    tools=[],
                )

    async def test_openrouter_thinking_budget_is_capped_to_request_output(self) -> None:
        rag = RAGComponents()
        rag._app_settings = {"openrouter_api_key": "openrouter-key"}

        with (
            mock.patch(
                "ragtime.rag.components.httpx.AsyncClient.get",
                new=mock.AsyncMock(side_effect=RuntimeError("metadata fetch disabled in test")),
            ),
            mock.patch(
                "ragtime.rag.components.supports_reasoning_effort",
                new=mock.AsyncMock(return_value=False),
            ),
            mock.patch(
                "ragtime.rag.components.supports_reasoning",
                new=mock.AsyncMock(return_value=False),
            ),
            mock.patch(
                "ragtime.rag.components.supports_thinking_budget",
                new=mock.AsyncMock(return_value=True),
            ),
            mock.patch(
                "ragtime.rag.components._CopilotChatOpenAI",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            llm_kwargs = await rag._build_llm("openrouter", "moonshotai/kimi-k2.6", 4_096)

        self.assertIsInstance(llm_kwargs, dict)
        llm_kwargs = cast(dict[str, Any], llm_kwargs)
        self.assertEqual(llm_kwargs["max_tokens"], 4_096)
        self.assertEqual(llm_kwargs["extra_body"]["thinking_budget"], 4_096)


class _TwoRoundExecutor:
    tools: list[object] = []

    def __init__(self) -> None:
        self.calls = 0

    def astream_events(self, *_args: object, **_kwargs: object):
        self.calls += 1

        async def stream():
            if self.calls == 1:
                yield {"event": "on_tool_start", "name": "write_file", "run_id": "tool-1", "data": {"input": {"path": "app.py"}}}
                yield {"event": "on_tool_end", "name": "write_file", "run_id": "tool-1", "data": {"output": '{"persisted": true}'}}
            else:
                yield {"event": "on_chat_model_stream", "run_id": "chat-2", "data": {"chunk": AIMessage(content="second-round final response")}}

        return stream()


class _EmptyExecutor:
    tools: list[object] = []

    def astream_events(self, *_args: object, **_kwargs: object):
        async def stream():
            if False:  # pragma: no cover - marks this as an async generator
                yield {}

        return stream()


class _PaymentFailureLLM:
    def __init__(self) -> None:
        self.calls = 0

    def astream(self, _messages: object):
        self.calls += 1

        async def stream():
            error = RuntimeError("provider payment payload")
            error.response = SimpleNamespace(status_code=402)  # type: ignore[attr-defined]
            raise error
            yield None  # pragma: no cover - marks this as an async generator

        return stream()


class MultiRoundStreamTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _request_context() -> dict[str, object]:
        return {
            "prompt_is_ui": True,
            "mode": "userspace",
            "allowed_tool_config_ids": [],
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
            "tool_skill_has_loadable": False,
            "tool_skill_binding_state": None,
            "tool_skill_hidden_ids": set(),
            "tool_skill_loaded_ids": [],
        }

    async def test_tool_round_continues_to_a_second_agent_round_before_final_text(self) -> None:
        rag = RAGComponents()
        executor = _TwoRoundExecutor()
        rag.agent_executor_ui = executor
        resolution = RequestLLMResolution(llm=object(), provider="openrouter", model="model")
        request_context = self._request_context()

        with (
            mock.patch.object(rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=resolution)),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(return_value=request_context)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value=""),
            mock.patch.object(rag, "_prepare_chat_context_window", new=mock.AsyncMock(return_value=(resolution, [], ""))),
            mock.patch.object(rag, "_build_runtime_executor", return_value=executor),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()),
            mock.patch.object(rag, "_seed_tool_skill_request_state"),
        ):
            events = [event async for event in rag.process_query_stream("continue", is_ui=True)]

        self.assertEqual(executor.calls, 2)
        self.assertEqual(events[0]["type"], "tool_start")
        self.assertEqual(events[1]["type"], "tool_end")
        self.assertIn("second-round final response", events)

    async def test_payment_failure_in_tool_free_synthesis_emits_terminal_error_event(self) -> None:
        rag = RAGComponents()
        executor = _EmptyExecutor()
        llm = _PaymentFailureLLM()
        rag.agent_executor_ui = executor
        resolution = RequestLLMResolution(llm=llm, provider="openrouter", model="model")

        with (
            mock.patch.object(rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=resolution)),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(return_value=self._request_context())),
            mock.patch.object(rag, "_build_request_system_prompt", return_value=""),
            mock.patch.object(rag, "_prepare_chat_context_window", new=mock.AsyncMock(return_value=(resolution, [], ""))),
            mock.patch.object(rag, "_build_runtime_executor", return_value=executor),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()),
            mock.patch.object(rag, "_seed_tool_skill_request_state"),
        ):
            events = [event async for event in rag.process_query_stream("continue", is_ui=True)]

        self.assertEqual(llm.calls, 1)
        self.assertEqual(events, [{"type": "error", "code": "payment_required", "content": "The provider requires available payment credit before this request can continue."}])


class CrossWorkspaceResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_granted_target_without_user_access_becomes_tool_denial(self) -> None:
        service = UserSpaceService()

        async def deny_workspace_access(*_args, **_kwargs):
            raise HTTPException(status_code=404, detail="Workspace not found")

        service._enforce_workspace_access = deny_workspace_access  # type: ignore[method-assign]

        with self.assertRaisesRegex(ValueError, "current user"):
            await service.resolve_cross_workspace_target(
                source_workspace_id="source-workspace",
                target_workspace_id="target-workspace",
                user_id="user-id",
                accessible_modes={"target-workspace": "read"},
                action="read",
            )

    async def test_admin_bypasses_missing_cross_workspace_grant(self) -> None:
        service = UserSpaceService()
        calls: list[dict[str, Any]] = []

        async def allow_workspace_access(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

        service._enforce_workspace_access = allow_workspace_access  # type: ignore[method-assign]
        service._record_runtime_audit_event = mock.AsyncMock()  # type: ignore[method-assign]

        resolved = await service.resolve_cross_workspace_target(
            source_workspace_id="source-workspace",
            target_workspace_id="target-workspace",
            user_id="admin-user-id",
            accessible_modes={},
            action="write",
            is_admin=True,
        )

        self.assertEqual(resolved, ("target-workspace", "admin-user-id"))
        self.assertEqual(calls[0]["kwargs"].get("required_role"), "editor")
        self.assertTrue(calls[0]["kwargs"].get("is_admin"))


if __name__ == "__main__":
    unittest.main()
