import sys
import types
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from fastapi import HTTPException

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


if __name__ == "__main__":
    unittest.main()
