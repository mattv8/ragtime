"""Focused outcome-policy tests for durable background chat tasks."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

import ragtime.indexer.background_tasks as background_tasks
from ragtime.content_protection.models import ContentProtectionError
from ragtime.indexer.task_policy import activity_summary, make_execution_policy, required_action_termination
from tests.content_protection_support import use_disabled_content_protection
from tests.generation_policy_test_support import enabled_generation_policy


class BackgroundTaskOutcomePolicyTests(unittest.TestCase):
    def test_build_without_executed_tools_is_interrupted(self) -> None:
        policy = make_execution_policy("build", source="workspace_agent")
        activity = activity_summary([])

        self.assertEqual(required_action_termination(policy, activity, False), "no_actions")

    def test_general_text_only_task_remains_completed(self) -> None:
        policy = make_execution_policy("general", source="workspace_agent")
        activity = activity_summary([])

        self.assertIsNone(required_action_termination(policy, activity, False))

    def test_synthetic_tool_recovery_does_not_count_as_action(self) -> None:
        activity = activity_summary([{"tool": "recovery", "synthetic": True}])

        self.assertEqual(activity, {"attempted": 0, "succeeded": 0, "failed": 0})

    def test_all_failed_executions_are_interrupted(self) -> None:
        policy = make_execution_policy("build", source="workspace_agent")
        activity = activity_summary(
            [
                {"tool": "write_file", "failed": True},
                {"tool": "run_command", "success": False},
            ]
        )

        self.assertEqual(activity, {"attempted": 2, "succeeded": 0, "failed": 2})
        self.assertEqual(required_action_termination(policy, activity, False), "all_tools_failed")

    def test_max_iterations_overrides_other_build_outcomes(self) -> None:
        policy = make_execution_policy("build", source="workspace_agent")

        self.assertEqual(required_action_termination(policy, {"attempted": 1, "succeeded": 1, "failed": 0}, True), "max_iterations")


class BackgroundTaskOutcomeExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        use_disabled_content_protection(self)

    def _dependencies(self, stream, *, model="openai::test"):
        conversation = SimpleNamespace(
            messages=[SimpleNamespace(role="user", content="build it", events=None)],
            user_id="user-1",
            model=model,
            workspace_id="ws-1",
        )
        repository = SimpleNamespace(
            get_chat_task=mock.AsyncMock(return_value=SimpleNamespace(id="task-1")),
            update_chat_task_status=mock.AsyncMock(),
            get_conversation=mock.AsyncMock(return_value=conversation),
            update_chat_task_streaming_state=mock.AsyncMock(return_value=None),
            add_message=mock.AsyncMock(return_value=SimpleNamespace(workspace_id="ws-1")),
            link_assistant_snapshot_tool_calls=mock.AsyncMock(),
            complete_chat_task=mock.AsyncMock(),
            cancel_chat_task=mock.AsyncMock(),
        )
        rag = SimpleNamespace(is_ready=True, process_query_stream=mock.Mock(return_value=stream))
        settings = SimpleNamespace(get_settings=mock.AsyncMock(return_value={"max_tool_output_chars": 5000}))
        return repository, rag, settings

    async def _run(self, stream, policy, *, usage_attempt_id=None, model="openai::test", settings_values=None):
        service = background_tasks.BackgroundTaskService()
        repository, rag, settings = self._dependencies(stream, model=model)
        settings.get_settings = mock.AsyncMock(return_value=settings_values or {"max_tool_output_chars": 5000})
        bus = SimpleNamespace(publish=mock.AsyncMock())
        with (
            enabled_generation_policy("user-1"),
            mock.patch.object(background_tasks, "repository", repository),
            mock.patch.object(background_tasks, "rag", rag),
            mock.patch.object(background_tasks, "task_event_bus", bus),
            mock.patch.object(background_tasks.SettingsCache, "get_instance", return_value=settings),
            mock.patch.object(background_tasks, "finalize_usage_attempt", mock.AsyncMock()) as finalize_usage,
        ):
            service.start_task("conv-1", "build it", existing_task_id="task-1", execution_policy=policy, usage_attempt_id=usage_attempt_id)
            await asyncio.wait_for(service._running_tasks["task-1"], timeout=1)
        return repository, finalize_usage

    async def test_early_content_protection_denial_fails_task_and_finalizes_usage(self) -> None:
        denial = ContentProtectionError("content_denied", "request-1", reason="Restricted input.")

        with mock.patch.object(background_tasks, "authorize_inbound", new=mock.AsyncMock(side_effect=denial)):
            repository, usage = await self._run(
                None,
                make_execution_policy("general", source="workspace_agent"),
                usage_attempt_id="usage-1",
            )

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.args[1].value, "failed")
        self.assertEqual(update.kwargs["termination_reason"], "content_denied")
        self.assertEqual(update.kwargs["outcome_summary"]["refusal"]["code"], "content_denied")
        usage.assert_awaited_once_with(
            "usage-1",
            status="failed",
            failure_reason="content_denied",
            output_tokens=0,
        )

    async def test_structured_payment_error_persists_partial_and_closes_usage(self) -> None:
        async def stream():
            yield "partial work"
            yield {"type": "error", "code": "payment_required", "content": "Payment required."}

        with mock.patch("ragtime.core.openrouter_credits.note_openrouter_payment_required", return_value="Credit warning"):
            repository, usage = await self._run(
                stream(), make_execution_policy("build", source="workspace_agent"), usage_attempt_id="usage-1", model="openrouter::test"
            )

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.args[1].value, "failed")
        self.assertEqual(update.kwargs["termination_reason"], "payment_required")
        self.assertEqual(update.kwargs["response_content"], "partial work")
        self.assertEqual(update.kwargs["outcome_summary"]["warnings"], ["Credit warning"])
        repository.link_assistant_snapshot_tool_calls.assert_awaited_once()
        usage.assert_awaited_once()

    async def test_generic_payment_stream_fails_without_openrouter_credit_side_effect(self) -> None:
        async def stream():
            yield {"type": "error", "code": "payment_required", "content": "Payment required.", "provider": "openai_compatible"}

        with mock.patch("ragtime.core.openrouter_credits.note_openrouter_payment_required") as note:
            repository, _usage = await self._run(stream(), make_execution_policy("build", source="workspace_agent"), model="openai_compatible::same-id")

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.args[1].value, "failed")
        self.assertEqual(update.kwargs["termination_reason"], "payment_required")
        self.assertEqual(update.kwargs["outcome_summary"]["warnings"], [])
        note.assert_not_called()

    async def test_generic_payment_exception_does_not_change_openrouter_credit_state(self) -> None:
        error = httpx.HTTPStatusError(
            "raw upstream payment body",
            request=httpx.Request("POST", "https://compatible.test/chat/completions"),
            response=httpx.Response(402, json={}),
        )

        async def stream():
            raise error
            yield  # pragma: no cover - marks this as an async generator

        with mock.patch("ragtime.core.openrouter_credits.note_openrouter_payment_required") as note:
            repository, _usage = await self._run(stream(), make_execution_policy("build", source="workspace_agent"), model="openai_compatible::same-id")

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.args[1].value, "failed")
        self.assertEqual(update.kwargs["termination_reason"], "payment_required")
        note.assert_not_called()

    async def test_generic_structured_payment_exception_preserves_shared_classification(self) -> None:
        error = httpx.HTTPStatusError(
            "raw upstream payment body",
            request=httpx.Request("POST", "https://compatible.test/chat/completions"),
            response=httpx.Response(400, json={"error": {"type": "insufficient_credit"}}),
        )

        async def stream():
            raise error
            yield  # pragma: no cover - marks this as an async generator

        with mock.patch("ragtime.core.openrouter_credits.note_openrouter_payment_required") as note:
            repository, _usage = await self._run(stream(), make_execution_policy("build", source="workspace_agent"), model="openai_compatible::same-id")

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.args[1].value, "failed")
        self.assertEqual(update.kwargs["termination_reason"], "payment_required")
        note.assert_not_called()

    async def test_bare_openrouter_payment_uses_configured_provider_for_credit_warning(self) -> None:
        async def stream():
            yield {"type": "error", "code": "payment_required", "content": "Payment required."}

        with mock.patch("ragtime.core.openrouter_credits.note_openrouter_payment_required", return_value="Credit warning"):
            repository, _usage = await self._run(
                stream(), make_execution_policy("build", source="workspace_agent"), model="same-id", settings_values={"llm_provider": "openrouter"}
            )

        self.assertEqual(repository.update_chat_task_status.await_args.kwargs["outcome_summary"]["warnings"], ["Credit warning"])

    async def test_bare_generic_platform_exception_stays_unclassified(self) -> None:
        async def stream():
            raise RuntimeError("platform implementation detail")
            yield  # pragma: no cover

        repository, _usage = await self._run(
            stream(), make_execution_policy("general", source="workspace_agent"), model="same-id", settings_values={"llm_provider": "openai_compatible"}
        )
        self.assertIsNone(repository.update_chat_task_status.await_args.kwargs["termination_reason"])

    async def test_generic_safe_rate_error_is_terminal_not_advisory_success(self) -> None:
        async def stream():
            yield {
                "type": "error",
                "code": "rate_limited",
                "content": "The configured OpenAI-compatible provider is rate limited. Please retry shortly.",
                "provider": "openai_compatible",
            }

        repository, _usage = await self._run(stream(), make_execution_policy("general", source="workspace_agent"), model="openai_compatible::same-id")

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.args[1].value, "failed")
        self.assertEqual(update.kwargs["termination_reason"], "rate_limited")
        repository.complete_chat_task.assert_not_awaited()

    async def test_plan_only_build_is_interrupted_and_finalizes_usage_and_snapshot(self) -> None:
        async def stream():
            yield "I will make a plan."

        repository, usage = await self._run(stream(), make_execution_policy("build", source="workspace_agent"), usage_attempt_id="usage-1")

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.args[1].value, "interrupted")
        self.assertEqual(update.kwargs["termination_reason"], "no_actions")
        repository.link_assistant_snapshot_tool_calls.assert_awaited_once()
        usage.assert_awaited_once()

    async def test_general_max_iterations_is_completed_with_warning(self) -> None:
        async def stream():
            yield "partial final"
            yield {"type": "max_iterations_reached"}

        repository, _usage = await self._run(stream(), make_execution_policy("general", source="workspace_agent"))

        complete = repository.complete_chat_task.await_args
        self.assertEqual(complete.kwargs["termination_reason"], "max_iterations")
        self.assertTrue(complete.kwargs["outcome_summary"]["warnings"])

    async def test_all_failed_real_tool_execution_is_interrupted(self) -> None:
        async def stream():
            yield {"type": "tool_start", "run_id": "tool-1", "tool": "write_file", "input": {"path": "a.txt"}}
            yield {"type": "tool_end", "run_id": "tool-1", "output": "Error: permission denied"}
            yield "I could not write the file."

        repository, _usage = await self._run(stream(), make_execution_policy("build", source="workspace_agent"))

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.kwargs["termination_reason"], "all_tools_failed")
        self.assertEqual(update.kwargs["outcome_summary"]["activity"], {"attempted": 1, "succeeded": 0, "failed": 1})


if __name__ == "__main__":
    unittest.main()
