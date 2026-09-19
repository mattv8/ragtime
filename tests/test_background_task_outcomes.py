"""Focused outcome-policy tests for durable background chat tasks."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import ragtime.indexer.background_tasks as background_tasks
from ragtime.indexer.task_policy import activity_summary, make_execution_policy, required_action_termination
from tests.hosted_execution_test_support import enabled_hosted_execution_policy


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
    def _dependencies(self, stream):
        conversation = SimpleNamespace(
            messages=[SimpleNamespace(role="user", content="build it", events=None)],
            user_id="user-1",
            model="openai::test",
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

    async def _run(self, stream, policy, *, usage_attempt_id=None):
        service = background_tasks.BackgroundTaskService()
        repository, rag, settings = self._dependencies(stream)
        bus = SimpleNamespace(publish=mock.AsyncMock())
        with (
            enabled_hosted_execution_policy("user-1"),
            mock.patch.object(background_tasks, "repository", repository),
            mock.patch.object(background_tasks, "rag", rag),
            mock.patch.object(background_tasks, "task_event_bus", bus),
            mock.patch.object(background_tasks.SettingsCache, "get_instance", return_value=settings),
            mock.patch.object(background_tasks, "finalize_usage_attempt", mock.AsyncMock()) as finalize_usage,
        ):
            service.start_task("conv-1", "build it", existing_task_id="task-1", execution_policy=policy, usage_attempt_id=usage_attempt_id)
            await asyncio.wait_for(service._running_tasks["task-1"], timeout=1)
        return repository, finalize_usage

    async def test_structured_payment_error_persists_partial_and_closes_usage(self) -> None:
        async def stream():
            yield "partial work"
            yield {"type": "error", "code": "payment_required", "content": "Payment required."}

        with mock.patch("ragtime.core.openrouter_credits.note_openrouter_payment_required", return_value="Credit warning"):
            repository, usage = await self._run(stream(), make_execution_policy("build", source="workspace_agent"), usage_attempt_id="usage-1")

        update = repository.update_chat_task_status.await_args
        self.assertEqual(update.args[1].value, "failed")
        self.assertEqual(update.kwargs["termination_reason"], "payment_required")
        self.assertEqual(update.kwargs["response_content"], "partial work")
        self.assertEqual(update.kwargs["outcome_summary"]["warnings"], ["Credit warning"])
        repository.link_assistant_snapshot_tool_calls.assert_awaited_once()
        usage.assert_awaited_once()

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
