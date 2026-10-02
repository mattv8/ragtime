from __future__ import annotations

import os
import unittest
from unittest import mock

import httpx
import openai
from langchain_core.messages import HumanMessage

import ragtime.indexer.background_tasks as background_tasks
from ragtime.core.openai_compatible_client import CompatibleChatOpenAI, compatible_chat_options
from ragtime.indexer.task_policy import make_execution_policy
from ragtime.rag.components import RAGComponents, RequestLLMResolution
from tests.content_protection_support import use_disabled_content_protection
from tests.generation_policy_test_support import enabled_generation_policy
from tests.test_background_task_outcomes import BackgroundTaskOutcomeExecutionTests


class OpenAICompatibleSDKErrorTests(unittest.IsolatedAsyncioTestCase):
    async def _sdk_error(self, response_or_error: httpx.Response | Exception) -> BaseException:
        def handler(request: httpx.Request) -> httpx.Response:
            if isinstance(response_or_error, Exception):
                raise response_or_error
            return response_or_error

        options = compatible_chat_options("")
        options["http_client"]._transport = httpx.MockTransport(handler)
        options["http_async_client"]._transport = httpx.MockTransport(handler)
        try:
            with mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "env-key", "OPENAI_ORG_ID": "env-org", "OPENAI_PROJECT_ID": "env-project", "OPENAI_PROXY": "http://invalid.proxy"},
            ):
                client = CompatibleChatOpenAI(model="test", base_url="https://compatible.test/v1", max_retries=0, use_responses_api=False, **options)
                with self.assertRaises(openai.APIError) as raised:
                    await client.ainvoke([HumanMessage(content="hello")])
                return raised.exception
        finally:
            options["http_client"].close()
            await options["http_async_client"].aclose()

    async def test_sdk_status_errors_are_safe_terminal_events_and_background_failures(self) -> None:
        for status, expected_code in ((401, "authentication"), (429, "rate_limited")):
            with self.subTest(status=status):
                error = await self._sdk_error(httpx.Response(status, json={"error": {"message": "leaked upstream key"}}))
                rag = RAGComponents()
                resolution = RequestLLMResolution(llm=object(), provider="openai_compatible", model="same-id")
                classification = rag._openai_compatible_error(error, "openai_compatible")
                assert classification is not None
                self.assertEqual(classification[0], expected_code)
                event = rag._provider_error_stream_event(error, resolution)
                assert event is not None
                self.assertEqual(event["code"], expected_code)
                self.assertNotIn("leaked upstream key", event["content"])

                runner = BackgroundTaskOutcomeExecutionTests(methodName="run")
                use_disabled_content_protection(runner)

                async def stream():
                    raise error
                    yield  # pragma: no cover

                with enabled_generation_policy("user-1"):
                    repository, _usage = await runner._run(
                        stream(),
                        make_execution_policy("general", source="workspace_agent"),
                        model="openai_compatible::same-id",
                    )
                self.assertEqual(repository.update_chat_task_status.await_args.args[1].value, "failed")
                self.assertEqual(repository.update_chat_task_status.await_args.kwargs["termination_reason"], expected_code)

    async def test_sdk_timeout_is_safe_and_terminal(self) -> None:
        error = await self._sdk_error(httpx.ReadTimeout("upstream timeout"))
        rag = RAGComponents()
        resolution = RequestLLMResolution(llm=object(), provider="openai_compatible", model="same-id")
        event = rag._provider_error_stream_event(error, resolution)
        assert event is not None
        self.assertEqual(event["code"], "timeout")
        self.assertIn("timed out", event["content"])

    def test_explicit_non_openrouter_prefix_never_uses_default_openrouter(self) -> None:
        self.assertEqual(background_tasks._explicit_provider_from_model("anthropic::same"), "anthropic")
        self.assertEqual(background_tasks._explicit_provider_from_model("openai::same"), "openai")
        self.assertIsNone(background_tasks._explicit_provider_from_model("same"))

    async def test_explicit_other_provider_payment_does_not_trigger_default_openrouter_credit_warning(self) -> None:
        runner = BackgroundTaskOutcomeExecutionTests(methodName="run")
        use_disabled_content_protection(runner)

        async def stream():
            yield {"type": "error", "code": "payment_required", "content": "Payment required."}

        with (
            enabled_generation_policy("user-1"),
            mock.patch("ragtime.core.openrouter_credits.note_openrouter_payment_required") as note,
        ):
            repository, _usage = await runner._run(
                stream(),
                make_execution_policy("general", source="workspace_agent"),
                model="anthropic::same-id",
                settings_values={"llm_provider": "openrouter"},
            )
        self.assertEqual(repository.update_chat_task_status.await_args.kwargs["outcome_summary"]["warnings"], [])
        note.assert_not_called()
