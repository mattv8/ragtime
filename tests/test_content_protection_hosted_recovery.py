import asyncio
import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

from langchain_core.messages import AIMessage, BaseMessage

from ragtime.content_protection import service as protection_service
from ragtime.content_protection.models import ContentProtectionError
from ragtime.rag import components as components_module
from ragtime.rag.components import RAGComponents


def _eligible_error() -> ContentProtectionError:
    return ContentProtectionError(
        "content_denied",
        "recovery-test",
        reason="The response is outside the access policy.",
        recovery_eligible=True,
        attempts_remaining=1,
    )


class HostedRecoveryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def _components(self, response: AIMessage, calls: list[list[BaseMessage]]) -> RAGComponents:
        components = object.__new__(RAGComponents)

        async def get_llm(_self, _model):
            class LLM:
                async def ainvoke(self, messages, **kwargs):
                    calls.append(list(messages))
                    self_outer.assertIn("callbacks", kwargs["config"])
                    return response

            return SimpleNamespace(llm=LLM())

        self_outer = self
        self.enterContext(mock.patch.object(components, "_get_request_scoped_llm", new=MethodType(get_llm, components)))
        return components

    async def test_nonstream_replaces_terminal_denial_once_without_denied_draft(self) -> None:
        recovery_calls: list[list[BaseMessage]] = []
        components = self._components(AIMessage(content="approved alternative"), recovery_calls)
        error = _eligible_error()

        async def unprotected(*_args):
            state = protection_service._turn.get()
            assert state is not None
            state.terminal = error
            return "DENIED-CANARY"

        with (
            mock.patch.object(components, "_process_query_unprotected", new=unprotected),
            mock.patch("ragtime.rag.components.require_hosted_execution", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_history", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_inbound", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_assistant", new=mock.AsyncMock()),
        ):
            answer = await components.process_query("safe request", [AIMessage(content="approved history")], user_id="caller", owner_user_id="owner")

        self.assertEqual(answer, "approved alternative")
        self.assertEqual(len(recovery_calls), 1)
        rendered_prompt = "\n".join(str(message.content) for message in recovery_calls[0])
        self.assertNotIn("DENIED-CANARY", rendered_prompt)
        self.assertNotIn("DENIED-CANARY", answer)
        self.assertIsNone(protection_service.terminal_error())

    async def test_stream_closes_original_before_single_tool_free_replacement(self) -> None:
        recovery_calls: list[list[BaseMessage]] = []
        components = self._components(AIMessage(content="approved alternative"), recovery_calls)
        components._tool_configs = []
        error = _eligible_error()
        closed = False

        async def unprotected(*_args):
            nonlocal closed
            try:
                yield "DENIED-CANARY"
            finally:
                closed = True

        async def buffered(stream, **_kwargs):
            async for _event in stream:
                state = protection_service._turn.get()
                assert state is not None
                state.terminal = error
                raise error
            yield None

        with (
            mock.patch.object(components, "_process_query_stream_unprotected", new=unprotected),
            mock.patch("ragtime.rag.components.require_hosted_execution", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_history", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_inbound", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_assistant", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.content_protection_buffered_stream", side_effect=buffered),
        ):
            events = [event async for event in components.process_query_stream("safe request", user_id="caller", owner_user_id="owner")]

        self.assertEqual(events, ["approved alternative"])
        self.assertTrue(closed)
        self.assertEqual(len(recovery_calls), 1)
        self.assertNotIn("DENIED-CANARY", "\n".join(str(message.content) for message in recovery_calls[0]))

    async def test_second_denial_does_not_generate_a_third_response(self) -> None:
        calls: list[list[BaseMessage]] = []
        components = self._components(AIMessage(content="second denied draft"), calls)
        error = _eligible_error()

        async def deny(*_args, **_kwargs):
            await self._deny_replacement(error)

        with (
            mock.patch("ragtime.rag.components.require_hosted_execution", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_assistant", new=mock.AsyncMock(side_effect=deny)),
        ):
            with protection_service.protection_context(protection_service.ProtectionContext(user_id="caller")):
                state = protection_service._turn.get()
                assert state is not None
                state.terminal = error
                with self.assertRaises(ContentProtectionError):
                    await components._recover_content_protection_denial(
                        error,
                        user_message="safe request",
                        chat_history=[],
                        conversation_model=None,
                        user_id="caller",
                        owner_user_id="owner",
                        context=protection_service.current_context(),
                    )

        self.assertEqual(len(calls), 1)

    async def _deny_replacement(self, error: ContentProtectionError) -> None:
        successor = protection_service._turn.get()
        assert successor is not None
        successor.terminal = ContentProtectionError("content_denied", "second-denial")
        raise successor.terminal

    async def test_recovery_rejects_model_returned_tool_calls(self) -> None:
        calls: list[list[BaseMessage]] = []
        components = self._components(AIMessage(content="", tool_calls=[{"name": "tool", "args": {}, "id": "call-1"}]), calls)
        error = _eligible_error()

        with mock.patch("ragtime.rag.components.require_hosted_execution", new=mock.AsyncMock()):
            with protection_service.protection_context(protection_service.ProtectionContext(user_id="caller")):
                state = protection_service._turn.get()
                assert state is not None
                state.terminal = error
                with self.assertRaises(ContentProtectionError):
                    await components._recover_content_protection_denial(
                        error,
                        user_message="safe request",
                        chat_history=[],
                        conversation_model=None,
                        user_id="caller",
                        owner_user_id="owner",
                        context=protection_service.current_context(),
                    )

        self.assertEqual(len(calls), 1)

    async def test_hosted_execution_gate_blocks_recovery_before_generation(self) -> None:
        calls: list[list[BaseMessage]] = []
        components = self._components(AIMessage(content="must not generate"), calls)
        error = _eligible_error()

        with mock.patch("ragtime.rag.components.require_hosted_execution", side_effect=RuntimeError("hosted disabled")):
            with protection_service.protection_context(protection_service.ProtectionContext(user_id="caller")):
                state = protection_service._turn.get()
                assert state is not None
                state.terminal = error
                with self.assertRaisesRegex(RuntimeError, "hosted disabled"):
                    await components._recover_content_protection_denial(
                        error,
                        user_message="safe request",
                        chat_history=[],
                        conversation_model=None,
                        user_id="caller",
                        owner_user_id="owner",
                        context=protection_service.current_context(),
                    )

        self.assertEqual(calls, [])

    async def test_noneligible_failure_does_not_attempt_recovery_generation(self) -> None:
        calls: list[list[BaseMessage]] = []
        components = self._components(AIMessage(content="must not generate"), calls)
        error = ContentProtectionError("classifier_unavailable", "operational")

        async def unprotected(*_args):
            state = protection_service._turn.get()
            assert state is not None
            state.terminal = error
            raise error

        with (
            mock.patch.object(components, "_process_query_unprotected", new=unprotected),
            mock.patch("ragtime.rag.components.require_hosted_execution", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_history", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_inbound", new=mock.AsyncMock()),
        ):
            with self.assertRaises(ContentProtectionError):
                await components.process_query("safe request", user_id="caller", owner_user_id="owner")

        self.assertEqual(calls, [])

    async def test_stream_cancellation_closes_the_original_generator(self) -> None:
        calls: list[list[BaseMessage]] = []
        components = self._components(AIMessage(content="must not generate"), calls)
        components._tool_configs = []
        closed = False

        async def unprotected(*_args):
            nonlocal closed
            try:
                yield "approved chunk"
                await asyncio.Event().wait()
            finally:
                closed = True

        async def passthrough(stream, **_kwargs):
            async for event in stream:
                yield event

        with (
            mock.patch.object(components, "_process_query_stream_unprotected", new=unprotected),
            mock.patch("ragtime.rag.components.require_hosted_execution", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_history", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_inbound", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.content_protection_buffered_stream", side_effect=passthrough),
        ):
            stream = components.process_query_stream("safe request", user_id="caller", owner_user_id="owner")
            self.assertEqual(await anext(stream), "approved chunk")
            await stream.aclose()

        self.assertTrue(closed)
        self.assertEqual(calls, [])

    def test_recovery_preserves_approved_api_history_and_omits_tool_replay(self) -> None:
        history = RAGComponents._recovery_history_messages(
            [
                {"role": "system", "content": "workspace constraint"},
                {"role": "user", "content": "prior approved question"},
                {"role": "assistant", "content": "prior approved answer"},
                AIMessage(content="", tool_calls=[{"name": "dangerous", "args": {}, "id": "1"}]),
            ]
        )

        self.assertEqual([message.content for message in history], ["workspace constraint", "prior approved question", "prior approved answer"])

    async def test_recovery_retains_approved_runtime_system_constraints(self) -> None:
        calls: list[list[BaseMessage]] = []
        components = self._components(AIMessage(content="approved alternative"), calls)
        error = _eligible_error()
        scope_token = components_module._recovery_request_scope.set(
            components_module._RecoveryRequestScope("workspace privacy constraint", "current user constraint", ({"role": "user", "content": "prior"},))
        )
        try:
            with (
                mock.patch("ragtime.rag.components.require_hosted_execution", new=mock.AsyncMock()),
                mock.patch("ragtime.rag.components.authorize_assistant", new=mock.AsyncMock()),
            ):
                with protection_service.protection_context(protection_service.ProtectionContext(user_id="caller")):
                    state = protection_service._turn.get()
                    assert state is not None
                    state.terminal = error
                    answer = await components._recover_content_protection_denial(
                        error,
                        user_message="safe request",
                        chat_history=[],
                        conversation_model=None,
                        user_id="caller",
                        owner_user_id="owner",
                        context=protection_service.current_context(),
                    )
        finally:
            components_module._recovery_request_scope.reset(scope_token)

        self.assertEqual(answer, "approved alternative")
        self.assertEqual(calls[0][0].content, "workspace privacy constraint")
        self.assertIn("current user constraint", [message.content for message in calls[0]])
