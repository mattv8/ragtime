import asyncio
import contextvars
import unittest
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, TypeGuard, cast
from unittest import mock

from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User

from ragtime.content_protection import hosted, service
from ragtime.content_protection.models import ContentProtectionConfig, ContentProtectionError, ProtectionContext
from ragtime.indexer import routes

NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)


def _user() -> User:
    return User(
        id="caller",
        username="caller",
        authProvider=AuthProvider.local,
        cachedGroups=cast(Json, "[]"),
        role=UserRole.user,
        roleManuallySet=False,
        createdAt=NOW,
        updatedAt=NOW,
        securityGeneration=0,
    )


@dataclass
class _CapturedRecovery:
    original: service._TurnState | None = None
    successor: service._TurnState | None = None
    persistence_before: tuple[ProtectionContext | None, service._TurnState | None] | None = None
    persistence_after: tuple[ProtectionContext | None, service._TurnState | None] | None = None


def _is_async_generator(value: object) -> TypeGuard[AsyncGenerator[Any, None]]:
    return isinstance(value, AsyncGenerator)


class ContentProtectionPersistenceRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = ContentProtectionConfig(enabled=True, classifier_model="openai::classifier")
        self.classifier_contexts: list[tuple[ProtectionContext, service._TurnState]] = []
        self.patches = [
            mock.patch.object(service, "load_config", new=mock.AsyncMock(return_value=self.config)),
            mock.patch.object(service, "resolve_identities", new=mock.AsyncMock(side_effect=self._identities)),
            mock.patch.object(service, "_provider_settings_identity", new=mock.AsyncMock(return_value={})),
            mock.patch.object(service, "_classify", new=self._allow),
            mock.patch.object(service, "_audit", new=mock.AsyncMock()),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    async def _identities(self, identities: set[str]) -> tuple[set[str], dict[str, set[str]], dict[str, None]]:
        return set(identities), {identity: set() for identity in identities}, {identity: None for identity in identities}

    async def _allow(self, _config, _envelope, **_kwargs) -> dict[str, str]:
        context = service.current_context()
        state = service._turn.get()
        assert context is not None
        assert state is not None
        self.classifier_contexts.append((context, state))
        return {"verdict": "allow", "reason_code": "permitted"}

    async def _stream_response(self, rag_stream: Callable[..., AsyncGenerator[str, None]]) -> list[str]:
        with mock.patch.object(routes, "rag", SimpleNamespace(process_query_stream=rag_stream)):
            response = await routes.send_message_stream(
                "conversation-1",
                routes.SendMessageRequest(message="safe request"),
                user=_user(),
            )
            body_iterator = response.body_iterator
            assert _is_async_generator(body_iterator)
            async with aclosing(body_iterator):
                chunks: list[str] = []
                async for chunk in body_iterator:
                    assert isinstance(chunk, str)
                    chunks.append(chunk)
        return chunks

    async def test_recovery_persistence_uses_successor_scope_and_budget(self) -> None:
        conversation = SimpleNamespace(
            id="conversation-1",
            user_id="owner",
            model="test-model",
            messages=[],
            workspace_id=None,
            disabled_builtin_tool_ids=[],
        )
        captured = _CapturedRecovery()

        async def add_message(_conversation_id, role, content, **_kwargs):
            if role == "assistant":
                captured.persistence_before = (service.current_context(), service._turn.get())
                await hosted.authorize_persistence({"role": role, "content": content}, direction="assistant_response")
                captured.persistence_after = (service.current_context(), service._turn.get())
            return conversation

        async def recovering_stream(*_args, **_kwargs):
            original = service._turn.get()
            captured.original = original
            denial = ContentProtectionError("content_denied", "denied", recovery_eligible=True)
            assert original is not None
            original.terminal = denial
            with service.recovery_attempt(denial) as successor:
                captured.successor = successor
                await service.authorize_content("approved replacement", direction="assistant_response")
                yield "approved replacement"

        with (
            mock.patch.object(routes, "_assert_workspace_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "check_conversation_access", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(routes.repository, "get_conversation", new=mock.AsyncMock(side_effect=[conversation, conversation])),
            mock.patch.object(routes.repository, "add_message", side_effect=add_message),
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(routes, "_resolve_workspace_runtime_scope", new=mock.AsyncMock(return_value=(None, set(), None))),
            mock.patch.object(routes, "_validate_generation_ready_after_user_message", new=mock.AsyncMock(return_value="test-model")),
            mock.patch.object(routes, "_apply_validated_conversation_model", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "schedule_title_generation"),
            mock.patch.object(routes, "_build_chat_history_for_conversation", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(routes, "create_usage_attempt", new=mock.AsyncMock(return_value="attempt")),
            mock.patch.object(routes, "finalize_usage_attempt", new=mock.AsyncMock()),
            mock.patch.object(routes, "_link_assistant_snapshot_tool_calls", new=mock.AsyncMock()),
        ):
            chunks = await self._stream_response(recovering_stream)

        self.assertTrue(any("approved replacement" in chunk for chunk in chunks))
        original = captured.original
        successor = captured.successor
        persistence_before = captured.persistence_before
        assert original is not None
        assert successor is not None
        assert persistence_before is not None
        persisted_context, persisted_state = persistence_before
        assert persisted_context is not None
        assert persisted_state is not None
        self.assertIs(persisted_state, successor)
        self.assertIsNot(persisted_state, original)
        self.assertIs(persisted_state.record, original.record)
        self.assertEqual(persisted_context.user_id, "caller")
        self.assertEqual(persisted_context.audience_user_ids, ("caller", "owner"))
        self.assertEqual(persisted_context.baseline, "user")
        self.assertGreater(persisted_state.record.spent, 0)
        self.assertTrue(all(context.user_id == "caller" for context, _state in self.classifier_contexts))
        self.assertTrue(all(context.baseline != "anonymous" for context, _state in self.classifier_contexts))
        self.assertIsNone(service.current_context())
        self.assertIsNone(service._turn.get())

    async def test_successful_stream_and_cancellation_close_bound_context(self) -> None:
        conversation = SimpleNamespace(
            id="conversation-1",
            user_id="owner",
            model="test-model",
            messages=[],
            workspace_id=None,
            disabled_builtin_tool_ids=[],
        )
        closed = False

        async def pending_stream(*_args, **_kwargs):
            nonlocal closed
            try:
                yield "approved"
                await asyncio.Event().wait()
            finally:
                closed = True

        with (
            mock.patch.object(routes, "_assert_workspace_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "check_conversation_access", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(routes.repository, "get_conversation", new=mock.AsyncMock(side_effect=[conversation, conversation])),
            mock.patch.object(routes.repository, "add_message", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(routes, "_resolve_workspace_runtime_scope", new=mock.AsyncMock(return_value=(None, set(), None))),
            mock.patch.object(routes, "_validate_generation_ready_after_user_message", new=mock.AsyncMock(return_value="test-model")),
            mock.patch.object(routes, "_apply_validated_conversation_model", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "schedule_title_generation"),
            mock.patch.object(routes, "_build_chat_history_for_conversation", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(routes, "create_usage_attempt", new=mock.AsyncMock(return_value="attempt")),
        ):
            with mock.patch.object(routes, "rag", SimpleNamespace(process_query_stream=pending_stream)):
                response = await routes.send_message_stream(
                    "conversation-1",
                    routes.SendMessageRequest(message="safe request"),
                    user=_user(),
                )
                body_iterator = response.body_iterator
                assert _is_async_generator(body_iterator)
                first_chunk = await anext(body_iterator)
                assert isinstance(first_chunk, str)
                self.assertIn("approved", first_chunk)
                await body_iterator.aclose()

        self.assertTrue(closed)
        self.assertIsNone(service.current_context())
        self.assertIsNone(service._turn.get())

    async def test_nested_recovery_keeps_old_child_terminal_and_persists_successor(self) -> None:
        calls = 0

        async def classify(_config, _envelope, **_kwargs):
            nonlocal calls
            calls += 1
            return {"verdict": "deny", "reason_code": "restricted_content"} if calls == 1 else {"verdict": "allow", "reason_code": "permitted"}

        with mock.patch.object(service, "_classify", new=classify):
            context = hosted.hosted_context(user_id="caller", owner_user_id="owner")
            with hosted.bind_context(context):
                with hosted.bind_context(context):
                    with self.assertRaises(ContentProtectionError) as raised:
                        await service.authorize_content("withheld", direction="assistant_response")
                    old_child = contextvars.copy_context()
                    original = service._turn.get()
                    assert original is not None
                    with service.recovery_attempt(raised.exception) as successor:
                        await service.authorize_content("approved replacement", direction="assistant_response")
                        await hosted.authorize_persistence("approved replacement", direction="assistant_response")
                        self.assertIs(service._turn.get(), successor)
                        self.assertIs(successor.record, original.record)
                        self.assertGreater(successor.record.spent, 0)
                    with self.assertRaises(ContentProtectionError) as stale:
                        old_child.run(service.ensure_active_attempt)

        self.assertIs(stale.exception, raised.exception)
        self.assertEqual(calls, 2)
