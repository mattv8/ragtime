from __future__ import annotations

import json
import unittest
from contextlib import aclosing, nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, AsyncGenerator, cast
from unittest import mock

from fastapi import HTTPException

from ragtime.core import generation_policy
from ragtime.indexer import routes

NOW = datetime(2026, 9, 21, tzinfo=timezone.utc)
MULTIMODAL = json.dumps([{"type": "text", "text": "hello"}])


def _policy_db(
    *,
    chat_enabled: bool,
    userspace_enabled: bool,
    caller_chat: bool | None = None,
    caller_userspace: bool | None = None,
    owner_chat: bool | None = None,
    owner_userspace: bool | None = None,
) -> SimpleNamespace:
    users = [
        SimpleNamespace(id="caller", chatEnabled=caller_chat, userspaceGenerationEnabled=caller_userspace),
        SimpleNamespace(id="owner", chatEnabled=owner_chat, userspaceGenerationEnabled=owner_userspace),
    ]
    return SimpleNamespace(
        appsettings=SimpleNamespace(
            find_unique=mock.AsyncMock(return_value=SimpleNamespace(chatEnabled=chat_enabled, userspaceGenerationEnabled=userspace_enabled))
        ),
        user=SimpleNamespace(find_many=mock.AsyncMock(return_value=users)),
    )


def _conversation(workspace_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        id="conversation-1",
        user_id="owner",
        model="provider::model",
        title="Existing title",
        workspace_id=workspace_id,
        disabled_builtin_tool_ids=[],
        messages=[
            SimpleNamespace(role="user", content=MULTIMODAL, events=None, timestamp=NOW),
            SimpleNamespace(role="user", content=MULTIMODAL, events=None, timestamp=NOW),
        ],
    )


def _user() -> SimpleNamespace:
    return SimpleNamespace(id="caller", username="caller", displayName="Caller", role="user")


class GenerationRouteIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_calls_real_surface_wrappers_without_unsupported_keywords(self) -> None:
        cases = [(None, True, False), ("workspace-1", False, True)]
        for workspace_id, chat_enabled, userspace_enabled in cases:
            with self.subTest(workspace_id=workspace_id):
                db = _policy_db(chat_enabled=chat_enabled, userspace_enabled=userspace_enabled)
                with (
                    mock.patch.object(generation_policy, "get_db", new=mock.AsyncMock(return_value=db)),
                    mock.patch.object(routes, "rag", SimpleNamespace(is_ready=True)),
                    mock.patch.object(
                        routes,
                        "_validate_conversation_model_before_send",
                        new=mock.AsyncMock(return_value="provider::model"),
                    ),
                    mock.patch.object(routes, "_persist_generation_failure_message", new=mock.AsyncMock()),
                ):
                    actual = await routes._validate_generation_ready_after_user_message(
                        "conversation-1",
                        "provider::model",
                        user_id="owner",
                        caller_user_id="caller",
                        workspace_id=workspace_id,
                    )
                self.assertEqual(actual, "provider::model")

    async def test_readiness_checks_caller_and_owner_and_global_vetoes_for_each_surface(self) -> None:
        cases = [
            (None, _policy_db(chat_enabled=True, userspace_enabled=True, owner_chat=False), "chat_generation_disabled"),
            (
                "workspace-1",
                _policy_db(chat_enabled=True, userspace_enabled=True, caller_userspace=False),
                "userspace_generation_disabled",
            ),
            (None, _policy_db(chat_enabled=False, userspace_enabled=False), "chat_generation_disabled"),
            (
                "workspace-1",
                _policy_db(chat_enabled=False, userspace_enabled=False),
                "userspace_generation_disabled",
            ),
        ]
        for workspace_id, db, expected_code in cases:
            with self.subTest(workspace_id=workspace_id, expected_code=expected_code):
                with (
                    mock.patch.object(generation_policy, "get_db", new=mock.AsyncMock(return_value=db)),
                    mock.patch.object(routes, "rag", SimpleNamespace(is_ready=True)),
                    mock.patch.object(routes, "_persist_generation_failure_message", new=mock.AsyncMock()),
                ):
                    with self.assertRaises(HTTPException) as raised:
                        await routes._validate_generation_ready_after_user_message(
                            "conversation-1",
                            "provider::model",
                            user_id="owner",
                            caller_user_id="caller",
                            workspace_id=workspace_id,
                        )
                detail = raised.exception.detail
                self.assertIsInstance(detail, dict)
                self.assertEqual(cast(dict[str, Any], detail)["code"], expected_code)

    async def _exercise_sync_surface(self, workspace_id: str | None, expected_surface: str) -> None:
        conversation = _conversation(workspace_id)
        observed: list[tuple[str, str | None]] = []

        async def preprocess(_content: Any, **_kwargs: Any) -> tuple[str, dict[str, Any]]:
            observed.append(("preprocess", generation_policy.current_generation_surface()))
            return "expanded", {}

        async def process_query(*_args: Any, **_kwargs: Any) -> str:
            observed.append(("provider", generation_policy.current_generation_surface()))
            await generation_policy.require_generation("caller", "owner")
            return "answer"

        fake_rag = SimpleNamespace(
            is_ready=True,
            preprocess_message_content_async=preprocess,
            process_query=process_query,
        )
        db = _policy_db(
            chat_enabled=expected_surface == "chat",
            userspace_enabled=expected_surface == "userspace",
        )
        with (
            mock.patch.object(generation_policy, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(routes, "rag", fake_rag),
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "add_message", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(
                routes,
                "_validate_conversation_model_before_send",
                new=mock.AsyncMock(return_value=conversation.model),
            ),
            mock.patch.object(routes, "schedule_title_generation"),
            mock.patch.object(routes, "create_usage_attempt", new=mock.AsyncMock(return_value="attempt")),
            mock.patch.object(routes, "finalize_usage_attempt", new=mock.AsyncMock()),
            mock.patch.object(routes, "_to_conversation_response", return_value=conversation),
        ):
            result = await routes._send_message_to_loaded_conversation(
                cast(Any, conversation),
                routes.SendMessageRequest(message=MULTIMODAL),
                cast(Any, _user()),
                workspace_id=None,
                blocked_tool_names=set(),
            )

        self.assertEqual(result["message"].content, "answer")
        self.assertEqual(
            observed,
            [("preprocess", expected_surface), ("preprocess", expected_surface), ("provider", expected_surface)],
        )
        self.assertIsNone(generation_policy.current_generation_surface())

    async def test_sync_chat_uses_chat_scope_when_userspace_is_disabled(self) -> None:
        await self._exercise_sync_surface(None, "chat")

    async def test_sync_userspace_uses_persisted_workspace_scope_when_chat_is_disabled(self) -> None:
        await self._exercise_sync_surface("workspace-1", "userspace")

    async def _exercise_stream_surface(self, workspace_id: str | None, expected_surface: str) -> None:
        conversation = _conversation(workspace_id)
        observed: list[tuple[str, str | None]] = []
        closed = False

        async def preprocess(_content: Any, **_kwargs: Any) -> tuple[str, dict[str, Any]]:
            observed.append(("preprocess", generation_policy.current_generation_surface()))
            return "expanded", {}

        async def process_query_stream(*_args: Any, **_kwargs: Any):
            nonlocal closed
            try:
                observed.append(("provider", generation_policy.current_generation_surface()))
                await generation_policy.require_generation("caller", "owner")
                yield "answer"
            finally:
                closed = True

        fake_rag = SimpleNamespace(
            is_ready=True,
            agent_executor=None,
            preprocess_message_content_async=preprocess,
            process_query_stream=process_query_stream,
        )
        db = _policy_db(
            chat_enabled=expected_surface == "chat",
            userspace_enabled=expected_surface == "userspace",
        )
        with (
            mock.patch.object(generation_policy, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(routes, "rag", fake_rag),
            mock.patch.object(routes, "_assert_workspace_access", new=mock.AsyncMock()),
            mock.patch.object(routes.repository, "check_conversation_access", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(routes.repository, "get_conversation", new=mock.AsyncMock(side_effect=[conversation, conversation])),
            mock.patch.object(routes.repository, "add_message", new=mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "authorize_inbound", new=mock.AsyncMock()),
            mock.patch.object(
                routes,
                "_resolve_workspace_runtime_scope",
                new=mock.AsyncMock(return_value=(workspace_id, set(), None)),
            ),
            mock.patch.object(
                routes,
                "_validate_conversation_model_before_send",
                new=mock.AsyncMock(return_value=conversation.model),
            ),
            mock.patch.object(routes, "schedule_title_generation"),
            mock.patch.object(routes, "create_usage_attempt", new=mock.AsyncMock(return_value="attempt")),
            mock.patch.object(routes, "finalize_usage_attempt", new=mock.AsyncMock()),
            mock.patch.object(routes, "_link_assistant_snapshot_tool_calls", new=mock.AsyncMock()),
            mock.patch.object(routes, "bind_content_protection_context", return_value=nullcontext()),
        ):
            response = await routes.send_message_stream(
                "conversation-1",
                routes.SendMessageRequest(message=MULTIMODAL),
                workspace_id=None,
                user=cast(Any, _user()),
            )
            body_iterator: AsyncGenerator[str, None] = cast(AsyncGenerator[str, None], response.body_iterator)
            async with aclosing(body_iterator):
                first_chunk = await anext(body_iterator)

        self.assertIn("answer", cast(str, first_chunk))
        self.assertEqual(
            observed,
            [("preprocess", expected_surface), ("preprocess", expected_surface), ("provider", expected_surface)],
        )
        self.assertTrue(closed)
        self.assertIsNone(generation_policy.current_generation_surface())

    async def test_stream_chat_keeps_chat_scope_during_iteration_when_userspace_is_disabled(self) -> None:
        await self._exercise_stream_surface(None, "chat")

    async def test_stream_userspace_keeps_persisted_scope_during_iteration_when_chat_is_disabled(self) -> None:
        await self._exercise_stream_surface("workspace-1", "userspace")


if __name__ == "__main__":
    unittest.main()
