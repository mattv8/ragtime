from __future__ import annotations

import asyncio
import json
import os
import re
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, mock

try:
    import psycopg2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional test dependency
    psycopg2 = cast(Any, None)

from ragtime.content_protection.models import ContentProtectionError
from ragtime.indexer import denial_notices
from ragtime.indexer.denial_notices import build_denial_message
from ragtime.indexer.models import ChatTaskStatus
from ragtime.indexer.repository import IndexerRepository, _to_prisma_task_status

CONVERSATION_WINDOW_TEST_DATABASE_URL = os.environ.get("CONVERSATION_WINDOW_TEST_DATABASE_URL")


def test_build_denial_message_persists_only_the_frozen_safe_contract() -> None:
    error = ContentProtectionError(
        "content_denied",
        "request-1",
        reason="Restricted information is not available.",
    )

    message = build_denial_message("conversation-1", error)

    assert message["role"] == "assistant"
    assert message["content"] == "Request blocked."
    assert message["message_id"] == build_denial_message("conversation-1", error)["message_id"]
    assert set(message["events"][0]) == {"type", "channel", "code", "public_reason", "request_id", "timestamp"}
    assert message["events"][0]["type"] == "content_protection_denial"
    assert message["events"][0]["channel"] == "final"
    assert "Restricted information" in message["events"][0]["public_reason"]
    assert "reason" not in message["events"][0]


def test_build_denial_message_rejects_unknown_error_codes() -> None:
    error = ContentProtectionError("unknown_code", "request-1")

    try:
        build_denial_message("conversation-1", error)
    except ValueError as exc:
        assert str(exc) == "unsupported content protection error code"
    else:
        raise AssertionError("unknown protection code must not be persisted")


def test_build_denial_message_rejects_duck_typed_errors() -> None:
    class DuckError:
        code = "content_denied"
        request_id = "request-1"

        def public_detail(self) -> dict[str, str]:
            return {"message": "untrusted"}

    try:
        build_denial_message("conversation-1", DuckError())  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "error must be a ContentProtectionError"
    else:
        raise AssertionError("duck typed errors must not be persisted")


def test_denial_helpers_identify_only_typed_notices_and_latest_message_id() -> None:
    ordinary = {"role": "assistant", "content": "Request blocked.", "message_id": "ordinary", "events": [{"type": "tool"}]}
    denial = build_denial_message("conversation-1", ContentProtectionError("content_denied", "request-1"))

    assert denial_notices.is_denial_notice(ordinary) is False
    assert denial_notices.is_denial_notice(denial) is True
    assert denial_notices.latest_denial_message_id([ordinary, denial]) == denial["message_id"]


class PersistChatDenialTests(IsolatedAsyncioTestCase):
    async def test_storage_failure_returns_none_without_annotating_or_replacing_original_error(self) -> None:
        error = ContentProtectionError("content_denied", "request-1")
        from ragtime.indexer.repository import repository

        with mock.patch.object(repository, "record_content_protection_denial", new=mock.AsyncMock(side_effect=RuntimeError("storage unavailable"))):
            self.assertIsNone(await denial_notices.persist_chat_denial("conversation-1", error))

        self.assertIsNone(error.denial_message_id)

    async def test_publish_failure_keeps_committed_notice_marker(self) -> None:
        error = ContentProtectionError("content_denied", "request-1")
        from ragtime.indexer.repository import repository

        with (
            mock.patch.object(repository, "record_content_protection_denial", new=mock.AsyncMock(return_value=SimpleNamespace())),
            mock.patch.object(denial_notices.task_event_bus, "publish", new=mock.AsyncMock(side_effect=RuntimeError("bus unavailable"))),
        ):
            message_id = await denial_notices.persist_chat_denial("conversation-1", error)

        self.assertEqual(error.denial_message_id, message_id)

    async def test_suppressed_notice_leaves_error_unannotated_and_does_not_publish(self) -> None:
        error = ContentProtectionError("content_denied", "request-1")
        from ragtime.indexer.repository import repository

        with (
            mock.patch.object(repository, "record_content_protection_denial", new=mock.AsyncMock(return_value=None)) as record,
            mock.patch.object(denial_notices.task_event_bus, "publish", new=mock.AsyncMock()) as publish,
        ):
            message_id = await denial_notices.persist_chat_denial("conversation-1", error, suppress_if_task_active=True)

        self.assertIsNone(message_id)
        self.assertIsNone(error.denial_message_id)
        record.assert_awaited_once_with("conversation-1", error, task_id=None, suppress_if_task_active=True)
        publish.assert_not_awaited()


class _FakeTransaction:
    def __init__(self, conversation: SimpleNamespace | None) -> None:
        self.conversation_record = conversation
        self.conversation = SimpleNamespace(update=self.update)
        self.chattask: SimpleNamespace | None = None
        self.update_count = 0

    async def __aenter__(self) -> "_FakeTransaction":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def update(self, *, data: dict[str, object], **_: object) -> SimpleNamespace:
        assert self.conversation_record is not None
        self.update_count += 1
        persisted_messages = data["messages"]
        self.conversation_record.messages = getattr(persisted_messages, "data", persisted_messages)
        return self.conversation_record


class _FakeDatabase:
    def __init__(self, conversation: SimpleNamespace | None) -> None:
        self.transaction = _FakeTransaction(conversation)

    def tx(self) -> _FakeTransaction:
        return self.transaction


class ContentProtectionDenialRepositoryTests(IsolatedAsyncioTestCase):
    def _repository(self, messages: list[dict[str, object]] | None = None) -> tuple[IndexerRepository, _FakeDatabase]:
        conversation = SimpleNamespace(messages=list(messages or []))
        db = _FakeDatabase(conversation)
        repository = IndexerRepository()
        repository._get_db = mock.AsyncMock(return_value=db)  # type: ignore[method-assign]
        repository._lock_conversation_for_branch_mutation = mock.AsyncMock(return_value=conversation)  # type: ignore[method-assign]
        repository._prisma_conversation_to_model = mock.Mock(side_effect=lambda row: row)  # type: ignore[method-assign]
        return repository, db

    async def test_concurrent_same_denial_appends_once_inside_the_branch_lock(self) -> None:
        repository, db = self._repository()
        error = ContentProtectionError("content_denied", "request-1")

        await asyncio.gather(
            repository.record_content_protection_denial("conversation-1", error),
            repository.record_content_protection_denial("conversation-1", error),
        )

        self.assertEqual(db.transaction.update_count, 1)
        conversation_record = db.transaction.conversation_record
        assert conversation_record is not None
        self.assertEqual(len(conversation_record.messages), 1)
        self.assertEqual(conversation_record.messages[0]["message_id"], build_denial_message("conversation-1", error)["message_id"])

    async def test_distinct_request_ids_append_distinct_denials(self) -> None:
        repository, db = self._repository()

        await repository.record_content_protection_denial("conversation-1", ContentProtectionError("content_denied", "request-1"))
        await repository.record_content_protection_denial("conversation-1", ContentProtectionError("content_denied", "request-2"))

        conversation_record = db.transaction.conversation_record
        assert conversation_record is not None
        messages = conversation_record.messages
        self.assertEqual(len(messages), 2)
        self.assertNotEqual(messages[0]["message_id"], messages[1]["message_id"])

    async def test_missing_conversation_does_not_annotate_a_public_error_marker(self) -> None:
        error = ContentProtectionError("content_denied", "request-1")
        from ragtime.indexer.repository import repository

        with mock.patch.object(repository, "record_content_protection_denial", new=mock.AsyncMock(return_value=None)):
            message_id = await denial_notices.persist_chat_denial("missing", error)

        self.assertIsNone(message_id)
        self.assertIsNone(error.denial_message_id)

    async def test_live_task_suppresses_denial_after_conversation_lock(self) -> None:
        for status in (ChatTaskStatus.pending, ChatTaskStatus.running):
            with self.subTest(status=status):
                repository, db = self._repository()
                conversation = db.transaction.conversation_record
                assert conversation is not None
                conversation.activeTaskId = "task-1"
                events: list[str] = []

                async def find_active_task(**_: object) -> SimpleNamespace:
                    events.append("task-read")
                    return SimpleNamespace(status=_to_prisma_task_status(status))

                async def lock_conversation(*_: object) -> SimpleNamespace:
                    events.append("conversation-lock")
                    return conversation

                db.transaction.chattask = SimpleNamespace(find_unique=find_active_task)
                repository._lock_conversation_for_branch_mutation = mock.AsyncMock(side_effect=lock_conversation)  # type: ignore[method-assign]

                result = await repository.record_content_protection_denial(
                    "conversation-1", ContentProtectionError("content_denied", "request-1"), suppress_if_task_active=True
                )

                self.assertIsNone(result)
                self.assertEqual(events, ["conversation-lock", "task-read"])
                self.assertEqual(db.transaction.update_count, 0)

    async def test_stale_task_and_default_behavior_persist_denial(self) -> None:
        cases = ((True, ChatTaskStatus.completed, True), (False, ChatTaskStatus.running, True), (True, None, False))
        for suppress_if_task_active, task_status, has_active_task in cases:
            with self.subTest(suppress_if_task_active=suppress_if_task_active, task_status=task_status):
                repository, db = self._repository()
                conversation = db.transaction.conversation_record
                assert conversation is not None
                if has_active_task:
                    assert task_status is not None
                    conversation.activeTaskId = "task-1"
                    db.transaction.chattask = SimpleNamespace(
                        find_unique=mock.AsyncMock(return_value=SimpleNamespace(status=_to_prisma_task_status(task_status)))
                    )

                result = await repository.record_content_protection_denial(
                    "conversation-1",
                    ContentProtectionError("content_denied", "request-1"),
                    suppress_if_task_active=suppress_if_task_active,
                )

                self.assertIsNotNone(result)
                self.assertEqual(db.transaction.update_count, 1)


def test_window_preview_exposes_only_typed_denial_event_fields() -> None:
    preview = IndexerRepository._window_preview(
        {
            "role": "assistant",
            "content": "Request blocked.",
            "message_id": "notice-1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "events": [
                {
                    "type": "content_protection_denial",
                    "channel": "final",
                    "code": "content_denied",
                    "public_reason": "Safe reason",
                    "request_id": "request-1",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "raw_output": "must not escape",
                },
                {"type": "tool", "output": "also private"},
            ],
        },
        max_bytes=1024,
    )

    assert preview.events == [
        {
            "type": "content_protection_denial",
            "channel": "final",
            "code": "content_denied",
            "public_reason": "Safe reason",
            "request_id": "request-1",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    ]


class ConversationSummaryProjectionTests(IsolatedAsyncioTestCase):
    async def test_summary_projection_keeps_workspace_filter_and_returns_latest_denial_id(self) -> None:
        row = {
            "id": "conversation-1",
            "title": "Chat",
            "model": "model",
            "message_count": 1,
            "total_tokens": 0,
            "subagents_enabled": True,
            "last_denial_message_id": "notice-1",
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[row]))
        repository = IndexerRepository()
        repository._get_db = mock.AsyncMock(return_value=db)  # type: ignore[method-assign]
        repository.get_subagent_conversation_ids_by_parent = mock.AsyncMock(return_value={})  # type: ignore[method-assign]

        summaries = await repository.list_conversation_summaries(workspace_id="workspace-1", include_all=True)

        self.assertEqual(summaries[0].last_denial_message_id, "notice-1")
        query = db.query_raw.await_args.args[0]
        self.assertIn("c.workspace_id = 'workspace-1'", query)
        self.assertIn("AS last_denial_message_id", query)


def _status_matches(current: object, condition: object) -> bool:
    if isinstance(condition, dict):
        return current in condition["in"]
    return current == condition


class _FakeChatTaskTable:
    """Mirrors ChatTask update_many semantics: filter by where, return the matched row count as int."""

    def __init__(self, task: SimpleNamespace) -> None:
        self.task = task

    async def find_unique(self, *, where: dict[str, object]) -> SimpleNamespace | None:
        return self.task if where["id"] == self.task.id else None

    async def update_many(self, *, where: dict[str, object], data: dict[str, object]) -> int:
        status_matches = _status_matches(self.task.status, where.get("status"))
        alternatives = where.get("OR")
        if isinstance(alternatives, list):
            status_matches = any(isinstance(clause, dict) and _status_matches(self.task.status, clause.get("status")) for clause in alternatives)
        if where["id"] != self.task.id or not status_matches:
            return 0
        for key, value in data.items():
            setattr(self.task, key, getattr(value, "data", value))
        return 1


def _chat_task(status: ChatTaskStatus) -> SimpleNamespace:
    return SimpleNamespace(
        id="task-1",
        conversationId="conversation-1",
        status=_to_prisma_task_status(status),
        responseContent=None,
        errorMessage=None,
        streamingState={
            "content": "private partial",
            "events": [{"type": "tool", "output": "private"}],
            "tool_calls": [{"output": "private"}],
            "hit_max_iterations": False,
            "version": 4,
            "content_length": len("private partial"),
        },
    )


_CLEAR_ACTIVE_TASK_SQL = re.compile(r"UPDATE conversations SET active_task_id = NULL WHERE id = '((?:[^']|'')*)' AND active_task_id = '((?:[^']|'')*)'")


def _clear_active_task_sql(conversation: SimpleNamespace) -> mock.AsyncMock:
    """Mirrors the production conditional clear: null the pointer only when it still names the task."""

    async def execute_raw(query: str) -> int:
        match = _CLEAR_ACTIVE_TASK_SQL.fullmatch(query)
        if match is None:
            raise AssertionError(f"fake database does not understand SQL: {query}")
        conversation_id, task_id = (literal.replace("''", "'") for literal in match.groups())
        if conversation.id != conversation_id or conversation.activeTaskId != task_id:
            return 0
        conversation.activeTaskId = None
        return 1

    return mock.AsyncMock(side_effect=execute_raw)


def _task_repository(task: SimpleNamespace, *, active_task_id: str | None = "task-1") -> tuple[IndexerRepository, SimpleNamespace]:
    conversation = SimpleNamespace(id="conversation-1", activeTaskId=active_task_id)
    db = SimpleNamespace(
        chattask=_FakeChatTaskTable(task),
        conversation=conversation,
        execute_raw=_clear_active_task_sql(conversation),
    )
    repository = IndexerRepository()
    repository._get_db = mock.AsyncMock(return_value=db)  # type: ignore[method-assign]
    repository._prisma_task_to_model = mock.Mock(side_effect=lambda row: row)  # type: ignore[method-assign]
    return repository, db


class ChatTaskTerminalStateTests(IsolatedAsyncioTestCase):
    async def test_running_task_terminal_clear_fails_it_and_replaces_private_stream_state(self) -> None:
        task = _chat_task(ChatTaskStatus.running)
        repository, db = _task_repository(task)

        result = await repository.update_chat_task_status(
            "task-1", ChatTaskStatus.failed, "Request blocked.", response_content="Request blocked.", clear_streaming_state=True
        )

        self.assertIs(result, task)
        self.assertEqual(task.status, _to_prisma_task_status(ChatTaskStatus.failed))
        self.assertEqual(task.responseContent, "Request blocked.")
        self.assertEqual(task.streamingState["content"], "")
        self.assertEqual(task.streamingState["events"], [])
        self.assertEqual(task.streamingState["tool_calls"], [])
        self.assertEqual(task.streamingState["version"], 5)
        self.assertIsNone(db.conversation.activeTaskId)
        db.execute_raw.assert_awaited_once()

    async def test_pending_task_denied_before_running_fails_and_clears_active_task(self) -> None:
        task = _chat_task(ChatTaskStatus.pending)
        repository, db = _task_repository(task)

        result = await repository.update_chat_task_status(
            "task-1", ChatTaskStatus.failed, "Request blocked.", response_content="Request blocked.", clear_streaming_state=True
        )

        self.assertIs(result, task)
        self.assertEqual(task.status, _to_prisma_task_status(ChatTaskStatus.failed))
        self.assertEqual(task.responseContent, "Request blocked.")
        self.assertEqual(task.streamingState["content"], "")
        self.assertEqual(task.streamingState["events"], [])
        self.assertEqual(task.streamingState["tool_calls"], [])
        self.assertEqual(task.streamingState["version"], 5)
        self.assertIsNone(db.conversation.activeTaskId)
        db.execute_raw.assert_awaited_once()

    async def test_terminal_clear_leaves_a_different_active_task_pointer_untouched(self) -> None:
        task = _chat_task(ChatTaskStatus.running)
        repository, db = _task_repository(task, active_task_id="task-2")

        result = await repository.update_chat_task_status("task-1", ChatTaskStatus.failed, "Request blocked.", clear_streaming_state=True)

        self.assertIs(result, task)
        self.assertEqual(task.status, _to_prisma_task_status(ChatTaskStatus.failed))
        self.assertEqual(db.conversation.activeTaskId, "task-2")
        db.execute_raw.assert_awaited_once()

    async def test_terminal_clear_never_downgrades_an_already_terminal_task(self) -> None:
        for terminal_status in (ChatTaskStatus.failed, ChatTaskStatus.cancelled):
            with self.subTest(status=terminal_status):
                task = _chat_task(terminal_status)
                repository, db = _task_repository(task)

                result = await repository.update_chat_task_status("task-1", ChatTaskStatus.failed, "Request blocked.", clear_streaming_state=True)

                self.assertIsNone(result)
                self.assertEqual(task.status, _to_prisma_task_status(terminal_status))
                self.assertEqual(task.streamingState["content"], "private partial")
                self.assertEqual(task.streamingState["version"], 4)
                self.assertEqual(db.conversation.activeTaskId, "task-1")
                db.execute_raw.assert_not_awaited()

    async def test_late_stream_write_after_terminal_clear_is_rejected(self) -> None:
        task = _chat_task(ChatTaskStatus.pending)
        repository, _db = _task_repository(task)
        await repository.update_chat_task_status("task-1", ChatTaskStatus.failed, "Request blocked.", clear_streaming_state=True)

        result = await repository.update_chat_task_streaming_state("task-1", "raw late partial", [], [], current_version=5)

        self.assertIsNone(result)
        self.assertEqual(task.streamingState["content"], "")
        self.assertEqual(task.streamingState["version"], 5)

    async def test_streaming_writes_are_accepted_only_while_running(self) -> None:
        task = _chat_task(ChatTaskStatus.pending)
        repository, _db = _task_repository(task)

        result = await repository.update_chat_task_streaming_state("task-1", "early partial", [], [], current_version=4)

        self.assertIsNone(result)
        self.assertEqual(task.streamingState["content"], "private partial")


class _Psycopg2QueryRaw:
    """Runs repository SQL on a real connection and returns Prisma-like row dictionaries."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    async def query_raw(self, query: str) -> list[dict[str, Any]]:
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            columns = [column.name for column in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


@unittest.skipUnless(
    CONVERSATION_WINDOW_TEST_DATABASE_URL and psycopg2,
    "requires disposable CONVERSATION_WINDOW_TEST_DATABASE_URL and psycopg2",
)
class LatestExchangeDenialProjectionPostgresTests(IsolatedAsyncioTestCase):
    """Executes the production latest-exchange SQL so denial notices keep only whitelisted events."""

    TIMESTAMP = "2026-01-01T00:00:00+00:00"

    async def asyncSetUp(self) -> None:
        assert psycopg2 is not None
        self.connection = psycopg2.connect(CONVERSATION_WINDOW_TEST_DATABASE_URL)
        self.addCleanup(self.connection.close)
        self.addCleanup(self.connection.rollback)
        messages = [
            {"role": "user", "content": "Blocked question", "timestamp": self.TIMESTAMP, "message_id": "user-1"},
            {
                "role": "assistant",
                "content": "Request blocked.",
                "timestamp": self.TIMESTAMP,
                "message_id": "notice-1",
                "tool_calls": [{"id": "call-1", "output": "private tool output"}],
                "events": [
                    {"type": "tool", "output": "private tool output"},
                    {
                        "type": "content_protection_denial",
                        "channel": "final",
                        "code": "content_denied",
                        "public_reason": "Safe reason",
                        "request_id": "request-1",
                        "timestamp": self.TIMESTAMP,
                        "raw_output": "private denial detail",
                    },
                ],
            },
        ]
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE TEMP TABLE users (id text PRIMARY KEY, username text, display_name text)")
            cursor.execute(
                """CREATE TEMP TABLE conversations (
                    id text PRIMARY KEY, title text, model text, user_id text, workspace_id text,
                    total_tokens integer, active_task_id text, active_branch_id text,
                    disabled_builtin_tool_ids jsonb, subagents_enabled boolean,
                    parent_conversation_id text, subagent_role text, subagent_index integer,
                    tool_selection_mode text, tool_output_mode text,
                    created_at timestamptz, updated_at timestamptz, messages jsonb)"""
            )
            cursor.execute(
                """INSERT INTO conversations VALUES (
                    'conversation-1', 'Denied chat', 'model-1', NULL, NULL, 0, NULL, NULL,
                    '[]'::jsonb, TRUE, NULL, NULL, NULL, 'custom', 'default', now(), now(), %s::jsonb)""",
                (json.dumps(messages),),
            )

    async def test_latest_exchange_sql_keeps_only_whitelisted_denial_event_fields(self) -> None:
        repository = IndexerRepository()
        adapter = _Psycopg2QueryRaw(self.connection)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=adapter)):
            window = await repository._build_conversation_window("conversation-1", before_index=None, latest_exchange=True)

        assert window is not None
        notice = next(entry for entry in window.entries if entry.key == "notice-1")
        self.assertIsNotNone(notice.preview)
        assert notice.preview is not None
        self.assertEqual(notice.preview.content, "Request blocked.")
        self.assertEqual(
            notice.preview.events,
            [
                {
                    "type": "content_protection_denial",
                    "channel": "final",
                    "code": "content_denied",
                    "public_reason": "Safe reason",
                    "request_id": "request-1",
                    "timestamp": self.TIMESTAMP,
                }
            ],
        )
        self.assertEqual(window.conversation.last_denial_message_id, "notice-1")
        self.assertNotIn("private tool output", json.dumps(window.model_dump(mode="json")))
        self.assertNotIn("private denial detail", json.dumps(window.model_dump(mode="json")))
