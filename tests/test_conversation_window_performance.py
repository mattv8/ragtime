"""Opt-in Prisma/PostgreSQL coverage for bounded conversation windows."""

import json
import os
import unittest
import uuid
from unittest import mock

from prisma import Prisma

from ragtime.core.sql import sql_quote_literal
from ragtime.indexer.repository import repository


@unittest.skipUnless(os.environ.get("CONVERSATION_WINDOW_TEST_DATABASE_URL"), "requires isolated PostgreSQL")
class ConversationWindowPostgresPerformanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db = Prisma(datasource={"url": os.environ["CONVERSATION_WINDOW_TEST_DATABASE_URL"]})
        self.tx_context = None
        self.db_patch = None
        await self.db.connect()
        try:
            self.tx_context = self.db.tx()
            self.tx = await self.tx_context.__aenter__()
            self.conversation_id = str(uuid.uuid4())
            self.large_event = "event-output-" + "x" * (256 * 1024)
            self.messages = [
                {"role": "user", "content": "small", "timestamp": "2026-09-16T12:00:00+00:00", "message_id": "user-1"},
                {
                    "role": "assistant",
                    "content": "",
                    "events": [
                        {"type": "content", "content": self.large_event},
                        {"type": "tool", "tool": "synthetic", "input": {"kind": "benchmark"}, "output": self.large_event},
                    ],
                    "tool_calls": [{"tool": "synthetic", "input": {"kind": "benchmark"}, "output": self.large_event}],
                    "timestamp": "2026-09-16T12:00:01+00:00",
                    "message_id": "assistant-1",
                },
                {"role": "assistant", "content": "y" * (100 * 1024), "timestamp": "2026-09-16T12:00:02+00:00", "message_id": "assistant-2"},
            ]
            await self.tx.execute_raw(
                f"INSERT INTO conversations(id, title, model, messages) VALUES "
                f"({sql_quote_literal(self.conversation_id)}, 'window performance', 'test', {sql_quote_literal(json.dumps(self.messages))}::jsonb)"
            )
            self.db_patch = mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=self.tx))
            self.db_patch.start()
        except Exception:
            if self.tx_context is not None:
                await self.tx_context.__aexit__(Exception, Exception(), None)
            await self.db.disconnect()
            raise

    async def asyncTearDown(self) -> None:
        if self.db_patch is not None:
            self.db_patch.stop()
        if self.tx_context is not None:
            await self.tx_context.__aexit__(Exception, Exception(), None)
        await self.db.disconnect()

    async def test_oversized_entry_is_bounded_and_full_read_is_identical(self) -> None:
        window = await repository._build_conversation_window(self.conversation_id, before_index=None, latest_exchange=False, limit=20)

        assert window is not None
        self.assertEqual([entry.index for entry in window.entries], [0, 1, 2])
        self.assertEqual(len(window.revision), 64)
        self.assertEqual(window.entries[1].state, "deferred")
        self.assertEqual(window.entries[1].preview.content if window.entries[1].preview else None, self.large_event[:1024])
        self.assertTrue(window.entries[1].preview.content_truncated if window.entries[1].preview else False)
        self.assertEqual(window.entries[2].state, "ready")
        self.assertLess(len(json.dumps(window.model_dump(mode="json")).encode()), 256 * 1024)

        full_entry = await repository.get_conversation_window_message(self.conversation_id, 1, window.revision)

        assert full_entry is not None and full_entry.message is not None
        assert full_entry.message.tool_calls is not None
        self.assertEqual(full_entry.state, "ready")
        self.assertEqual(full_entry.message.tool_calls[0].output, self.large_event)
        expected = repository._parse_messages_json([self.messages[1]])[0]
        self.assertEqual(full_entry.message.model_dump(mode="json"), expected.model_dump(mode="json"))
