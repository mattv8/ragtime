"""Disposable-Postgres coverage for the received-share repository query.

Set RECEIVED_SHARES_TEST_DATABASE_URL to a database created solely for this
test. The test uses temporary tables in one transaction and rolls it back.
"""

import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

try:
    import psycopg2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional test dependency
    psycopg2 = cast(Any, None)

from ragtime.indexer.repository import repository  # noqa: E402

DATABASE_URL = os.environ.get("RECEIVED_SHARES_TEST_DATABASE_URL")
NOW = datetime(2020, 1, 1, tzinfo=timezone.utc)


@unittest.skipUnless(DATABASE_URL and psycopg2, "requires disposable RECEIVED_SHARES_TEST_DATABASE_URL and psycopg2")
class ReceivedConversationSharesPostgresTests(unittest.TestCase):
    def setUp(self) -> None:
        assert psycopg2 is not None
        self.connection = psycopg2.connect(DATABASE_URL)
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE TEMP TABLE users (id text PRIMARY KEY, username text, display_name text)")
            cursor.execute("CREATE TEMP TABLE conversations (id text PRIMARY KEY, title text, user_id text, workspace_id text, parent_conversation_id text)")
            cursor.execute(
                """CREATE TEMP TABLE conversation_shares (
                    id text PRIMARY KEY, conversation_id text, owner_user_id text,
                    share_access_mode text, share_selected_user_ids jsonb, share_token text,
                    label text, granted_role text, scope_anchor_message_idx integer,
                    scope_direction text, created_at timestamp)"""
            )
            cursor.execute("INSERT INTO users VALUES ('owner', 'alice', 'Alice'), ('recipient', 'bob', 'Bob')")
            cursor.execute(
                """INSERT INTO conversations VALUES
                ('old', 'An old personal chat', 'owner', NULL, NULL),
                ('workspace', 'Workspace', 'owner', 'workspace-1', NULL),
                ('child', 'Subagent', 'owner', NULL, 'old'),
                ('own', 'My chat', 'recipient', NULL, NULL)"""
            )
            cursor.execute(
                """INSERT INTO conversation_shares
                (id, conversation_id, owner_user_id, share_access_mode, share_selected_user_ids,
                 share_token, label, granted_role, scope_anchor_message_idx, scope_direction, created_at)
                VALUES
                ('s2', 'old', 'owner', 'selected_users', '["recipient"]', 'token2', 'Later', 'editor', 3, 'forward', %s),
                ('s1', 'old', 'owner', 'selected_users', '["recipient"]', 'token1', 'Earlier', 'viewer', 1, 'backward', %s),
                ('other', 'old', 'owner', 'selected_users', '["other"]', 'other', NULL, 'viewer', NULL, NULL, '2026-01-01'),
                ('substring', 'old', 'owner', 'selected_users', '["recipients"]', 'substring', NULL, 'viewer', NULL, NULL, '2026-01-01'),
                ('token', 'old', 'owner', 'token', '["recipient"]', 'public', NULL, 'viewer', NULL, NULL, '2026-01-01'),
                ('workspace', 'workspace', 'owner', 'selected_users', '["recipient"]', 'workspace', NULL, 'viewer', NULL, NULL, '2026-01-01'),
                ('child', 'child', 'owner', 'selected_users', '["recipient"]', 'child', NULL, 'viewer', NULL, NULL, '2026-01-01'),
                ('own', 'own', 'recipient', 'selected_users', '["recipient"]', 'own', NULL, 'viewer', NULL, NULL, '2026-01-01'),
                ('blank', 'old', 'owner', 'selected_users', '["recipient"]', '  ', NULL, 'viewer', NULL, NULL, '2026-01-01')""",
                (NOW, NOW),
            )

        async def query_raw(query: str) -> list[dict[str, Any]]:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                columns = [column.name for column in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        self.db = SimpleNamespace(query_raw=AsyncMock(side_effect=query_raw))

    def tearDown(self) -> None:
        self.connection.rollback()
        self.connection.close()

    def test_query_filters_recipients_and_reflects_revocation(self) -> None:
        async def run() -> None:
            with patch.object(repository, "_get_db", AsyncMock(return_value=self.db)):
                shares = await repository.list_received_conversation_shares("recipient")
                self.assertEqual([share.id for share in shares], ["s2", "s1"])
                self.assertEqual((shares[0].scope_anchor_message_idx, shares[1].scope_direction), (3, "backward"))
                self.assertEqual(await repository.list_received_conversation_shares("admin-not-selected"), [])

                first = await repository.list_received_conversation_shares("recipient", limit=1)
                second = await repository.list_received_conversation_shares(
                    "recipient", limit=1, cursor_created_at=first[0].created_at, cursor_id=first[0].id
                )
                self.assertEqual([share.id for share in first + second], ["s2", "s1"])

                with self.connection.cursor() as cursor:
                    cursor.execute("UPDATE conversation_shares SET share_selected_user_ids = '[]' WHERE id = 's2'")
                self.assertEqual([share.id for share in await repository.list_received_conversation_shares("recipient")], ["s1"])
                with self.connection.cursor() as cursor:
                    cursor.execute("UPDATE conversation_shares SET share_access_mode = 'token' WHERE id = 's1'")
                self.assertEqual(await repository.list_received_conversation_shares("recipient"), [])
                with self.connection.cursor() as cursor:
                    cursor.execute("UPDATE conversation_shares SET share_access_mode = 'selected_users' WHERE id = 's1'")
                    cursor.execute("DELETE FROM conversation_shares WHERE id = 's1'")
                    cursor.execute(
                        "UPDATE conversation_shares SET share_selected_user_ids = %s::jsonb WHERE id = 's2'",
                        (json.dumps(["quote'\\slash\"unicode-é"]),),
                    )
                self.assertEqual(
                    [share.id for share in await repository.list_received_conversation_shares("quote'\\slash\"unicode-é")],
                    ["s2"],
                )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
