from __future__ import annotations

import asyncio
import logging
import unittest
from itertools import count
from typing import Any
from unittest import mock

from prisma._types import TransactionId

from ragtime.config import settings
from ragtime.core.database import DatabaseManager, InstrumentedPrisma


class _FakeEngine:
    def __init__(self) -> None:
        self.error: BaseException | None = None
        self.block: asyncio.Event | None = None
        self.queries: list[str] = []
        self.committed: list[str] = []

    async def query(self, query: str, *, tx_id: str | None = None) -> dict[str, Any]:
        self.queries.append(query)
        if self.block is not None:
            await self.block.wait()
        if self.error is not None:
            raise self.error
        if "queryRaw" in query:
            return {"data": {"result": {"columns": [], "types": [], "rows": []}}}
        return {"data": {"result": []}}

    async def start_transaction(self, *, content: str) -> TransactionId:
        return TransactionId("test-transaction")

    async def commit_transaction(self, tx_id: str) -> None:
        self.committed.append(tx_id)

    async def rollback_transaction(self, tx_id: str) -> None:
        del tx_id

    def stop(self, timeout: object = None) -> None:
        del timeout


class InstrumentedPrismaTests(unittest.IsolatedAsyncioTestCase):
    def _client(self, engine: _FakeEngine) -> InstrumentedPrisma:
        client = InstrumentedPrisma(http={"timeout": 12.5})
        client._engine = engine  # type: ignore[assignment]
        return client

    async def test_regular_raw_errors_cancellation_and_transaction_copies_are_timed_without_payloads(self) -> None:
        engine = _FakeEngine()
        client = self._client(engine)

        with (
            mock.patch("ragtime.core.performance.monotonic", side_effect=count(0.0, 2.0).__next__),
            self.assertLogs("ragtime.performance", level=logging.WARNING) as logs,
        ):
            self.assertEqual(await client.user.find_many(), [])
            self.assertEqual(await client.query_raw("SELECT secret_value FROM private_table", "private-param"), [])

            engine.error = RuntimeError("secret database failure")
            with self.assertRaisesRegex(RuntimeError, "secret database failure"):
                await client.user.find_many()
            engine.error = None

            async with client.tx() as transaction:
                self.assertIsInstance(transaction, InstrumentedPrisma)
                self.assertEqual(await transaction.user.find_many(), [])
            self.assertEqual(engine.committed, ["test-transaction"])

            engine.block = asyncio.Event()
            task = asyncio.create_task(client.query_raw("SELECT cancelled_secret", "cancelled-param"))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        rendered = "\n".join(logs.output)
        self.assertIn("db.User.find_many", rendered)
        self.assertIn("db.raw.query_raw", rendered)
        self.assertNotIn("secret_value", rendered)
        self.assertNotIn("private-param", rendered)
        self.assertNotIn("database failure", rendered)

    def test_manager_uses_instrumented_client_and_preserves_http_timeout(self) -> None:
        manager = DatabaseManager()
        with mock.patch.object(InstrumentedPrisma, "is_connected", return_value=True):
            client = asyncio.run(manager.connect())
        self.assertIsInstance(client, InstrumentedPrisma)
        self.assertEqual(client._http_config["timeout"], settings.prisma_timeout)
