import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from ragtime.pdm_automation.repository import PdmAutomationRepository


class _Tx:
    def __init__(self):
        self.sql = []

    async def execute_raw(self, query):
        self.sql.append(query)
        return 1

    async def query_raw(self, query):
        self.sql.append(query)
        return []


class _Context:
    def __init__(self, tx):
        self.tx = tx

    async def __aenter__(self):
        return self.tx

    async def __aexit__(self, *_):
        return None


class _Db:
    def __init__(self):
        self.transaction = _Tx()

    def tx(self):
        return _Context(self.transaction)

    async def execute_raw(self, query):
        return await self.transaction.execute_raw(query)

    async def query_raw(self, query):
        return await self.transaction.query_raw(query)


class PdmAutomationRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_last_attempt_at_normalizes_raw_timestamp_representations(self):
        cases = [
            ("Z string", [{"last_attempt_at": "2026-09-25T12:00:00Z"}], datetime(2026, 9, 25, 12, tzinfo=timezone.utc)),
            ("offset string", [{"last_attempt_at": "2026-09-25T14:00:00+02:00"}], datetime(2026, 9, 25, 12, tzinfo=timezone.utc)),
            ("naive string", [{"last_attempt_at": "2026-09-25T12:00:00"}], datetime(2026, 9, 25, 12, tzinfo=timezone.utc)),
            ("naive datetime", [{"last_attempt_at": datetime(2026, 9, 25, 12)}], datetime(2026, 9, 25, 12, tzinfo=timezone.utc)),
            (
                "aware datetime",
                [{"last_attempt_at": datetime(2026, 9, 25, 14, tzinfo=timezone(timedelta(hours=2)))}],
                datetime(2026, 9, 25, 12, tzinfo=timezone.utc),
            ),
            ("missing state", [], None),
            ("null timestamp", [{"last_attempt_at": None}], None),
        ]
        for name, rows, expected in cases:
            with self.subTest(name):
                db = mock.Mock(query_raw=mock.AsyncMock(return_value=rows))
                with mock.patch("ragtime.pdm_automation.repository.get_db", mock.AsyncMock(return_value=db)):
                    self.assertEqual(await PdmAutomationRepository().last_attempt_at("tool-1"), expected)

    async def test_automation_claim_links_job_in_the_admission_transaction(self):
        tx = _Tx()
        tx.query_raw = mock.AsyncMock(
            side_effect=[
                [{"pending_webhook": True, "pending_schedule": False, "pending_generation": 1}],
                [{"due": True}],
            ]
        )
        claimed = await PdmAutomationRepository().claim_for_admission(tx, "tool-1", "job-1")
        self.assertTrue(claimed)
        self.assertTrue(any("active_job_id='job-1'" in query for query in tx.sql))
        self.assertTrue(any("pending_webhook=FALSE" in query for query in tx.sql))
