import unittest
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.sqlite_backup_queue_store import SqliteBackupQueueStore


class _Tx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def query_raw(self, query, *params):
        self.calls.append((query, params))
        if "request_key" in query and "SELECT" in query:
            return self.existing
        if "global_count" in query:
            return [{"global_count": 0, "workspace_count": 0}]
        if "INSERT" in query:
            return [self.row]
        return []

    def __init__(self):
        self.calls = []
        self.existing = []
        self.row = {
            "id": "job",
            "workspace_id": "ws",
            "trigger": "manual",
            "database_names": [],
            "request_key": "key",
            "request_hash": "hash",
            "status": "pending",
        }


class _Db:
    def __init__(self):
        self.transaction = _Tx()

    def tx(self):
        return self.transaction


class SqliteBackupQueueStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_normalizes_database_names_and_uses_advisory_transaction_lock(self):
        db = _Db()
        with mock.patch("ragtime.userspace.sqlite_backup_queue_store.get_db", new=mock.AsyncMock(return_value=db)):
            await SqliteBackupQueueStore().enqueue("ws", trigger="manual", database_names=["b.sqlite3", "a.sqlite3", "a.sqlite3"], request_key="key")
        insert = next(params for query, params in db.transaction.calls if "INSERT INTO" in query)
        self.assertEqual(insert[3], ["a.sqlite3", "b.sqlite3"])
        self.assertTrue(any("pg_advisory_xact_lock" in query for query, _ in db.transaction.calls))

    async def test_conflicting_idempotency_key_returns_409_before_capacity_check(self):
        db = _Db()
        db.transaction.existing = [{**db.transaction.row, "request_hash": "different"}]
        with mock.patch("ragtime.userspace.sqlite_backup_queue_store.get_db", new=mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as raised:
                await SqliteBackupQueueStore().enqueue("ws", trigger="manual", request_key="key")
        self.assertEqual(raised.exception.status_code, 409)
        self.assertFalse(any("global_count" in query for query, _ in db.transaction.calls))

    async def test_claim_uses_available_fifo_skip_locked_and_owner_fence(self):
        db = _Db()
        with mock.patch("ragtime.userspace.sqlite_backup_queue_store.get_db", new=mock.AsyncMock(return_value=db)):
            await SqliteBackupQueueStore().claim_next("owner")
        claim = next(query for query, _ in db.transaction.calls if "WITH next_job" in query)
        self.assertIn("available_at <= NOW()", claim)
        self.assertIn("FOR UPDATE SKIP LOCKED", claim)
        self.assertIn("owner_token = $1", claim)
        self.assertIn("running.status = 'running'", claim)

    async def test_ownership_loss_is_cancelled_but_database_errors_propagate(self):
        class MissingJobDb:
            async def query_raw(self, *_):
                return []

        with mock.patch("ragtime.userspace.sqlite_backup_queue_store.get_db", new=mock.AsyncMock(return_value=MissingJobDb())):
            self.assertTrue(await SqliteBackupQueueStore().is_cancel_requested("job", "owner"))

        class BrokenDb:
            async def query_raw(self, *_):
                raise RuntimeError("database unavailable")

        with mock.patch("ragtime.userspace.sqlite_backup_queue_store.get_db", new=mock.AsyncMock(return_value=BrokenDb())):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await SqliteBackupQueueStore().is_cancel_requested("job", "owner")

    async def test_explicit_empty_or_invalid_database_names_are_rejected(self):
        with self.assertRaises(HTTPException) as empty:
            await SqliteBackupQueueStore().enqueue("ws", trigger="manual", database_names=[], request_key="key")
        self.assertEqual(empty.exception.status_code, 400)
        with self.assertRaises(HTTPException) as invalid:
            await SqliteBackupQueueStore().enqueue("ws", trigger="manual", database_names=["../app.sqlite3"], request_key="key")
        self.assertEqual(invalid.exception.status_code, 400)
