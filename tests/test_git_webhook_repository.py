import asyncio
import importlib
import re
import unittest
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest import mock

from ragtime.git_webhooks.models import GitPushEvent, GitWebhookDeliveryStatus, GitWebhookTarget, GitWebhookTargetType
from ragtime.git_webhooks.repository import GitWebhookRepository

UniqueViolationError: Any = getattr(importlib.import_module("prisma.errors"), "UniqueViolationError")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _delivery_row(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": "delivery-1",
        "target_type": "git_index",
        "index_metadata_id": "index-id",
        "workspace_id": None,
        "provider_delivery_id": "delivery-provider-1",
        "event_name": "push",
        "branch": "main",
        "head_commit": "abc123",
        "status": "pending",
        "index_job_id": None,
        "message": None,
        "received_at": _utcnow(),
        "started_at": None,
        "completed_at": None,
    }
    values.update(overrides)
    return values


def _target_row(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "id": "index-id",
        "name": "git-index",
        "configSnapshot": None,
        "webhookId": "wh_123",
        "webhookSecret": None,
        "webhookPaused": False,
        "webhookCreatedAt": _utcnow(),
        "scmWebhookId": None,
        "scmWebhookSecret": None,
        "scmWebhookPaused": False,
        "scmWebhookCreatedAt": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeTransaction:
    def __init__(
        self,
        *,
        query_results: list[list[dict[str, Any]]] | None = None,
        duplicate_rows: list[dict[str, Any]] | None = None,
        duplicate_query_results: list[list[dict[str, Any]]] | None = None,
        claim_rows: list[dict[str, Any]] | None = None,
        list_rows: list[dict[str, Any]] | None = None,
        recoverable_rows: list[dict[str, Any]] | None = None,
        newer_pending_rows: list[dict[str, Any]] | None = None,
        pending_rows: list[dict[str, Any]] | None = None,
        reset_pending_rows: list[dict[str, Any]] | None = None,
        reset_status_rows: list[dict[str, Any]] | None = None,
        reset_update_exception: Exception | None = None,
        execute_results: list[int] | None = None,
        execute_exceptions: list[Exception | None] | None = None,
        insert_exception: Exception | None = None,
    ) -> None:
        self.query_results = list(query_results or [])
        self.duplicate_rows = list(duplicate_rows or [])
        self.duplicate_query_results = list(duplicate_query_results or [])
        self.claim_rows = list(claim_rows or [])
        self.list_rows = list(list_rows or [])
        self.recoverable_rows = list(recoverable_rows or [])
        self.newer_pending_rows = list(newer_pending_rows or [])
        self.pending_rows = list(pending_rows or [])
        self.reset_pending_rows = list(reset_pending_rows or [])
        self.reset_status_rows = list(reset_status_rows or [])
        self.reset_update_exception = reset_update_exception
        self.execute_results = list(execute_results or [])
        self.execute_exceptions = list(execute_exceptions or [])
        self.insert_exception = insert_exception
        self.executed: list[str] = []

    async def query_raw(self, query: str) -> list[dict[str, Any]]:
        self.executed.append(query)
        normalized = " ".join(query.split())
        if "FROM git_webhook_deliveries" in normalized and "provider_delivery_id =" in normalized and "LIMIT 1" in normalized:
            if self.duplicate_query_results:
                return self.duplicate_query_results.pop(0)
            return list(self.duplicate_rows)
        if "FROM git_webhook_deliveries" in normalized and "WHERE id =" in normalized and "LIMIT 1" in normalized:
            if self.query_results:
                return self.query_results.pop(0)
            return []
        if normalized.startswith("INSERT INTO git_webhook_deliveries"):
            if self.insert_exception is not None:
                raise self.insert_exception
            if self.query_results:
                return self.query_results.pop(0)
            return []
        if "SET status = 'processing'" in normalized and "RETURNING" in normalized:
            return list(self.claim_rows)
        if "SELECT 1 AS exists" in normalized:
            return list(self.newer_pending_rows)
        if normalized.startswith("UPDATE git_webhook_deliveries") and "SET status = 'pending'" in normalized and "RETURNING status" in normalized:
            if self.reset_update_exception is not None:
                raise self.reset_update_exception
            if self.insert_exception is not None:
                raise self.insert_exception
            if self.reset_status_rows:
                return list(self.reset_status_rows)
            return [{"status": "pending"}]
        if normalized.startswith("SELECT") and "FROM git_webhook_deliveries" in normalized and "status = 'pending'" in normalized and "id <>" in normalized:
            return list(self.reset_pending_rows or self.pending_rows)
        if (
            normalized.startswith("SELECT")
            and "status = 'pending'" in normalized
            and "LIMIT 1" in normalized
            and "WHERE id =" not in normalized
            and "provider_delivery_id =" not in normalized
        ):
            return list(self.pending_rows)
        if "FROM git_webhook_deliveries" in normalized and "status = 'processing'" in normalized:
            return list(self.recoverable_rows)
        if normalized.startswith("SELECT") and "FROM git_webhook_deliveries" in normalized and "ORDER BY received_at DESC" in normalized:
            return list(self.list_rows)
        if self.query_results:
            return self.query_results.pop(0)
        return []

    async def execute_raw(self, query: str) -> int:
        self.executed.append(query)
        if self.execute_exceptions:
            exc = self.execute_exceptions.pop(0)
            if exc is not None:
                raise exc
        if self.execute_results:
            return self.execute_results.pop(0)
        return 1


class _TxContext(AbstractAsyncContextManager[FakeTransaction]):
    def __init__(self, tx: FakeTransaction) -> None:
        self._tx = tx

    async def __aenter__(self) -> FakeTransaction:
        return self._tx

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _LookupDelegate:
    def __init__(self, row: SimpleNamespace | None = None) -> None:
        self.row = row
        self.updated: list[dict[str, Any]] = []

    async def find_unique(self, **_kwargs: Any) -> SimpleNamespace | None:
        return self.row

    async def update(self, *, data: dict[str, Any], **_kwargs: Any) -> SimpleNamespace:
        self.updated.append(data)
        if self.row is None:
            raise AssertionError("update called without a row")
        for key, value in data.items():
            setattr(self.row, key, value)
        return self.row


class FakeDb:
    def __init__(self, tx: FakeTransaction) -> None:
        self._tx = tx
        self.indexmetadata = _LookupDelegate()
        self.workspace = _LookupDelegate()

    def tx(self) -> _TxContext:
        return _TxContext(self._tx)

    async def query_raw(self, query: str) -> list[dict[str, Any]]:
        return await self._tx.query_raw(query)

    async def execute_raw(self, query: str) -> int:
        return await self._tx.execute_raw(query)


class FakeLookupDb:
    def __init__(
        self,
        row: SimpleNamespace | None,
        *,
        target_type: GitWebhookTargetType = GitWebhookTargetType.GIT_INDEX,
        tx: FakeTransaction | None = None,
    ) -> None:
        self._tx = tx or FakeTransaction()
        self.indexmetadata = _LookupDelegate(row if target_type is GitWebhookTargetType.GIT_INDEX else None)
        self.workspace = _LookupDelegate(row if target_type is GitWebhookTargetType.WORKSPACE_SCM else None)

    def tx(self) -> _TxContext:
        return _TxContext(self._tx)

    async def query_raw(self, query: str) -> list[dict[str, Any]]:
        return await self._tx.query_raw(query)

    async def execute_raw(self, query: str) -> int:
        return await self._tx.execute_raw(query)


def _query_target_type(normalized: str) -> str:
    if " workspace_id = " in normalized or '("workspace_id")' in normalized or " workspace_id)" in normalized:
        return "workspace_scm"
    return "git_index"


def _sql_literal_at(values_sql: str, index: int) -> str | None:
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    i = 0
    while i < len(values_sql):
        char = values_sql[i]
        if char == "'":
            current.append(char)
            if in_quote and i + 1 < len(values_sql) and values_sql[i + 1] == "'":
                current.append(values_sql[i + 1])
                i += 1
            else:
                in_quote = not in_quote
        elif char == "," and not in_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1
    if current:
        parts.append("".join(current).strip())
    if index >= len(parts):
        return None
    value = parts[index]
    if value == "NULL":
        return None
    return value.strip("'")


class _ConcurrentEnableTransaction:
    def __init__(self, db: "ConcurrentEnableDb") -> None:
        self._db = db

    async def query_raw(self, query: str) -> list[dict[str, Any]]:
        return []

    async def execute_raw(self, query: str) -> int:
        self._db.executed.append(query)
        if "pg_advisory_xact_lock" in query:
            match = re.search(r"git-webhook:[^']+", query)
            if match is None:
                raise AssertionError("missing advisory lock key")
            self._db.last_lock_key = match.group(0)
        return 1


class _ConcurrentEnableContext(AbstractAsyncContextManager[_ConcurrentEnableTransaction]):
    def __init__(self, db: "ConcurrentEnableDb") -> None:
        self._db = db
        self._tx = _ConcurrentEnableTransaction(db)

    async def __aenter__(self) -> _ConcurrentEnableTransaction:
        await self._db.lock.acquire()
        return self._tx

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._db.lock.release()
        return None


class ConcurrentEnableDb:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.executed: list[str] = []
        self.last_lock_key: str | None = None
        self.row = _target_row(webhookId=None, webhookSecret=None, webhookCreatedAt=None)
        self.indexmetadata = _LookupDelegate(self.row)
        self.workspace = _LookupDelegate()

    def tx(self) -> _ConcurrentEnableContext:
        return _ConcurrentEnableContext(self)


class _ConcurrentQueueTransaction:
    def __init__(self, db: "ConcurrentQueueDb") -> None:
        self._db = db

    async def execute_raw(self, query: str) -> int:
        self._db.executed.append(query)
        return 1

    async def query_raw(self, query: str) -> list[dict[str, Any]]:
        self._db.executed.append(query)
        normalized = " ".join(query.split())
        target_type = _query_target_type(normalized)
        target_id = "workspace-1" if target_type == "workspace_scm" else "index-id"
        if "provider_delivery_id" in normalized and "LIMIT 1" in normalized:
            provider_match = re.search(r"provider_delivery_id = '([^']+)'", normalized)
            provider_delivery_id = provider_match.group(1) if provider_match else None
            return [row.copy() for row in self._db.deliveries if row["target_type"] == target_type and row["provider_delivery_id"] == provider_delivery_id]
        if normalized.startswith("UPDATE git_webhook_deliveries") and "status = 'skipped'" in normalized:
            pending = [
                row
                for row in self._db.deliveries
                if row["target_type"] == target_type and row[self._db.target_field(target_type)] == target_id and row["status"] == "pending"
            ]
            if not pending:
                return []
            latest = pending[-1]
            latest["status"] = "skipped"
            return [latest.copy()]
        if normalized.startswith("INSERT INTO git_webhook_deliveries"):
            values_match = re.search(r"VALUES \((.*)\) RETURNING", normalized)
            values_sql = values_match.group(1) if values_match else ""
            row = _delivery_row(
                id=f"delivery-{len(self._db.deliveries) + 1}",
                target_type=target_type,
                index_metadata_id=target_id if target_type == "git_index" else None,
                workspace_id=target_id if target_type == "workspace_scm" else None,
                provider_delivery_id=_sql_literal_at(values_sql, 3),
                event_name=_sql_literal_at(values_sql, 4) or "push",
                branch=_sql_literal_at(values_sql, 5) or "main",
                head_commit=_sql_literal_at(values_sql, 6),
                status="pending",
            )
            self._db.deliveries.append(row)
            return [row.copy()]
        return []


class _ConcurrentQueueContext(AbstractAsyncContextManager[_ConcurrentQueueTransaction]):
    def __init__(self, db: "ConcurrentQueueDb") -> None:
        self._db = db
        self._tx = _ConcurrentQueueTransaction(db)

    async def __aenter__(self) -> _ConcurrentQueueTransaction:
        await self._db.lock.acquire()
        return self._tx

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._db.lock.release()
        return None


class ConcurrentQueueDb:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.executed: list[str] = []
        self.deliveries: list[dict[str, Any]] = []
        self.indexmetadata = _LookupDelegate()
        self.workspace = _LookupDelegate()

    @staticmethod
    def target_field(target_type: str) -> str:
        return "workspace_id" if target_type == "workspace_scm" else "index_metadata_id"

    def tx(self) -> _ConcurrentQueueContext:
        return _ConcurrentQueueContext(self)


class GitWebhookRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_target_accepts_git_webhook_target_type_enum(self) -> None:
        repository = GitWebhookRepository()
        row = _target_row(id="index-id", webhookId="wh_123", webhookSecret="enc::secret")
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeLookupDb(row)):
            target = await repository.resolve_target(GitWebhookTargetType.GIT_INDEX, "index-id")
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.target_type, GitWebhookTargetType.GIT_INDEX)
        self.assertEqual(target.target_id, "index-id")

    async def test_enqueue_push_supersedes_only_the_previous_pending_delivery(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(
            query_results=[
                [{"id": "old-pending"}],
                [{"id": "new-delivery", "status": "pending"}],
            ]
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            delivery = await repository.enqueue_push(
                target_type=GitWebhookTargetType.GIT_INDEX,
                target_id="index-id",
                provider_delivery_id="delivery-2",
                event_name="push",
                branch="main",
                head_commit="abc123",
            )
        self.assertEqual(delivery.id, "new-delivery")
        self.assertTrue(any("status = 'skipped'" in query for query in tx.executed))

    async def test_link_index_job_updates_only_processing_delivery(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(execute_results=[1])
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            await repository.link_index_job("delivery-1", "job-1")
        update_queries = [query for query in tx.executed if "SET index_job_id = 'job-1'" in query]
        self.assertEqual(len(update_queries), 1)
        self.assertIn("WHERE id = 'delivery-1'", update_queries[0])
        self.assertIn("status = 'processing'", update_queries[0])

    async def test_has_pending_uses_target_specific_pending_lookup(self) -> None:
        repository = GitWebhookRepository()
        git_tx = FakeTransaction(newer_pending_rows=[{"exists": 1}])
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(git_tx)):
            git_pending = await repository.has_pending(GitWebhookTargetType.GIT_INDEX, "index-id")
        self.assertTrue(git_pending)
        git_queries = [query for query in git_tx.executed if "SELECT 1 AS exists" in query]
        self.assertEqual(len(git_queries), 1)
        self.assertIn("index_metadata_id = 'index-id'", git_queries[0])
        self.assertNotIn("workspace_id = 'index-id'", git_queries[0])

        workspace_tx = FakeTransaction(newer_pending_rows=[])
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(workspace_tx)):
            workspace_pending = await repository.has_pending(GitWebhookTargetType.WORKSPACE_SCM, "workspace-1")
        self.assertFalse(workspace_pending)
        workspace_queries = [query for query in workspace_tx.executed if "SELECT 1 AS exists" in query]
        self.assertEqual(len(workspace_queries), 1)
        self.assertIn("workspace_id = 'workspace-1'", workspace_queries[0])
        self.assertNotIn("index_metadata_id = 'workspace-1'", workspace_queries[0])

    async def test_pruning_keeps_newest_fifty_and_never_deletes_active_rows(self) -> None:
        repository = GitWebhookRepository()
        target = GitWebhookTarget(
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id="index-id",
            webhook_id="wh_123",
            secret="plain",
        )
        event = GitPushEvent(provider_delivery_id="provider-1", event_name="push", branch="main", head_commit="abc123")
        tx = FakeTransaction(
            query_results=[
                [_delivery_row(id="ignored-1", status="ignored", provider_delivery_id="provider-1")],
                [_delivery_row(id="pending-1", status="pending", provider_delivery_id="provider-2")],
                [_delivery_row(id="processing-1", status="processing", provider_delivery_id="provider-2")],
                [_delivery_row(id="processing-1", status="processing", provider_delivery_id="provider-2")],
            ]
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            await repository.record_ignored(target, event, "ignored")
            await repository.enqueue_push(
                target_type=GitWebhookTargetType.GIT_INDEX,
                target_id="index-id",
                provider_delivery_id="provider-2",
                event_name="push",
                branch="main",
                head_commit="def456",
            )
            await repository.complete(
                "processing-1",
                status=GitWebhookDeliveryStatus.COMPLETED,
                message="done",
                index_job_id="job-1",
            )

        prune_queries = [query for query in tx.executed if query.lstrip().startswith("DELETE FROM git_webhook_deliveries")]
        self.assertEqual(len(prune_queries), 3)
        for query in prune_queries:
            self.assertIn("OFFSET 50", query)
            self.assertIn("status NOT IN ('pending', 'processing')", query)
        self.assertEqual(sum("pg_advisory_xact_lock" in query for query in tx.executed), 3)
        self.assertTrue(any("WHERE id = 'processing-1'" in query and "status = 'processing'" in query for query in tx.executed))

    async def test_duplicate_provider_delivery_returns_existing_row(self) -> None:
        repository = GitWebhookRepository()
        existing = _delivery_row(id="existing", provider_delivery_id="same", target_type="workspace_scm", index_metadata_id=None, workspace_id="workspace-1")
        tx = FakeTransaction(duplicate_rows=[existing])
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            delivery = await repository.enqueue_push(
                target_type=GitWebhookTargetType.WORKSPACE_SCM,
                target_id="workspace-1",
                provider_delivery_id="same",
                event_name="push",
                branch="main",
                head_commit="abc123",
            )
        self.assertEqual(delivery.id, "existing")
        self.assertFalse(any("INSERT INTO git_webhook_deliveries" in query for query in tx.executed))

    async def test_get_config_never_returns_encrypted_secret(self) -> None:
        repository = GitWebhookRepository()
        row = _target_row(webhookSecret="enc::ciphertext", webhookPaused=True)
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeLookupDb(row)):
            config = await repository.get_index_config("git-index", "https://ragtime.example")
        self.assertTrue(config.enabled)
        self.assertTrue(config.paused)
        self.assertFalse(hasattr(config, "secret"))
        self.assertNotIn("ciphertext", config.model_dump_json())

    async def test_get_workspace_config_reports_paused_only_when_credentials_exist(self) -> None:
        repository = GitWebhookRepository()
        row = _target_row(
            id="workspace-1",
            webhookId=None,
            webhookSecret=None,
            webhookPaused=False,
            scmWebhookId="ws_wh_123",
            scmWebhookSecret="enc::ciphertext",
            scmWebhookPaused=True,
            scmProvider="github",
            scmGitBranch="main",
        )
        with mock.patch(
            "ragtime.git_webhooks.repository.get_db",
            return_value=FakeLookupDb(row, target_type=GitWebhookTargetType.WORKSPACE_SCM),
        ):
            config = await repository.get_workspace_config("workspace-1", "https://ragtime.example")
        self.assertTrue(config.enabled)
        self.assertTrue(config.paused)
        self.assertEqual(config.webhook_url, "https://ragtime.example/webhooks/git/ws_wh_123")

    async def test_pause_index_preserves_credentials_and_skips_pending_rows(self) -> None:
        repository = GitWebhookRepository()
        original_secret = "enc::secret"
        original_created_at = _utcnow()
        row = _target_row(webhookSecret=original_secret, webhookPaused=False, webhookCreatedAt=original_created_at)
        tx = FakeTransaction(execute_results=[1, 1])
        db = FakeLookupDb(row, tx=tx)
        db.indexmetadata = _LookupDelegate(row)
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db):
            config = await repository.pause_index("git-index", "https://ragtime.example")
        updated = db.indexmetadata.updated[-1]
        self.assertTrue(config.enabled)
        self.assertTrue(config.paused)
        self.assertEqual(config.webhook_url, "https://ragtime.example/webhooks/git/wh_123")
        self.assertTrue(updated["webhookPaused"])
        self.assertEqual(row.webhookId, "wh_123")
        self.assertEqual(row.webhookSecret, original_secret)
        self.assertEqual(row.webhookCreatedAt, original_created_at)
        self.assertTrue(any("Webhook paused before processing." in query for query in tx.executed))

    async def test_pause_workspace_preserves_credentials_and_skips_pending_rows(self) -> None:
        repository = GitWebhookRepository()
        original_secret = "enc::workspace-secret"
        original_created_at = _utcnow()
        row = _target_row(
            id="workspace-1",
            webhookId=None,
            webhookSecret=None,
            scmWebhookId="ws_wh_123",
            scmWebhookSecret=original_secret,
            scmWebhookPaused=False,
            scmWebhookCreatedAt=original_created_at,
            scmProvider="github",
            scmGitBranch="main",
        )
        tx = FakeTransaction(execute_results=[1, 1])
        db = FakeLookupDb(row, target_type=GitWebhookTargetType.WORKSPACE_SCM, tx=tx)
        db.workspace = _LookupDelegate(row)
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db):
            config = await repository.pause_workspace("workspace-1", "https://ragtime.example")
        updated = db.workspace.updated[-1]
        self.assertTrue(config.enabled)
        self.assertTrue(config.paused)
        self.assertEqual(config.webhook_url, "https://ragtime.example/webhooks/git/ws_wh_123")
        self.assertTrue(updated["scmWebhookPaused"])
        self.assertEqual(row.scmWebhookId, "ws_wh_123")
        self.assertEqual(row.scmWebhookSecret, original_secret)
        self.assertEqual(row.scmWebhookCreatedAt, original_created_at)
        self.assertTrue(any("Webhook paused before processing." in query for query in tx.executed))

    async def test_resume_index_clears_paused_without_rotating_credentials(self) -> None:
        repository = GitWebhookRepository()
        original_secret = "enc::secret"
        row = _target_row(webhookSecret=original_secret, webhookPaused=True)
        tx = FakeTransaction(execute_results=[1])
        db = FakeLookupDb(row, tx=tx)
        db.indexmetadata = _LookupDelegate(row)
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db):
            config = await repository.resume_index("git-index", "https://ragtime.example")
        updated = db.indexmetadata.updated[-1]
        self.assertTrue(config.enabled)
        self.assertFalse(config.paused)
        self.assertFalse(updated["webhookPaused"])
        self.assertEqual(row.webhookSecret, original_secret)
        self.assertEqual(row.webhookId, "wh_123")

    async def test_rotate_secret_preserves_paused_flag(self) -> None:
        repository = GitWebhookRepository()
        row = _target_row(webhookSecret="enc::secret", webhookPaused=True)
        db = FakeLookupDb(row)
        db.indexmetadata = _LookupDelegate(row)
        with (
            mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db),
            mock.patch("ragtime.git_webhooks.repository.encrypt_secret", side_effect=lambda value: f"enc::{value}"),
        ):
            response = await repository.rotate_index_secret("git-index", "https://ragtime.example")
        updated = db.indexmetadata.updated[-1]
        self.assertTrue(response.enabled)
        self.assertTrue(response.paused)
        self.assertTrue(updated["webhookPaused"])

    async def test_enable_index_normalizes_schedule_snapshot_for_new_webhook(self) -> None:
        repository = GitWebhookRepository()
        original_snapshot = {
            "reindex_interval_hours": 24,
            "reindex_start_minute": 180,
            "reindex_timezone": "America/Denver",
            "chunk_size": 500,
        }
        row = _target_row(webhookId=None, webhookSecret=None, webhookCreatedAt=None, configSnapshot=original_snapshot)
        db = FakeLookupDb(row)
        db.indexmetadata = _LookupDelegate(row)
        with (
            mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db),
            mock.patch("ragtime.git_webhooks.repository.encrypt_secret", side_effect=lambda value: f"enc::{value}"),
        ):
            response = await repository.enable_index("git-index", "https://ragtime.example")
        updated = db.indexmetadata.updated[-1]
        self.assertTrue(response.enabled)
        self.assertEqual(
            updated["configSnapshot"],
            {
                "reindex_interval_hours": 0,
                "reindex_start_minute": None,
                "reindex_timezone": None,
                "chunk_size": 500,
            },
        )
        self.assertIsNot(updated["configSnapshot"], original_snapshot)
        self.assertEqual(
            original_snapshot,
            {
                "reindex_interval_hours": 24,
                "reindex_start_minute": 180,
                "reindex_timezone": "America/Denver",
                "chunk_size": 500,
            },
        )

    async def test_enable_index_reuses_credentials_timestamp_and_normalizes_existing_snapshot(self) -> None:
        repository = GitWebhookRepository()
        original_created_at = _utcnow()
        original_secret = "enc::secret"
        original_snapshot = {
            "reindex_interval_hours": 24,
            "reindex_start_minute": 180,
            "reindex_timezone": "America/Denver",
            "branch": "main",
        }
        row = _target_row(
            webhookSecret=original_secret,
            webhookPaused=True,
            webhookCreatedAt=original_created_at,
            configSnapshot=original_snapshot,
        )
        db = FakeLookupDb(row, tx=FakeTransaction(execute_results=[1]))
        db.indexmetadata = _LookupDelegate(row)
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db):
            response = await repository.enable_index("git-index", "https://ragtime.example")
        updated = db.indexmetadata.updated[-1]
        self.assertTrue(response.enabled)
        self.assertFalse(updated["webhookPaused"])
        self.assertEqual(
            updated["configSnapshot"],
            {
                "reindex_interval_hours": 0,
                "reindex_start_minute": None,
                "reindex_timezone": None,
                "branch": "main",
            },
        )
        self.assertEqual(row.webhookId, "wh_123")
        self.assertEqual(row.webhookSecret, original_secret)
        self.assertEqual(row.webhookCreatedAt, original_created_at)
        self.assertEqual(
            original_snapshot,
            {
                "reindex_interval_hours": 24,
                "reindex_start_minute": 180,
                "reindex_timezone": "America/Denver",
                "branch": "main",
            },
        )

    async def test_enable_workspace_does_not_write_config_snapshot(self) -> None:
        repository = GitWebhookRepository()
        row = _target_row(
            id="workspace-1",
            webhookId=None,
            webhookSecret=None,
            scmWebhookId=None,
            scmWebhookSecret=None,
            scmWebhookCreatedAt=None,
            scmProvider="github",
            scmGitBranch="main",
            configSnapshot={
                "reindex_interval_hours": 24,
                "reindex_start_minute": 180,
                "reindex_timezone": "America/Denver",
            },
        )
        db = FakeLookupDb(row, target_type=GitWebhookTargetType.WORKSPACE_SCM)
        db.workspace = _LookupDelegate(row)
        with (
            mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db),
            mock.patch("ragtime.git_webhooks.repository.encrypt_secret", side_effect=lambda value: f"enc::{value}"),
        ):
            response = await repository.enable_workspace("workspace-1", "https://ragtime.example")
        updated = db.workspace.updated[-1]
        self.assertTrue(response.enabled)
        self.assertNotIn("configSnapshot", updated)

    async def test_claim_latest_pending_refuses_when_processing_delivery_exists(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(claim_rows=[])
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            delivery = await repository.claim_latest_pending(GitWebhookTargetType.GIT_INDEX, "index-id")
        self.assertIsNone(delivery)
        self.assertTrue(any("status = 'processing'" in query for query in tx.executed))

    async def test_latest_queue_sql_uses_id_desc_tie_breakers(self) -> None:
        repository = GitWebhookRepository()
        claim_tx = FakeTransaction(claim_rows=[])
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(claim_tx)):
            await repository.claim_latest_pending(GitWebhookTargetType.GIT_INDEX, "index-id")
        claim_query = next(query for query in claim_tx.executed if "SET status = 'processing'" in query)
        self.assertIn("ORDER BY latest.received_at DESC, latest.id DESC", claim_query)

        skip_query = repository._skip_pending_sql(target_type=GitWebhookTargetType.GIT_INDEX, target_id="index-id")
        self.assertIn("ORDER BY received_at DESC, id DESC", skip_query)

        latest_query = repository._latest_pending_sql(target_type=GitWebhookTargetType.GIT_INDEX, target_id="index-id")
        self.assertIn("ORDER BY received_at DESC, id DESC", latest_query)

        prune_tx = FakeTransaction(query_results=[[_delivery_row(id="processing-1", status="processing", provider_delivery_id="provider-2")]])
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(prune_tx)):
            await repository.complete(
                "processing-1",
                status=GitWebhookDeliveryStatus.COMPLETED,
                message="done",
                index_job_id="job-1",
            )
        prune_query = next(query for query in prune_tx.executed if query.lstrip().startswith("DELETE FROM git_webhook_deliveries"))
        self.assertIn("ORDER BY older.received_at DESC, older.id DESC", prune_query)

    async def test_defer_claim_marks_delivery_skipped_when_newer_pending_exists(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(
            query_results=[[_delivery_row(id="delivery-1", status="processing", received_at=_utcnow())]],
            pending_rows=[_delivery_row(id="newer-pending", status="pending", received_at=_utcnow())],
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            status = await repository.defer_claim("delivery-1")
        self.assertEqual(status, GitWebhookDeliveryStatus.SKIPPED)

    async def test_defer_claim_prunes_when_marking_delivery_skipped(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(
            query_results=[[_delivery_row(id="delivery-1", status="processing", received_at=_utcnow())]],
            pending_rows=[_delivery_row(id="pending-2", status="pending", received_at=_utcnow())],
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            status = await repository.defer_claim("delivery-1")
        self.assertEqual(status, GitWebhookDeliveryStatus.SKIPPED)
        prune_queries = [query for query in tx.executed if query.lstrip().startswith("DELETE FROM git_webhook_deliveries")]
        self.assertEqual(len(prune_queries), 1)
        self.assertTrue(any("pg_advisory_xact_lock" in query for query in tx.executed))

    async def test_disable_target_skips_pending_rows(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(execute_results=[1, 1])
        row = _target_row(webhookId="wh_123", webhookSecret="enc::secret", webhookPaused=True)
        db = FakeLookupDb(row, tx=tx)
        db.indexmetadata = _LookupDelegate(row)
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db):
            await repository.disable_index("git-index")
        updated = db.indexmetadata.updated[-1]
        self.assertEqual(updated["webhookId"], None)
        self.assertEqual(updated["webhookSecret"], None)
        self.assertFalse(updated["webhookPaused"])
        prune_queries = [query for query in tx.executed if query.lstrip().startswith("DELETE FROM git_webhook_deliveries")]
        self.assertEqual(len(prune_queries), 1)

    async def test_defer_claim_resets_to_pending_when_no_other_pending_exists(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(
            query_results=[[_delivery_row(id="delivery-1", status="processing", received_at=_utcnow())]],
            pending_rows=[],
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            status = await repository.defer_claim("delivery-1")
        self.assertEqual(status, GitWebhookDeliveryStatus.PENDING)
        self.assertTrue(any("SET status = 'pending'" in query for query in tx.executed))

    async def test_defer_claim_marks_skipped_when_equal_or_older_pending_exists(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(
            query_results=[[_delivery_row(id="delivery-1", status="processing", received_at=_utcnow())]],
            pending_rows=[_delivery_row(id="older-pending", status="pending", received_at=_utcnow())],
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            status = await repository.defer_claim("delivery-1")
        self.assertEqual(status, GitWebhookDeliveryStatus.SKIPPED)
        self.assertTrue(any("status = 'skipped'" in query for query in tx.executed))

    async def test_defer_claim_unique_violation_recovers_to_skipped(self) -> None:
        repository = GitWebhookRepository()
        tx = FakeTransaction(
            query_results=[[_delivery_row(id="delivery-1", status="processing", received_at=_utcnow())]],
            pending_rows=[_delivery_row(id="winner", status="pending", received_at=_utcnow())],
            insert_exception=UniqueViolationError({}),
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            status = await repository.defer_claim("delivery-1")
        self.assertEqual(status, GitWebhookDeliveryStatus.SKIPPED)

    async def test_record_ignored_returns_existing_provider_delivery(self) -> None:
        repository = GitWebhookRepository()
        existing = _delivery_row(id="ignored-1", status="ignored", provider_delivery_id="provider-1")
        tx = FakeTransaction(duplicate_rows=[existing])
        target = GitWebhookTarget(
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id="index-id",
            webhook_id="wh_123",
            secret="plain",
        )
        event = GitPushEvent(provider_delivery_id="provider-1", event_name="push", branch="main", head_commit="abc123")
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            delivery = await repository.record_ignored(target, event, "ignored")
        self.assertEqual(delivery.id, "ignored-1")
        self.assertFalse(any("INSERT INTO git_webhook_deliveries" in query for query in tx.executed))
        self.assertTrue(any("pg_advisory_xact_lock" in query for query in tx.executed))

    async def test_record_ignored_recovers_unique_violation_with_winning_row(self) -> None:
        repository = GitWebhookRepository()
        winner = _delivery_row(id="ignored-winner", status="ignored", provider_delivery_id="provider-1")
        tx = FakeTransaction(duplicate_query_results=[[], [winner]], insert_exception=UniqueViolationError({}))
        target = GitWebhookTarget(
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id="index-id",
            webhook_id="wh_123",
            secret="plain",
        )
        event = GitPushEvent(provider_delivery_id="provider-1", event_name="push", branch="main", head_commit="abc123")
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            delivery = await repository.record_ignored(target, event, "ignored")
        self.assertEqual(delivery.id, "ignored-winner")

    async def test_list_recoverable_returns_processing_rows(self) -> None:
        repository = GitWebhookRepository()
        rows = [
            _delivery_row(id="delivery-1", status="processing", started_at=_utcnow()),
            _delivery_row(id="delivery-2", status="processing", started_at=_utcnow()),
        ]
        tx = FakeTransaction(recoverable_rows=rows)
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            recoverable = await repository.list_recoverable()
        self.assertEqual([item.id for item in recoverable], ["delivery-1", "delivery-2"])

    async def test_reset_processing_to_pending_skips_when_pending_winner_exists(self) -> None:
        repository = GitWebhookRepository()
        processing = _delivery_row(id="processing-1", status="processing", received_at=_utcnow())
        tx = FakeTransaction(
            query_results=[[processing]],
            reset_pending_rows=[_delivery_row(id="pending-winner", status="pending", received_at=_utcnow())],
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            status = await repository.reset_processing_to_pending("processing-1")
        self.assertEqual(status, GitWebhookDeliveryStatus.SKIPPED)
        self.assertTrue(any("pg_advisory_xact_lock" in query for query in tx.executed))
        self.assertTrue(any("status = 'skipped'" in query for query in tx.executed))
        prune_queries = [query for query in tx.executed if query.lstrip().startswith("DELETE FROM git_webhook_deliveries")]
        self.assertEqual(len(prune_queries), 1)

    async def test_reset_processing_to_pending_unique_violation_rereads_pending_winner(self) -> None:
        repository = GitWebhookRepository()
        processing = _delivery_row(id="processing-1", status="processing", received_at=_utcnow())
        tx = FakeTransaction(
            query_results=[[processing], [{"status": "skipped"}]],
            reset_pending_rows=[_delivery_row(id="pending-winner", status="pending", received_at=_utcnow())],
            reset_update_exception=UniqueViolationError({}),
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            status = await repository.reset_processing_to_pending("processing-1")
        self.assertEqual(status, GitWebhookDeliveryStatus.SKIPPED)

    async def test_reset_processing_to_pending_recovers_to_pending_without_winner(self) -> None:
        repository = GitWebhookRepository()
        processing = _delivery_row(id="processing-1", status="processing", received_at=_utcnow())
        tx = FakeTransaction(query_results=[[processing]], reset_pending_rows=[])
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            status = await repository.reset_processing_to_pending("processing-1")
        self.assertEqual(status, GitWebhookDeliveryStatus.PENDING)

    async def test_unique_violation_rereads_surviving_pending_row(self) -> None:
        repository = GitWebhookRepository()
        winner = _delivery_row(id="winner", provider_delivery_id="delivery-2")
        tx = FakeTransaction(
            duplicate_rows=[winner],
            insert_exception=UniqueViolationError({}),
        )
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=FakeDb(tx)):
            delivery = await repository.enqueue_push(
                target_type=GitWebhookTargetType.GIT_INDEX,
                target_id="index-id",
                provider_delivery_id="delivery-2",
                event_name="push",
                branch="main",
                head_commit="abc123",
            )
        self.assertEqual(delivery.id, "winner")

    async def test_concurrent_enqueue_push_serializes_to_one_pending_and_one_skipped(self) -> None:
        repository = GitWebhookRepository()
        db = ConcurrentQueueDb()
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db):
            first, second = await asyncio.gather(
                repository.enqueue_push(
                    target_type=GitWebhookTargetType.GIT_INDEX,
                    target_id="index-id",
                    provider_delivery_id="delivery-1",
                    event_name="push",
                    branch="main",
                    head_commit="abc123",
                ),
                repository.enqueue_push(
                    target_type=GitWebhookTargetType.GIT_INDEX,
                    target_id="index-id",
                    provider_delivery_id="delivery-2",
                    event_name="push",
                    branch="main",
                    head_commit="def456",
                ),
            )
        self.assertNotEqual(first.id, second.id)
        statuses = sorted(row["status"] for row in db.deliveries)
        self.assertEqual(statuses, ["pending", "skipped"])

    async def test_concurrent_duplicate_provider_delivery_returns_same_row(self) -> None:
        repository = GitWebhookRepository()
        db = ConcurrentQueueDb()
        with mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db):
            first, second = await asyncio.gather(
                repository.enqueue_push(
                    target_type=GitWebhookTargetType.GIT_INDEX,
                    target_id="index-id",
                    provider_delivery_id="same-id",
                    event_name="push",
                    branch="main",
                    head_commit="abc123",
                ),
                repository.enqueue_push(
                    target_type=GitWebhookTargetType.GIT_INDEX,
                    target_id="index-id",
                    provider_delivery_id="same-id",
                    event_name="push",
                    branch="main",
                    head_commit="abc123",
                ),
            )
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(db.deliveries), 1)

    async def test_concurrent_enable_index_returns_plaintext_once(self) -> None:
        repository = GitWebhookRepository()
        db = ConcurrentEnableDb()
        with (
            mock.patch("ragtime.git_webhooks.repository.get_db", return_value=db),
            mock.patch("ragtime.git_webhooks.repository.encrypt_secret", side_effect=lambda value: f"enc::{value}"),
        ):
            first, second = await asyncio.gather(
                repository.enable_index("git-index", "https://ragtime.example"),
                repository.enable_index("git-index", "https://ragtime.example"),
            )
        self.assertEqual(db.last_lock_key, "git-webhook:git_index:index-id")
        self.assertEqual(sorted([first.secret is None, second.secret is None]), [False, True])
        self.assertTrue(bool(db.row.webhookSecret))


if __name__ == "__main__":
    unittest.main()
