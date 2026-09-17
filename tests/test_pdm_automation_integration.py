"""Opt-in PostgreSQL checks for PDM automation transaction invariants."""

import asyncio
import json
import os
import unittest
import uuid
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from prisma import Prisma

from ragtime.core.encryption import encrypt_secret
from ragtime.core.sql import sql_quote_literal
from ragtime.indexer.models import ToolType
from ragtime.indexer.pdm_service import PdmIndexerService
from ragtime.pdm_automation.repository import PdmAutomationRepository
from ragtime.pdm_automation.service import PdmAutomationService


@unittest.skipUnless(os.getenv("RAGTIME_TEST_DATABASE_URL"), "requires isolated PostgreSQL")
class PdmAutomationPostgresIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db = Prisma(datasource={"url": os.environ["RAGTIME_TEST_DATABASE_URL"]})
        await self.db.connect()
        self.tool_id = str(uuid.uuid4())
        self.repo = PdmAutomationRepository()
        self.indexer = PdmIndexerService()
        self.config = {"host": "pdm.example", "user": "reader", "database": "vault"}
        await self.db.execute_raw(
            f"INSERT INTO tool_configs(id,name,tool_type,description,connection_config) VALUES "
            f"({sql_quote_literal(self.tool_id)},'pdm-test','solidworks_pdm','', "
            f"{sql_quote_literal(json.dumps(self.config))}::jsonb)"
        )
        self.db_patch = ExitStack()
        self.db_patch.enter_context(mock.patch("ragtime.indexer.pdm_service.get_db", mock.AsyncMock(return_value=self.db)))
        self.db_patch.enter_context(mock.patch("ragtime.pdm_automation.repository.get_db", mock.AsyncMock(return_value=self.db)))
        self.db_patch.enter_context(mock.patch.object(self.indexer, "_process_index", mock.AsyncMock()))

    async def asyncTearDown(self) -> None:
        self.db_patch.close()
        for task in self.indexer._running_tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.indexer._running_tasks.values(), return_exceptions=True)
        await self.db.execute_raw(f"DELETE FROM tool_configs WHERE id={sql_quote_literal(self.tool_id)}")
        await self.db.disconnect()

    async def _insert_pending(self, *, webhook: bool = True, generation: int = 1) -> None:
        await self.db.execute_raw(
            f"INSERT INTO pdm_automation_state(tool_config_id,pending_webhook,pending_generation,"
            f"first_pending_at,last_received_at) VALUES ({sql_quote_literal(self.tool_id)},"
            f"{'TRUE' if webhook else 'FALSE'},{generation},NOW()-interval '61 seconds',NOW()-interval '61 seconds')"
        )

    async def test_actual_trigger_admission_links_job_in_same_transaction(self) -> None:
        await self._insert_pending()
        service = PdmAutomationService(
            repo=self.repo,
            tools=SimpleNamespace(
                get_tool_config=mock.AsyncMock(
                    return_value=SimpleNamespace(
                        id=self.tool_id,
                        name="pdm-test",
                        tool_type=ToolType.SOLIDWORKS_PDM,
                        enabled=True,
                        connection_config=self.config,
                    )
                )
            ),
            indexer=self.indexer,
        )
        with (
            mock.patch("ragtime.pdm_automation.service.ensure_pgvector_extension", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.pdm_automation.service.validate_embedding_provider", mock.AsyncMock(return_value=SimpleNamespace(valid=True))),
        ):
            await service._dispatch_tool(self.tool_id)
        jobs = await self.db.query_raw(f"SELECT id FROM pdm_index_jobs WHERE tool_config_id={sql_quote_literal(self.tool_id)}")
        state = await self.db.query_raw(
            f"SELECT active_job_id,pending_webhook FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(self.tool_id)}"
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(state[0]["active_job_id"], jobs[0]["id"])
        self.assertFalse(state[0]["pending_webhook"])
        await service.stop()

    async def test_manual_job_and_webhook_leave_one_job_and_pending_followup(self) -> None:
        await self.indexer.trigger_index(self.tool_id, self.config)
        await self.db.execute_raw(
            f"INSERT INTO pdm_automation_state(tool_config_id,webhook_id,webhook_secret) VALUES "
            f"({sql_quote_literal(self.tool_id)},'hook',{sql_quote_literal(encrypt_secret('secret'))})"
        )
        tools = SimpleNamespace(
            get_tool_config=mock.AsyncMock(
                return_value=SimpleNamespace(id=self.tool_id, name="pdm-test", tool_type=ToolType.SOLIDWORKS_PDM, enabled=True, connection_config=self.config)
            )
        )
        service = PdmAutomationService(repo=self.repo, tools=tools, indexer=self.indexer)
        with (
            mock.patch("ragtime.pdm_automation.service.ensure_pgvector_extension", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.pdm_automation.service.validate_embedding_provider", mock.AsyncMock(return_value=SimpleNamespace(valid=True))),
        ):
            await asyncio.gather(
                self.repo.accept_authenticated_webhook("hook", "secret", "event-1"),
                service._dispatch_tool(self.tool_id),
            )
        jobs = await self.db.query_raw(f"SELECT id FROM pdm_index_jobs WHERE tool_config_id={sql_quote_literal(self.tool_id)}")
        state = await self.db.query_raw(
            f"SELECT pending_webhook,active_job_id FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(self.tool_id)}"
        )
        self.assertEqual(len(jobs), 1)
        self.assertTrue(state[0]["pending_webhook"])
        self.assertIsNone(state[0]["active_job_id"])
        await service.stop()

    async def test_claim_rollback_preserves_pending_event(self) -> None:
        await self._insert_pending()
        with self.assertRaises(RuntimeError):
            async with self.db.tx() as tx:
                self.assertEqual(await self.repo.claim_for_admission(tx, self.tool_id, "job-lost"), 1)
                raise RuntimeError("simulate crash before job insert")
        rows = await self.db.query_raw(f"SELECT pending_webhook,active_job_id FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(self.tool_id)}")
        self.assertTrue(rows[0]["pending_webhook"])
        self.assertIsNone(rows[0]["active_job_id"])

    async def test_reconcile_terminal_link_retains_newer_pending_generation(self) -> None:
        job = await self.indexer.trigger_index(self.tool_id, self.config)
        await self.db.execute_raw(f"UPDATE pdm_index_jobs SET status='failed' WHERE id={sql_quote_literal(job.id)}")
        await self.db.execute_raw(
            f"INSERT INTO pdm_automation_state(tool_config_id,pending_webhook,pending_generation,claimed_generation,active_job_id) VALUES "
            f"({sql_quote_literal(self.tool_id)},TRUE,2,1,{sql_quote_literal(job.id)})"
        )
        await self.repo.reconcile()
        state = await self.db.query_raw(
            f"SELECT pending_webhook,pending_generation,active_job_id,last_attempt_at FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(self.tool_id)}"
        )
        self.assertTrue(state[0]["pending_webhook"])
        self.assertEqual(state[0]["pending_generation"], 2)
        self.assertIsNone(state[0]["active_job_id"])
        self.assertIsNotNone(state[0]["last_attempt_at"])

    async def test_index_reset_preserves_webhook_credentials(self) -> None:
        secret = encrypt_secret("not-returned")
        await self.db.execute_raw(
            f"INSERT INTO pdm_automation_state(tool_config_id,webhook_id,webhook_secret,pending_webhook,pending_schedule,pending_generation,active_job_id,last_error) VALUES "
            f"({sql_quote_literal(self.tool_id)},'hook',{sql_quote_literal(secret)},TRUE,TRUE,3,'old-job','failure')"
        )
        await self.repo.reset_index_state(self.tool_id)
        state = await self.db.query_raw(
            f"SELECT webhook_id,webhook_secret,pending_webhook,pending_schedule,active_job_id,last_error FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(self.tool_id)}"
        )
        self.assertEqual(state[0]["webhook_id"], "hook")
        self.assertEqual(state[0]["webhook_secret"], secret)
        self.assertFalse(state[0]["pending_webhook"])
        self.assertFalse(state[0]["pending_schedule"])
        self.assertIsNone(state[0]["active_job_id"])
        self.assertIsNone(state[0]["last_error"])

    async def test_event_during_slow_failed_preflight_remains_pending(self) -> None:
        await self.db.execute_raw(
            f"INSERT INTO pdm_automation_state(tool_config_id,webhook_id,webhook_secret,pending_webhook,pending_generation,first_pending_at,last_received_at) VALUES "
            f"({sql_quote_literal(self.tool_id)},'hook',{sql_quote_literal(encrypt_secret('secret'))},TRUE,1,NOW()-interval '61 seconds',NOW()-interval '61 seconds')"
        )
        tools = SimpleNamespace(
            get_tool_config=mock.AsyncMock(
                return_value=SimpleNamespace(id=self.tool_id, name="pdm-test", tool_type=ToolType.SOLIDWORKS_PDM, enabled=True, connection_config=self.config)
            )
        )
        service = PdmAutomationService(repo=self.repo, tools=tools, indexer=self.indexer)
        preflight_started = asyncio.Event()
        allow_failure = asyncio.Event()

        async def slow_preflight(**_kwargs):
            preflight_started.set()
            await allow_failure.wait()
            return False

        with mock.patch("ragtime.pdm_automation.service.ensure_pgvector_extension", slow_preflight):
            dispatch = asyncio.create_task(service._dispatch_tool(self.tool_id))
            await preflight_started.wait()
            self.assertEqual(await self.repo.accept_authenticated_webhook("hook", "secret", "event-2"), "accepted")
            allow_failure.set()
            await dispatch
        state = await self.db.query_raw(
            f"SELECT pending_webhook,pending_generation,last_attempt_at,last_error FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(self.tool_id)}"
        )
        self.assertTrue(state[0]["pending_webhook"])
        self.assertEqual(state[0]["pending_generation"], 2)
        self.assertIsNotNone(state[0]["last_attempt_at"])
        self.assertEqual(state[0]["last_error"], "pgvector extension is not available.")
