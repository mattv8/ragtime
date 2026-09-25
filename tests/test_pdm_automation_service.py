import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer.models import PdmIndexStatus, ToolType
from ragtime.pdm_automation.repository import PdmAutomationRepository
from ragtime.pdm_automation.service import PdmAutomationService


class PdmAutomationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_uses_real_repository_timestamp_to_compare_and_dispatch(self):
        now = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
        db = SimpleNamespace(query_raw=mock.AsyncMock())
        repo = PdmAutomationRepository()
        repo.clear_schedule_pending = mock.AsyncMock()
        repo.mark_schedule_pending = mock.AsyncMock()
        repo.pending_tool_ids = mock.AsyncMock(side_effect=[[], [], ["pdm-1"]])
        tools = SimpleNamespace(
            list_tool_configs=mock.AsyncMock(
                return_value=[
                    SimpleNamespace(
                        id="pdm-1",
                        tool_type=ToolType.SOLIDWORKS_PDM,
                        connection_config={"host": "h", "user": "u", "database": "d", "reindex_interval_hours": 1},
                    )
                ]
            )
        )
        indexer = SimpleNamespace(get_active_job=mock.AsyncMock(return_value=None), get_latest_job=mock.AsyncMock())
        service = PdmAutomationService(repo=repo, tools=tools, indexer=indexer)
        service._dispatch_tool = mock.AsyncMock()

        async def dispatch(state_attempt, latest):
            db.query_raw.return_value = [{"last_attempt_at": state_attempt}] if state_attempt else []
            indexer.get_latest_job.return_value = latest
            await service.dispatch_once()

        with (
            mock.patch("ragtime.pdm_automation.repository.get_db", mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.pdm_automation.service.is_anchored_schedule_due", return_value=None),
            mock.patch("ragtime.pdm_automation.service.utc_now", return_value=now),
        ):
            await dispatch(
                "2026-09-25T11:30:00Z",
                SimpleNamespace(status=PdmIndexStatus.COMPLETED, completed_at=now - timedelta(hours=2)),
            )
            repo.mark_schedule_pending.assert_not_awaited()

            await dispatch("2026-09-25T11:59:30Z", None)
            repo.mark_schedule_pending.assert_not_awaited()

            await dispatch(
                "2026-09-25T09:00:00Z",
                SimpleNamespace(status=PdmIndexStatus.COMPLETED, completed_at=now - timedelta(hours=2)),
            )
        repo.mark_schedule_pending.assert_awaited_once_with("pdm-1")
        service._dispatch_tool.assert_awaited_once_with("pdm-1")

    async def test_disabled_schedule_does_not_enqueue(self):
        repo = SimpleNamespace(
            mark_schedule_pending=mock.AsyncMock(), clear_schedule_pending=mock.AsyncMock(), pending_tool_ids=mock.AsyncMock(return_value=[])
        )
        tools = SimpleNamespace(
            list_tool_configs=mock.AsyncMock(
                return_value=[SimpleNamespace(id="pdm-1", tool_type=ToolType.SOLIDWORKS_PDM, connection_config={"host": "h", "user": "u", "database": "d"})]
            )
        )
        indexer = SimpleNamespace(get_latest_job=mock.AsyncMock())
        await PdmAutomationService(repo=repo, tools=tools, indexer=indexer).dispatch_once()
        repo.mark_schedule_pending.assert_not_awaited()

    async def test_preflight_failure_consumes_observed_generation_without_hot_loop(self):
        repo = SimpleNamespace(
            clear_schedule_pending=mock.AsyncMock(),
            ready_generation=mock.AsyncMock(return_value=4),
            consume_preflight_failure=mock.AsyncMock(),
        )
        tools = SimpleNamespace(
            get_tool_config=mock.AsyncMock(
                return_value=SimpleNamespace(
                    id="pdm-1",
                    name="PDM",
                    tool_type=ToolType.SOLIDWORKS_PDM,
                    enabled=True,
                    connection_config={"host": "h", "user": "u", "database": "d"},
                )
            )
        )
        indexer = SimpleNamespace(trigger_index=mock.AsyncMock())
        with (
            mock.patch("ragtime.pdm_automation.service.ensure_pgvector_extension", mock.AsyncMock(return_value=False)),
            mock.patch("ragtime.pdm_automation.service.validate_embedding_provider", mock.AsyncMock()) as validate,
        ):
            await PdmAutomationService(repo=repo, tools=tools, indexer=indexer)._dispatch_tool("pdm-1")
        repo.consume_preflight_failure.assert_awaited_once_with("pdm-1", 4, "pgvector extension is not available.")
        indexer.trigger_index.assert_not_awaited()
        validate.assert_not_awaited()

    async def test_schedule_uses_durable_preflight_attempt_to_avoid_hot_loop(self):
        from ragtime.core.datetimes import utc_now

        repo = SimpleNamespace(
            clear_schedule_pending=mock.AsyncMock(),
            mark_schedule_pending=mock.AsyncMock(),
            last_attempt_at=mock.AsyncMock(return_value=utc_now() - timedelta(seconds=1)),
            pending_tool_ids=mock.AsyncMock(return_value=[]),
        )
        tools = SimpleNamespace(
            list_tool_configs=mock.AsyncMock(
                return_value=[
                    SimpleNamespace(
                        id="pdm-1",
                        tool_type=ToolType.SOLIDWORKS_PDM,
                        connection_config={"host": "h", "user": "u", "database": "d", "reindex_interval_hours": 1},
                    )
                ]
            )
        )
        indexer = SimpleNamespace(
            get_active_job=mock.AsyncMock(return_value=None),
            get_latest_job=mock.AsyncMock(return_value=None),
        )
        with mock.patch("ragtime.pdm_automation.service.is_anchored_schedule_due", return_value=None):
            await PdmAutomationService(repo=repo, tools=tools, indexer=indexer).dispatch_once()
        repo.mark_schedule_pending.assert_not_awaited()
