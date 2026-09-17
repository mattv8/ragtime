from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from ragtime.core.datetimes import utc_now
from ragtime.core.logging import get_logger
from ragtime.core.scheduling import is_anchored_schedule_due
from ragtime.core.validation import validate_embedding_provider
from ragtime.indexer.models import PdmIndexStatus, SolidworksPdmConnectionConfig, ToolType
from ragtime.indexer.pdm_service import pdm_indexer
from ragtime.indexer.repository import repository
from ragtime.indexer.utils import safe_tool_name
from ragtime.indexer.vector_utils import ensure_pgvector_extension
from ragtime.pdm_automation.repository import pdm_automation_repository

logger = get_logger(__name__)


class PdmAutomationService:
    """Single-process dispatcher; DB state survives restarts and coalesces requests."""

    def __init__(self, *, repo: Any = pdm_automation_repository, tools: Any = repository, indexer: Any = pdm_indexer) -> None:
        self._repo, self._tools, self._indexer = repo, tools, indexer
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._monitor_tasks: set[asyncio.Task[None]] = set()
        self._stopping = False

    async def start(self) -> None:
        self._stopping = False
        await self._repo.reconcile()
        self._task = asyncio.create_task(self._run(), name="pdm-automation-dispatcher")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        monitors = list(self._monitor_tasks)
        for task in monitors:
            task.cancel()
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)
        self._monitor_tasks.clear()

    def wake(self) -> None:
        self._wake.set()

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self.dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("PDM automation dispatch iteration failed")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass

    async def dispatch_once(self) -> None:
        for tool in await self._tools.list_tool_configs(enabled_only=True):
            if tool.tool_type != ToolType.SOLIDWORKS_PDM or not tool.id:
                continue
            config_data = tool.connection_config or {}
            try:
                config = SolidworksPdmConnectionConfig(**config_data)
            except Exception:
                continue
            if config.reindex_interval_hours <= 0:
                await self._repo.clear_schedule_pending(tool.id)
                continue
            if await self._indexer.get_active_job(tool.id):
                continue
            if config.reindex_interval_hours > 0:
                latest = await self._indexer.get_latest_job(tool.id)
                job_attempt = (
                    latest.completed_at if latest and latest.status in {PdmIndexStatus.COMPLETED, PdmIndexStatus.FAILED, PdmIndexStatus.CANCELLED} else None
                )
                state_attempt = await self._repo.last_attempt_at(tool.id)
                last_attempt = max((value for value in (job_attempt, state_attempt) if value is not None), default=None)
                due = is_anchored_schedule_due(
                    interval_seconds=config.reindex_interval_hours * 3600,
                    start_minute=config.reindex_start_minute,
                    timezone_name=config.reindex_timezone,
                    last_run_at=last_attempt,
                )
                if due is True or (due is None and (last_attempt is None or utc_now() >= last_attempt + timedelta(hours=config.reindex_interval_hours))):
                    await self._repo.mark_schedule_pending(tool.id)
        for tool_id in await self._repo.pending_tool_ids():
            await self._dispatch_tool(tool_id)

    async def _dispatch_tool(self, tool_id: str) -> None:
        tool = await self._tools.get_tool_config(tool_id)
        if tool is None or tool.tool_type != ToolType.SOLIDWORKS_PDM or not tool.enabled:
            return
        config = tool.connection_config or {}
        if int(config.get("reindex_interval_hours", 0) or 0) <= 0:
            await self._repo.clear_schedule_pending(tool_id)
        generation = await self._repo.ready_generation(tool_id)
        if generation is None:
            return
        try:
            # These checks may contact configured providers.  Do not hold the
            # admission transaction's per-tool advisory lock while waiting.
            if not await ensure_pgvector_extension(logger_override=logger):
                await self._repo.consume_preflight_failure(tool_id, generation, "pgvector extension is not available.")
                return
            embedding_ready = await validate_embedding_provider()
            if not embedding_ready.valid:
                await self._repo.consume_preflight_failure(tool_id, generation, "Embedding provider is not ready.")
                return

            async def admit(tx, job):
                return await self._repo.claim_for_admission(tx, tool_id, job.id) is not None

            job = await self._indexer.trigger_index(tool_id, config, full_reindex=False, tool_name=safe_tool_name(tool.name) or None, admission_hook=admit)
            if job is None:
                return
            task = asyncio.create_task(self._monitor(tool_id, job.id), name=f"pdm-automation-job:{job.id}")
            self._monitor_tasks.add(task)
            task.add_done_callback(self._monitor_tasks.discard)
        except Exception as exc:
            await self._repo.consume_preflight_failure(tool_id, generation, "Automatic PDM indexing could not start.")
            logger.warning("Automatic PDM indexing could not start for tool %s: %s", tool_id, type(exc).__name__)

    async def _monitor(self, tool_id: str, job_id: str) -> None:
        while not self._stopping:
            job = await self._indexer.get_job_status(job_id)
            if job is None:
                await self._repo.finish_job(tool_id, job_id, False, "Automatic PDM indexing job disappeared.")
                return
            if job.status in {PdmIndexStatus.COMPLETED, PdmIndexStatus.FAILED, PdmIndexStatus.CANCELLED}:
                error = None if job.status == PdmIndexStatus.COMPLETED else "Automatic PDM indexing did not complete."
                await self._repo.finish_job(tool_id, job_id, job.status == PdmIndexStatus.COMPLETED, error)
                self.wake()
                return
            await asyncio.sleep(1)


pdm_automation_service = PdmAutomationService()
