"""Runtime-owned due claims and bounded Restic housekeeping."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .storage import AsyncRepositoryGate


class RuntimeHistoryMaintenance:
    """Small lifecycle facade; catalog policy remains in ``service.py``."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._next_repository_check = 0.0
        self._next_repository_prune = 0.0

    async def claim_due(self, workspace_ids: list[str]) -> list[dict[str, str]]:
        return await self.service.claim_due(workspace_ids)

    async def ack_due(self, workspace_id: str, occurrence_id: str, job_id: str) -> bool:
        return await self.service.ack_due(workspace_id, occurrence_id, job_id)

    async def run_once(self, *, read_data: bool = False, max_repack_size: int = 64 * 1024 * 1024, capacity_pressure: bool = False) -> None:
        await self.service.run_maintenance_once()
        repository = getattr(self.service, "repository", None)
        if repository is None:
            return
        gate = await AsyncRepositoryGate.try_exclusive(self.service._runtime.root)
        if gate is None:
            # This is a periodic best-effort pass.  Crucially, do not leave a
            # queued writer behind a long guarded restore: that would prevent
            # all new shared history reads until the guard finishes.
            return
        try:
            await self.service.drain_restic_forget_tombstones()
            now = time.monotonic()
            if read_data or now >= self._next_repository_check:
                await repository.check(read_data=read_data)
                self._next_repository_check = now + 24 * 60 * 60
            if capacity_pressure or now >= self._next_repository_prune:
                await repository.prune(max_repack_size=max_repack_size)
                self._next_repository_prune = now + 24 * 60 * 60
        finally:
            await gate.aclose()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._loop(), name="runtime-history-restic-maintenance")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stopping:
            await self.run_once()
            await asyncio.sleep(300)
