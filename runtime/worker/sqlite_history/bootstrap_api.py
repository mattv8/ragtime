"""Authenticated bootstrap routes shared by manager and worker surfaces."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .bootstrap import BootstrapManager


class BootstrapRequest(BaseModel):
    run_id: str
    user_id: str = Field(min_length=1)
    workspace_ids: list[str] | None = None


class ResumeRequest(BaseModel):
    retry_failed: bool = False


def bootstrap_router(prefix: str, auth: Any, coordinator: Callable[[], Any]) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["Runtime SQLite History Bootstrap"])

    def manager() -> BootstrapManager:
        return coordinator().sqlite_history_bootstrap_manager()

    @router.get("/sqlite-history/bootstrap/inventory")
    async def inventory(
        workspace_id: list[str] | None = Query(default=None),
        _auth: None = auth,
    ) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(manager().inventory, workspace_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid workspace ID") from exc

    @router.post("/sqlite-history/bootstrap", status_code=202)
    async def accept(payload: BootstrapRequest, _auth: None = auth) -> dict[str, Any]:
        return await manager().accept(payload.run_id, payload.user_id, payload.workspace_ids)

    @router.get("/sqlite-history/bootstrap/{run_id}")
    async def status(run_id: str, _auth: None = auth) -> dict[str, Any]:
        return await manager().get(run_id)

    @router.post("/sqlite-history/bootstrap/{run_id}/resume")
    async def resume(run_id: str, payload: ResumeRequest, _auth: None = auth) -> dict[str, Any]:
        return await manager().resume(run_id, payload.retry_failed)

    @router.post("/sqlite-history/bootstrap/{run_id}/cancel")
    async def cancel(run_id: str, _auth: None = auth) -> dict[str, Any]:
        return await manager().cancel(run_id)

    return router
