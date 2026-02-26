from __future__ import annotations

from fastapi import FastAPI

from runtime.manager.models import RuntimeSessionResponse, StartSessionRequest
from runtime.manager.service import SessionManager


def create_app() -> FastAPI:
    application = FastAPI(title="Ragtime User Space Runtime Manager", version="0.1.0")
    manager = SessionManager()

    @application.on_event("startup")
    async def on_startup() -> None:
        await manager.startup()

    @application.on_event("shutdown")
    async def on_shutdown() -> None:
        await manager.shutdown()

    @application.get("/health")
    async def health() -> dict[str, str]:
        pool = await manager.pool_status()
        return {
            "status": "ok",
            "warm_slots": str(pool["warm_slots"]),
            "leased_slots": str(pool["leased_slots"]),
        }

    @application.post("/sessions/start", response_model=RuntimeSessionResponse)
    async def start_session(request: StartSessionRequest) -> RuntimeSessionResponse:
        return await manager.start_session(request)

    @application.get(
        "/sessions/{provider_session_id}", response_model=RuntimeSessionResponse
    )
    async def get_session(provider_session_id: str) -> RuntimeSessionResponse:
        return await manager.get_session(provider_session_id)

    @application.post(
        "/sessions/{provider_session_id}/stop", response_model=RuntimeSessionResponse
    )
    async def stop_session(provider_session_id: str) -> RuntimeSessionResponse:
        return await manager.stop_session(provider_session_id)

    @application.post(
        "/sessions/{provider_session_id}/restart", response_model=RuntimeSessionResponse
    )
    async def restart_session(provider_session_id: str) -> RuntimeSessionResponse:
        return await manager.restart_devserver(provider_session_id)

    return application


app = create_app()
