from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from runtime.auth import ManagerAuth
from runtime.manager.models import (
    RuntimeFileReadResponse,
    RuntimeFileWriteRequest,
    RuntimeManagerHealthResponse,
    RuntimePtyUrlResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    RuntimeSessionResponse,
    StartSessionRequest,
)
from runtime.manager.service import SessionManager


def create_app() -> FastAPI:
    manager = SessionManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await manager.startup()
        yield
        await manager.shutdown()

    application = FastAPI(
        title="Ragtime User Space Runtime Manager",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=RuntimeManagerHealthResponse)
    async def health() -> RuntimeManagerHealthResponse:
        pool = await manager.pool_status()
        return RuntimeManagerHealthResponse(
            status="ok",
            workers_total=pool["workers_total"],
            workers_leased=pool["workers_leased"],
            active_sessions=pool["active_sessions"],
            max_sessions=pool["max_sessions"],
            sessions=pool["sessions"],
        )

    @application.post("/sessions/start", response_model=RuntimeSessionResponse)
    async def start_session(
        request: StartSessionRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.start_session(request)

    @application.get(
        "/sessions/{provider_session_id}", response_model=RuntimeSessionResponse
    )
    async def get_session(
        provider_session_id: str,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.get_session(provider_session_id)

    @application.post(
        "/sessions/{provider_session_id}/stop", response_model=RuntimeSessionResponse
    )
    async def stop_session(
        provider_session_id: str,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.stop_session(provider_session_id)

    @application.post(
        "/sessions/{provider_session_id}/restart", response_model=RuntimeSessionResponse
    )
    async def restart_session(
        provider_session_id: str,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.restart_devserver(provider_session_id)

    @application.get(
        "/sessions/{provider_session_id}/pty/ws-url",
        response_model=RuntimePtyUrlResponse,
    )
    async def get_pty_ws_url(
        provider_session_id: str,
        _auth: None = ManagerAuth,
    ) -> RuntimePtyUrlResponse:
        return await manager.get_pty_websocket_url(provider_session_id)

    @application.get(
        "/sessions/{provider_session_id}/fs/{file_path:path}",
        response_model=RuntimeFileReadResponse,
    )
    async def read_file(
        provider_session_id: str,
        file_path: str,
        _auth: None = ManagerAuth,
    ) -> RuntimeFileReadResponse:
        return await manager.read_file(provider_session_id, file_path)

    @application.put(
        "/sessions/{provider_session_id}/fs/{file_path:path}",
        response_model=RuntimeFileReadResponse,
    )
    async def write_file(
        provider_session_id: str,
        file_path: str,
        payload: RuntimeFileWriteRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeFileReadResponse:
        return await manager.write_file(
            provider_session_id,
            file_path,
            payload.content,
        )

    @application.delete("/sessions/{provider_session_id}/fs/{file_path:path}")
    async def delete_file(
        provider_session_id: str,
        file_path: str,
        _auth: None = ManagerAuth,
    ) -> dict[str, Any]:
        return await manager.delete_file(provider_session_id, file_path)

    @application.post(
        "/sessions/{provider_session_id}/screenshot",
        response_model=RuntimeScreenshotResponse,
    )
    async def capture_screenshot(
        provider_session_id: str,
        payload: RuntimeScreenshotRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeScreenshotResponse:
        return await manager.capture_screenshot(provider_session_id, payload)

    return application


app = create_app()
